#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_monotonic_time_sec(row):
    # dF/d(monotonic_sec)는 monotonic time 기반으로만 계산한다.
    for key in ("clock_monotonic_ns", "monotonic_ns"):
        value = row.get(key, None)
        if isinstance(value, (int, float)):
            return float(value) / 1e9

    msg = row.get("msg", None)
    if isinstance(msg, dict):
        for key in ("clock_monotonic_ns", "monotonic_ns"):
            value = msg.get(key, None)
            if isinstance(value, (int, float)):
                return float(value) / 1e9
    return None


def get_foot_force(row):
    msg = row.get("msg", None)
    if isinstance(msg, dict):
        ff = msg.get("foot_force", None)
        if isinstance(ff, list) and len(ff) >= 4:
            return [float(ff[0]), float(ff[1]), float(ff[2]), float(ff[3])]

    ff = row.get("foot_force", None)
    if isinstance(ff, list) and len(ff) >= 4:
        return [float(ff[0]), float(ff[1]), float(ff[2]), float(ff[3])]
    return None


def moving_average_1d(x, win):
    if win <= 1:
        return np.asarray(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    half = win // 2
    n = len(x)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        out[i] = np.mean(x[s:e])
    return out


class ReservoirSampler:
    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.buffer = np.empty((self.capacity,), dtype=np.float64)
        self.size = 0
        self.seen = 0
        self.rng = random.Random(seed)

    def add_many(self, values):
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        for v in vals:
            self.seen += 1
            if self.size < self.capacity:
                self.buffer[self.size] = float(v)
                self.size += 1
            else:
                j = self.rng.randint(1, self.seen)
                if j <= self.capacity:
                    self.buffer[j - 1] = float(v)

    def as_array(self):
        return self.buffer[: self.size].copy()


def normal_pdf(x, mean, std):
    if std <= 1e-12:
        return np.zeros_like(x)
    return (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def safe_stats(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return None
    mean = float(np.mean(x))
    std = float(np.std(x))
    centered = x - mean
    if std > 1e-12:
        skew = float(np.mean((centered / std) ** 3))
        kurt = float(np.mean((centered / std) ** 4) - 3.0)
    else:
        skew = 0.0
        kurt = 0.0
    return {
        "count": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": mean,
        "std": std,
        "p50": float(np.quantile(x, 0.5)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "skew": skew,
        "kurtosis_excess": kurt,
    }


def filter_distribution_rare_outliers(abs_values, tail_frac, log_mad_z, min_samples):
    x = np.asarray(abs_values, dtype=np.float64).reshape(-1)
    keep = np.ones(x.size, dtype=bool)
    info = {
        "applied": False,
        "reason": "",
        "n": int(x.size),
        "tail_count": 0,
        "tail_threshold": math.inf,
        "robust_upper": math.inf,
        "upper_bound": math.inf,
    }

    if x.size == 0:
        info["reason"] = "empty"
        return keep, info
    if x.size < int(min_samples):
        info["reason"] = "not_enough_samples"
        return keep, info
    if not (0.0 < float(tail_frac) < 0.5):
        info["reason"] = "invalid_tail_frac"
        return keep, info
    if log_mad_z <= 0:
        info["reason"] = "invalid_log_mad_z"
        return keep, info

    x = np.clip(x, 1e-12, None)
    log_x = np.log10(x)
    med_log = float(np.median(log_x))
    mad_log = float(np.median(np.abs(log_x - med_log)))
    if mad_log > 1e-12:
        robust_upper_log = med_log + float(log_mad_z) * 1.4826 * mad_log
        robust_upper = float(10.0 ** robust_upper_log)
    else:
        robust_upper = float(np.quantile(x, 0.999))

    tail_count = max(1, int(math.ceil(x.size * float(tail_frac))))
    k = max(0, x.size - tail_count)
    tail_threshold = float(np.partition(x, k)[k])

    upper_bound = max(robust_upper, tail_threshold)
    keep = x <= upper_bound

    info["applied"] = True
    info["tail_count"] = int(tail_count)
    info["tail_threshold"] = tail_threshold
    info["robust_upper"] = robust_upper
    info["upper_bound"] = upper_bound
    return keep, info


def normalize_abs_diff_by_divisor(abs_diff, divisor_value=0.0, divisor_percentile=0.0):
    x = np.asarray(abs_diff, dtype=np.float64).reshape(-1)
    out = np.zeros_like(x, dtype=np.float64)
    info = {
        "applied": False,
        "reason": "",
        "count": int(x.size),
        "max_value": 0.0,
        "divisor_source": "",
        "divisor_used": 0.0,
        "divisor_percentile": 0.0,
        "divisor_percentile_value": 0.0,
    }

    if x.size == 0:
        info["reason"] = "empty"
        return out, info

    max_value = float(np.max(x))
    if not math.isfinite(max_value):
        info["reason"] = "invalid_max"
        return out, info

    if divisor_value is not None and float(divisor_value) > 0:
        divisor = float(divisor_value)
        info["divisor_source"] = "user_value"
    elif divisor_percentile is not None and float(divisor_percentile) > 0:
        p = float(divisor_percentile)
        divisor = float(np.quantile(x, p / 100.0))
        info["divisor_source"] = "sample_percentile"
        info["divisor_percentile"] = p
        info["divisor_percentile_value"] = divisor
    else:
        divisor = max_value
        info["divisor_source"] = "sample_max"

    if not math.isfinite(divisor) or divisor <= 0:
        info["reason"] = "non_positive_divisor"
        return out, info

    out = x / divisor
    out = np.clip(out, 0.0, 1.0)

    info["applied"] = True
    info["max_value"] = max_value
    info["divisor_used"] = divisor
    return out, info


def normalize_signed_diff_by_divisor(diff, divisor_value=0.0, divisor_percentile=90.0):
    x = np.asarray(diff, dtype=np.float64).reshape(-1)
    out = np.zeros_like(x, dtype=np.float64)
    info = {
        "applied": False,
        "reason": "",
        "count": int(x.size),
        "mode": "signed_divisor",
        "divisor_source": "",
        "divisor_used": 0.0,
        "divisor_percentile": 0.0,
        "divisor_percentile_value": 0.0,
    }

    if x.size == 0:
        info["reason"] = "empty"
        return out, info

    if divisor_value is not None and float(divisor_value) > 0:
        divisor = float(divisor_value)
        info["divisor_source"] = "user_value"
    elif divisor_percentile is not None and float(divisor_percentile) > 0:
        p = float(divisor_percentile)
        # Keep signed diff for output, but use magnitude percentile as scale
        # so skewed-sign data does not produce near-zero divisor.
        divisor = float(np.quantile(np.abs(x), p / 100.0))
        info["divisor_source"] = "sample_abs_percentile"
        info["divisor_percentile"] = p
        info["divisor_percentile_value"] = divisor
    else:
        info["reason"] = "need_positive_divisor_or_percentile"
        return out, info

    if not math.isfinite(divisor) or divisor <= 1e-12:
        info["reason"] = "non_finite_or_zero_divisor"
        return out, info

    out = x / divisor
    out = np.clip(out, -1.0, 1.0)
    info["applied"] = True
    info["divisor_used"] = divisor
    return out, info


def preprocess_diff_model_tanh(deriv, d_force_scale):
    x = np.asarray(deriv, dtype=np.float64).reshape(-1)
    out = np.zeros_like(x, dtype=np.float64)
    info = {
        "applied": False,
        "reason": "",
        "count": int(x.size),
        "mode": "model_tanh",
        "d_force_scale": float(d_force_scale),
    }

    if x.size == 0:
        info["reason"] = "empty"
        return out, info

    s = max(float(d_force_scale), 1e-12)
    out = np.tanh(x / s)
    info["applied"] = True
    return out, info


def print_distribution_summary(name, x):
    s = safe_stats(x)
    if s is None:
        print(f"{name}: no data")
        return
    print(
        f"{name}: count={s['count']} min={s['min']:.6f} max={s['max']:.6f} "
        f"mean={s['mean']:.6f} std={s['std']:.6f} "
        f"p50={s['p50']:.6f} p95={s['p95']:.6f} p99={s['p99']:.6f} "
        f"skew={s['skew']:.4f} kurtosis_excess={s['kurtosis_excess']:.4f}"
    )


def plot_main_distributions(
    raw,
    raw_abs_deriv,
    preproc_diff,
    preproc_info,
    legacy_tanh_diff,
    legacy_tanh_info,
    raw_plot_upper_percentile,
    raw_plot_abs_min,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 4, figsize=(26, 5.2))

    # 1) raw foot_force distribution (signed)
    ax = axes[0]
    if raw.size > 0:
        q = float(np.quantile(np.abs(raw), 0.999))
        q = max(q, 1.0)
        bins = np.linspace(-q, q, 220)
        ax.hist(raw, bins=bins, density=True, alpha=0.78, color="#3A86FF", label="foot_force")
        mean = float(np.mean(raw))
        std = float(np.std(raw))
        xx = np.linspace(-q, q, 400)
        ax.plot(xx, normal_pdf(xx, mean, std), color="#FF006E", linewidth=1.7, label="normal fit")
        ax.set_title(
            "Foot Force Distribution (signed)\n"
            f"mean={mean:.2f}, std={std:.2f}, p99(|x|)={np.quantile(np.abs(raw), 0.99):.2f}"
        )
        ax.set_xlabel("foot_force")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    else:
        ax.set_title("Foot Force Distribution (no data)")
        ax.grid(True, alpha=0.25)

    # 2) raw diff distribution (no preprocessing)
    ax = axes[1]
    if raw_abs_deriv.size > 0:
        x_full = np.clip(raw_abs_deriv, 1e-8, None)
        upper_q = float(np.quantile(x_full, raw_plot_upper_percentile / 100.0))
        x = x_full[(x_full >= raw_plot_abs_min) & (x_full <= upper_q)]
        if x.size == 0:
            x = x_full[x_full <= upper_q]
        if x.size == 0:
            x = x_full

        lo = float(np.quantile(x, 0.001))
        hi = float(np.quantile(x, 0.999))
        lo = max(lo, 1e-8)
        if hi <= lo:
            hi = max(lo * 10.0, 1.0)
        bins = np.logspace(np.log10(lo), np.log10(hi), 220)
        ax.hist(x, bins=bins, density=True, alpha=0.78, color="#06D6A0")
        ax.set_xscale("log")
        ax.set_title(
            f"Raw |dF/dt| Distribution (No Preprocessing, >= {raw_plot_abs_min:.1e}, <=P{raw_plot_upper_percentile:.1f})\n"
            f"mean={np.mean(x):.2f}, p95={np.quantile(x, 0.95):.2f}, p99={np.quantile(x, 0.99):.2f}"
        )
        ax.set_xlabel("raw |dF/d(monotonic_sec)| (log x)")
        ax.set_ylabel("density")
        ax.grid(True, which="both", alpha=0.25)
    else:
        ax.set_title("Raw |dF/dt| Distribution (No Preprocessing, no data)")
        ax.grid(True, alpha=0.25)

    # 3) preprocessed diff distribution
    ax = axes[2]
    if preproc_diff.size > 0:
        mode = str(preproc_info.get("mode", "custom_divisor"))
        if mode == "model_tanh":
            bins = np.linspace(-1.0, 1.0, 220)
            ax.hist(preproc_diff, bins=bins, density=True, alpha=0.8, color="#8338EC")
            frac_small = float(np.mean(np.abs(preproc_diff) < 0.05))
            frac_sat = float(np.mean(np.abs(preproc_diff) > 0.95))
            s = float(preproc_info.get("d_force_scale", 0.0))
            ax.set_title(
                f"Model Preprocessed Diff: tanh(dF/dt / {s:.3g})\n"
                f"frac(|x|<0.05)={frac_small:.2%}, frac(|x|>0.95)={frac_sat:.2%}"
            )
            ax.set_xlabel("preprocessed diff [-1, 1]")
        elif mode == "signed_divisor":
            q = float(np.quantile(np.abs(preproc_diff), 0.999))
            q = max(q, 1e-6)
            bins = np.linspace(-q, q, 240)
            ax.hist(preproc_diff, bins=bins, density=True, alpha=0.8, color="#8338EC")
            div_v = float(preproc_info.get("divisor_used", 0.0))
            p = float(preproc_info.get("divisor_percentile", 0.0))
            ax.set_title(
                f"Preprocessed Diff (signed / P{p:.1f}(|x|), divisor={div_v:.3g})\n"
                f"mean={np.mean(preproc_diff):.3f}, std={np.std(preproc_diff):.3f}"
            )
            ax.set_xlabel("preprocessed diff (signed)")
        else:
            bins = np.linspace(0.0, 1.0, 180)
            ax.hist(preproc_diff, bins=bins, density=True, alpha=0.8, color="#8338EC")
            frac_low = float(np.mean(preproc_diff < 0.05))
            frac_high = float(np.mean(preproc_diff > 0.95))
            div_v = float(preproc_info.get("divisor_used", 0.0))
            ax.set_title(
                f"Preprocessed Diff (x / divisor, divisor={div_v:.3g})\n"
                f"frac(<0.05)={frac_low:.2%}, frac(>0.95)={frac_high:.2%}"
            )
            ax.set_xlabel("preprocessed diff [0, 1]")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
    else:
        ax.set_title("Preprocessed Diff Distribution (no data)")
        ax.grid(True, alpha=0.25)

    # 4) legacy preprocessing: tanh(dF/dt / scale)
    ax = axes[3]
    if legacy_tanh_diff.size > 0:
        bins = np.linspace(-1.0, 1.0, 220)
        ax.hist(legacy_tanh_diff, bins=bins, density=True, alpha=0.8, color="#FF9F1C")
        frac_small = float(np.mean(np.abs(legacy_tanh_diff) < 0.05))
        frac_sat = float(np.mean(np.abs(legacy_tanh_diff) > 0.95))
        s = float(legacy_tanh_info.get("d_force_scale", 0.0))
        ax.set_title(
            "Legacy Preprocessed Diff (raw-like outlier filtered)\n"
            f"tanh(dF/dt / {s:.3g}), frac(|x|<0.05)={frac_small:.2%}, frac(|x|>0.95)={frac_sat:.2%}"
        )
        ax.set_xlabel("legacy preprocessed diff [-1, 1]")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
    else:
        ax.set_title("Legacy Preprocessed Diff (no data)")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/noisy")
    parser.add_argument("--pattern", default="lowstate_segment.jsonl")
    parser.add_argument("--smooth-win", type=int, default=1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1_000_000,
        help="분포 시각화를 위한 reservoir sample 크기",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--diff-abs-min",
        type=float,
        default=0.0,
        help="|d(foot_force)/d(monotonic_sec)| 사전 필터 최소값(이보다 작으면 제외)",
    )
    parser.add_argument(
        "--diff-abs-max",
        type=float,
        default=float("inf"),
        help="|d(foot_force)/d(monotonic_sec)| 사전 필터 최대값(이보다 크면 제외)",
    )
    parser.add_argument(
        "--preproc-abs-min",
        type=float,
        default=1e-7,
        help="정규화(x/max) 전 |dF/dt| 하한. 이보다 작은 값은 제거",
    )
    parser.add_argument(
        "--preproc-upper-percentile",
        type=float,
        default=99.9,
        help="정규화(x/max) 전 |dF/dt| 상한 percentile. 이보다 큰 값은 제거",
    )
    parser.add_argument(
        "--norm-divisor",
        type=float,
        default=0.0,
        help="정규화 분모 값. 0이면 자동으로 max 사용, 양수면 해당 값으로 나눔",
    )
    parser.add_argument(
        "--norm-divisor-percentile",
        type=float,
        default=0.0,
        help="정규화 분모를 샘플 percentile 값으로 사용 (예: 90 -> p90). norm-divisor가 우선",
    )
    parser.add_argument(
        "--preprocess-mode",
        default="model_tanh",
        choices=["model_tanh", "custom_divisor", "signed_divisor"],
        help="3번째(diff 전처리) 플롯 방식 선택",
    )
    parser.add_argument(
        "--d-force-scale",
        type=float,
        default=9220.325595510363,
        help="model_tanh 모드에서 tanh(dF/dt / d_force_scale)의 scale",
    )
    parser.add_argument(
        "--raw-plot-upper-percentile",
        type=float,
        default=99.9,
        help="2번째(raw diff) 플롯에서 시각화용 상한 percentile",
    )
    parser.add_argument(
        "--raw-plot-abs-min",
        type=float,
        default=1e-7,
        help="2번째(raw diff) 플롯에서 시각화용 하한값(|dF/dt|). 이보다 작은 값은 제외",
    )
    parser.add_argument(
        "--rare-tail-frac",
        type=float,
        default=1e-4,
        help="분포 기반 희소 이상치 비율(상위 꼬리 비율, 예: 1e-4=상위 0.01%%)",
    )
    parser.add_argument(
        "--rare-log-mad-z",
        type=float,
        default=8.0,
        help="로그 스케일 MAD 기준 임계치(z). 클수록 덜 제거",
    )
    parser.add_argument(
        "--rare-min-samples",
        type=int,
        default=5000,
        help="분포 기반 희소 이상치 필터를 적용할 최소 샘플 수",
    )
    parser.add_argument(
        "--enable-rare-outlier-filter",
        action="store_true",
        help="분포 기반 희소 이상치 필터를 추가로 적용",
    )
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--out-dir", default="outputs/plots/condition_dist")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")
    if args.smooth_win > 1 and args.smooth_win % 2 == 0:
        raise ValueError("smooth-win should be odd when > 1")
    if args.diff_abs_min < 0:
        raise ValueError("diff-abs-min should be >= 0")
    if args.diff_abs_max <= args.diff_abs_min:
        raise ValueError("diff-abs-max should be > diff-abs-min")
    if args.preproc_abs_min < 0:
        raise ValueError("preproc-abs-min should be >= 0")
    if not (0.0 < args.preproc_upper_percentile <= 100.0):
        raise ValueError("need 0 < preproc-upper-percentile <= 100")
    if args.norm_divisor < 0:
        raise ValueError("norm-divisor should be >= 0")
    if not (0.0 <= args.norm_divisor_percentile <= 100.0):
        raise ValueError("norm-divisor-percentile should be in [0, 100]")
    if args.d_force_scale <= 0:
        raise ValueError("d-force-scale should be > 0")
    if not (0.0 < args.raw_plot_upper_percentile <= 100.0):
        raise ValueError("need 0 < raw-plot-upper-percentile <= 100")
    if args.raw_plot_abs_min < 0:
        raise ValueError("raw-plot-abs-min should be >= 0")
    if not (0.0 < args.rare_tail_frac < 0.5):
        raise ValueError("rare-tail-frac should be in (0, 0.5)")
    if args.rare_log_mad_z <= 0:
        raise ValueError("rare-log-mad-z should be > 0")
    if args.rare_min_samples < 1:
        raise ValueError("rare-min-samples should be >= 1")

    files = sorted(root.rglob(args.pattern))
    if len(files) == 0:
        raise RuntimeError(f"no files found: root={root}, pattern={args.pattern}")

    raw_sampler = ReservoirSampler(args.max_samples, seed=args.seed + 1)
    raw_abs_deriv_sampler = ReservoirSampler(args.max_samples, seed=args.seed + 2)
    raw_deriv_sampler = ReservoirSampler(args.max_samples, seed=args.seed + 3)
    deriv_sampler = ReservoirSampler(args.max_samples, seed=args.seed + 4)

    total_lines = 0
    valid_foot_force = 0
    valid_derivs = 0
    parse_errors = 0
    missing_monotonic_time = 0
    bad_monotonic_diff = 0
    filtered_abs_range_outliers = 0

    for i, path in enumerate(files, start=1):
        rows_t = []
        rows_f = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                total_lines += 1
                try:
                    row = json.loads(line)
                except Exception:
                    parse_errors += 1
                    continue

                ff = get_foot_force(row)
                if ff is None:
                    continue
                valid_foot_force += 1
                raw_sampler.add_many(ff)

                t = get_monotonic_time_sec(row)
                if t is None or not math.isfinite(t):
                    missing_monotonic_time += 1
                    continue
                rows_t.append(float(t))
                rows_f.append(ff)

        if len(rows_t) < 2:
            if i % args.log_every == 0 or i == len(files):
                print(f"[{i}/{len(files)}] lines={total_lines} valid_ff={valid_foot_force} valid_deriv={valid_derivs}")
            continue

        t_arr = np.asarray(rows_t, dtype=np.float64)
        f_arr = np.asarray(rows_f, dtype=np.float64)

        order = np.argsort(t_arr)
        t_arr = t_arr[order]
        f_arr = f_arr[order]

        if args.smooth_win > 1:
            f_used = np.empty_like(f_arr)
            for k in range(4):
                f_used[:, k] = moving_average_1d(f_arr[:, k], args.smooth_win)
        else:
            f_used = f_arr

        monotonic_diff = np.diff(t_arr)
        valid = np.isfinite(monotonic_diff) & (monotonic_diff > 0)
        if np.any(~valid):
            bad_monotonic_diff += int(np.sum(~valid))
        if not np.any(valid):
            if i % args.log_every == 0 or i == len(files):
                print(f"[{i}/{len(files)}] lines={total_lines} valid_ff={valid_foot_force} valid_deriv={valid_derivs}")
            continue

        df = np.diff(f_used, axis=0)[valid]
        monotonic_diff_valid = monotonic_diff[valid]
        deriv = df / monotonic_diff_valid[:, None]
        deriv_flat = deriv.reshape(-1)
        finite_deriv_mask = np.isfinite(deriv_flat)
        if np.any(finite_deriv_mask):
            deriv_finite = deriv_flat[finite_deriv_mask]
            raw_deriv_sampler.add_many(deriv_finite)
            raw_abs_deriv_sampler.add_many(np.abs(deriv_finite))
        abs_deriv_flat = np.abs(deriv_flat)
        valid_deriv_mask = (
            np.isfinite(deriv_flat)
            & (abs_deriv_flat >= float(args.diff_abs_min))
            & (abs_deriv_flat <= float(args.diff_abs_max))
        )
        filtered_abs_range_outliers += int(np.size(deriv_flat) - np.sum(valid_deriv_mask))
        if np.any(valid_deriv_mask):
            deriv_sampler.add_many(deriv_flat[valid_deriv_mask])
            valid_derivs += int(np.sum(valid_deriv_mask))

        if i % args.log_every == 0 or i == len(files):
            print(f"[{i}/{len(files)}] lines={total_lines} valid_ff={valid_foot_force} valid_deriv={valid_derivs}")

    raw = raw_sampler.as_array()
    raw_abs_deriv = raw_abs_deriv_sampler.as_array()
    raw_deriv = raw_deriv_sampler.as_array()
    deriv = deriv_sampler.as_array()
    abs_deriv = np.abs(deriv)
    filtered_rare_outliers = 0
    rare_filter_info = {
        "applied": False,
        "reason": "not_enabled",
        "n": int(abs_deriv.size),
        "tail_count": 0,
        "tail_threshold": math.inf,
        "robust_upper": math.inf,
        "upper_bound": math.inf,
    }
    if args.enable_rare_outlier_filter:
        keep_mask, rare_filter_info = filter_distribution_rare_outliers(
            abs_values=abs_deriv,
            tail_frac=args.rare_tail_frac,
            log_mad_z=args.rare_log_mad_z,
            min_samples=args.rare_min_samples,
        )
        filtered_rare_outliers = int(np.sum(~keep_mask))
        if filtered_rare_outliers > 0:
            deriv = deriv[keep_mask]
            abs_deriv = abs_deriv[keep_mask]

    preproc_input = abs_deriv
    preproc_q_upper = math.inf
    filtered_preproc_extremes = 0
    if preproc_input.size > 0:
        preproc_q_upper = float(np.quantile(preproc_input, args.preproc_upper_percentile / 100.0))
        preproc_keep = (
            np.isfinite(preproc_input)
            & (preproc_input >= float(args.preproc_abs_min))
            & (preproc_input <= preproc_q_upper)
        )
        filtered_preproc_extremes = int(np.sum(~preproc_keep))
        if np.any(preproc_keep):
            preproc_input = preproc_input[preproc_keep]

    # raw diff 플롯(2번째)과 같은 기준으로 signed diff 극단값 제거
    # (model_tanh / signed_divisor 전처리 경로에서 사용)
    preproc_signed_input = raw_deriv
    preproc_signed_q_upper = math.inf
    filtered_preproc_signed_extremes = 0
    if preproc_signed_input.size > 0:
        preproc_signed_abs = np.abs(preproc_signed_input)
        preproc_signed_q_upper = float(
            np.quantile(preproc_signed_abs, args.raw_plot_upper_percentile / 100.0)
        )
        preproc_signed_keep = (
            np.isfinite(preproc_signed_input)
            & (preproc_signed_abs >= float(args.raw_plot_abs_min))
            & (preproc_signed_abs <= preproc_signed_q_upper)
        )
        filtered_preproc_signed_extremes = int(np.sum(~preproc_signed_keep))
        if np.any(preproc_signed_keep):
            preproc_signed_input = preproc_signed_input[preproc_signed_keep]
        else:
            # 최소값 조건 때문에 비는 경우 상한 조건만으로 fallback
            fallback_keep = np.isfinite(preproc_signed_input) & (
                preproc_signed_abs <= preproc_signed_q_upper
            )
            if np.any(fallback_keep):
                preproc_signed_input = preproc_signed_input[fallback_keep]

    if args.preprocess_mode == "model_tanh":
        preproc_diff, preproc_info = preprocess_diff_model_tanh(
            deriv=preproc_signed_input,
            d_force_scale=args.d_force_scale,
        )
    elif args.preprocess_mode == "signed_divisor":
        preproc_diff, preproc_info = normalize_signed_diff_by_divisor(
            diff=preproc_signed_input,
            divisor_value=args.norm_divisor,
            divisor_percentile=args.norm_divisor_percentile,
        )
    else:
        preproc_diff, preproc_info = normalize_abs_diff_by_divisor(
            abs_diff=preproc_input,
            divisor_value=args.norm_divisor,
            divisor_percentile=args.norm_divisor_percentile,
        )
        preproc_info["mode"] = "custom_divisor"

    # 4번째 패널: 기존 전처리 (scale division + tanh), 2/3번과 같은 사전 이상치 제거 입력 사용
    legacy_tanh_diff, legacy_tanh_info = preprocess_diff_model_tanh(
        deriv=preproc_signed_input,
        d_force_scale=args.d_force_scale,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_path = out_dir / "distribution_overview.png"

    plot_main_distributions(
        raw=raw,
        raw_abs_deriv=raw_abs_deriv,
        preproc_diff=preproc_diff,
        preproc_info=preproc_info,
        legacy_tanh_diff=legacy_tanh_diff,
        legacy_tanh_info=legacy_tanh_info,
        raw_plot_upper_percentile=args.raw_plot_upper_percentile,
        raw_plot_abs_min=args.raw_plot_abs_min,
        out_path=dist_path,
    )

    print("=" * 80)
    print("summary")
    print("=" * 80)
    print(f"root={root}")
    print(f"pattern={args.pattern}")
    print(f"files={len(files)}")
    print(f"total_lines={total_lines}")
    print(f"valid_foot_force={valid_foot_force}")
    print(f"valid_derivative_values={valid_derivs}")
    print(f"parse_errors={parse_errors}")
    print(f"missing_monotonic_time={missing_monotonic_time}")
    print(f"bad_monotonic_diff={bad_monotonic_diff}")
    print(f"diff_abs_filter=[{args.diff_abs_min}, {args.diff_abs_max}]")
    print(f"filtered_abs_range_outliers={filtered_abs_range_outliers}")
    print(f"rare_outlier_filter_enabled={args.enable_rare_outlier_filter}")
    print(f"rare_filter_applied={rare_filter_info['applied']}")
    if rare_filter_info["reason"]:
        print(f"rare_filter_reason={rare_filter_info['reason']}")
    if rare_filter_info["applied"]:
        print(f"rare_filter_tail_count={rare_filter_info['tail_count']}")
        print(f"rare_filter_tail_threshold={rare_filter_info['tail_threshold']}")
        print(f"rare_filter_robust_upper={rare_filter_info['robust_upper']}")
        print(f"rare_filter_upper_bound={rare_filter_info['upper_bound']}")
    print(f"filtered_rare_outliers={filtered_rare_outliers}")
    print(f"preprocess_mode={args.preprocess_mode}")
    print(f"d_force_scale={args.d_force_scale}")
    print("normalization=x/divisor (custom_divisor mode only)")
    print(f"norm_divisor_arg={args.norm_divisor}")
    print(f"norm_divisor_percentile_arg={args.norm_divisor_percentile}")
    print(f"preproc_abs_min={args.preproc_abs_min}")
    print(f"preproc_upper_percentile={args.preproc_upper_percentile}")
    print(f"preproc_q_upper={preproc_q_upper}")
    print(f"filtered_preproc_extremes={filtered_preproc_extremes}")
    print(f"preproc_signed_abs_min(raw_like)={args.raw_plot_abs_min}")
    print(f"preproc_signed_upper_percentile(raw_like)={args.raw_plot_upper_percentile}")
    print(f"preproc_signed_q_upper(raw_like)={preproc_signed_q_upper}")
    print(f"filtered_preproc_signed_extremes(raw_like)={filtered_preproc_signed_extremes}")
    print(f"raw_plot_upper_percentile={args.raw_plot_upper_percentile}")
    print(f"raw_plot_abs_min={args.raw_plot_abs_min}")
    print(f"preproc_applied={preproc_info['applied']}")
    if preproc_info["reason"]:
        print(f"preproc_reason={preproc_info['reason']}")
    if preproc_info["applied"]:
        if preproc_info.get("mode") == "model_tanh":
            print(f"preproc_mode=model_tanh")
            print(f"preproc_d_force_scale={preproc_info['d_force_scale']}")
        elif preproc_info.get("mode") == "signed_divisor":
            print(f"preproc_mode=signed_divisor")
            print(f"norm_divisor_source={preproc_info['divisor_source']}")
            print(f"norm_divisor_used={preproc_info['divisor_used']}")
            if preproc_info["divisor_source"] in {"sample_percentile", "sample_abs_percentile"}:
                print(f"norm_divisor_percentile={preproc_info['divisor_percentile']}")
                print(f"norm_divisor_percentile_value={preproc_info['divisor_percentile_value']}")
        else:
            print(f"preproc_mode=custom_divisor")
            print(f"norm_divisor_source={preproc_info['divisor_source']}")
            print(f"norm_divisor_used={preproc_info['divisor_used']}")
            if preproc_info["divisor_source"] == "sample_percentile":
                print(f"norm_divisor_percentile={preproc_info['divisor_percentile']}")
                print(f"norm_divisor_percentile_value={preproc_info['divisor_percentile_value']}")
            print(f"norm_max_value={preproc_info['max_value']}")
    print_distribution_summary("raw(sampled)", raw)
    print_distribution_summary("raw_abs_deriv_no_preproc(sampled)", raw_abs_deriv)
    print_distribution_summary("raw_deriv_signed_no_preproc(sampled)", raw_deriv)
    print_distribution_summary("deriv(sampled)", deriv)
    print_distribution_summary("abs_deriv(sampled)", abs_deriv)
    print_distribution_summary("preproc_input_abs_deriv(sampled)", preproc_input)
    print_distribution_summary("preproc_signed_input_filtered(sampled)", preproc_signed_input)
    print_distribution_summary("preprocessed_diff(sampled)", preproc_diff)
    print_distribution_summary("legacy_tanh_preprocessed_diff(sampled)", legacy_tanh_diff)
    print(f"saved: {dist_path}")


if __name__ == "__main__":
    main()
