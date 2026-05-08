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


def normalize_abs_diff_asinh_percentile(abs_diff, upper_percentile=99.5, asinh_scale=0.0):
    x = np.asarray(abs_diff, dtype=np.float64).reshape(-1)
    out = np.zeros_like(x, dtype=np.float64)
    info = {
        "applied": False,
        "reason": "",
        "count": int(x.size),
        "upper_percentile": float(upper_percentile),
        "q_upper": 0.0,
        "clip_upper": 0.0,
        "asinh_scale": 0.0,
        "above_upper_count": 0,
    }

    if x.size == 0:
        info["reason"] = "empty"
        return out, info
    if not (0.0 < upper_percentile <= 100.0):
        info["reason"] = "invalid_upper_percentile"
        return out, info

    q_upper = float(np.quantile(x, upper_percentile / 100.0))
    clip_upper = max(q_upper, 1e-12)
    x_clip = np.clip(x, 0.0, clip_upper)

    if asinh_scale > 0:
        s = float(asinh_scale)
    else:
        positive = x_clip[x_clip > 0]
        if positive.size > 0:
            s = float(np.quantile(positive, 0.5))
        else:
            s = clip_upper
        s = max(min(s, clip_upper), clip_upper * 1e-3, 1e-12)

    denom = float(np.arcsinh(clip_upper / s))
    if denom > 1e-12:
        out = np.arcsinh(x_clip / s) / denom
    else:
        out = x_clip / clip_upper
    out = np.clip(out, 0.0, 1.0)

    info["applied"] = True
    info["q_upper"] = q_upper
    info["clip_upper"] = clip_upper
    info["asinh_scale"] = s
    info["above_upper_count"] = int(np.sum(x > clip_upper))
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


def plot_main_distributions(raw, raw_abs_deriv, norm_diff, norm_info, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))

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
        x = np.clip(raw_abs_deriv, 1e-8, None)
        lo = float(np.quantile(x, 0.001))
        hi = float(np.quantile(x, 0.999))
        lo = max(lo, 1e-8)
        if hi <= lo:
            hi = max(lo * 10.0, 1.0)
        bins = np.logspace(np.log10(lo), np.log10(hi), 220)
        ax.hist(x, bins=bins, density=True, alpha=0.78, color="#06D6A0")
        ax.set_xscale("log")
        ax.set_title(
            "Raw |dF/dt| Distribution (No Preprocessing)\n"
            f"mean={np.mean(x):.2f}, p95={np.quantile(x, 0.95):.2f}, p99={np.quantile(x, 0.99):.2f}"
        )
        ax.set_xlabel("raw |dF/d(monotonic_sec)| (log x)")
        ax.set_ylabel("density")
        ax.grid(True, which="both", alpha=0.25)
    else:
        ax.set_title("Raw |dF/dt| Distribution (No Preprocessing, no data)")
        ax.grid(True, alpha=0.25)

    # 3) normalized diff distribution
    ax = axes[2]
    if norm_diff.size > 0:
        bins = np.linspace(0.0, 1.0, 180)
        ax.hist(norm_diff, bins=bins, density=True, alpha=0.8, color="#8338EC")
        frac_low = float(np.mean(norm_diff < 0.05))
        frac_high = float(np.mean(norm_diff > 0.95))
        up = float(norm_info.get("upper_percentile", 99.5))
        ax.set_title(
            f"Preprocessed Diff (clip@P{up:.1f} + asinh)\n"
            f"frac(<0.05)={frac_low:.2%}, frac(>0.95)={frac_high:.2%}"
        )
        ax.set_xlabel("preprocessed diff [0, 1]")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
    else:
        ax.set_title("Preprocessed Diff Distribution (no data)")
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
        "--norm-upper-percentile",
        type=float,
        default=99.5,
        help="정규화 전 clipping 상한 percentile (권장: 99.5~99.9)",
    )
    parser.add_argument(
        "--norm-asinh-scale",
        type=float,
        default=0.0,
        help="asinh 스케일 상수 s. 0이면 데이터 기반 자동 추정",
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
    if not (0.0 < args.norm_upper_percentile <= 100.0):
        raise ValueError("need 0 < norm-upper-percentile <= 100")
    if args.norm_asinh_scale < 0:
        raise ValueError("norm-asinh-scale should be >= 0")
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
    deriv_sampler = ReservoirSampler(args.max_samples, seed=args.seed + 3)

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
            raw_abs_deriv_sampler.add_many(np.abs(deriv_flat[finite_deriv_mask]))
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

    norm_diff, norm_info = normalize_abs_diff_asinh_percentile(
        abs_diff=abs_deriv,
        upper_percentile=args.norm_upper_percentile,
        asinh_scale=args.norm_asinh_scale,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_path = out_dir / "distribution_overview.png"

    plot_main_distributions(
        raw=raw,
        raw_abs_deriv=raw_abs_deriv,
        norm_diff=norm_diff,
        norm_info=norm_info,
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
    print(f"norm_upper_percentile={args.norm_upper_percentile}")
    print(f"norm_asinh_scale_arg={args.norm_asinh_scale}")
    print(f"norm_applied={norm_info['applied']}")
    if norm_info["reason"]:
        print(f"norm_reason={norm_info['reason']}")
    if norm_info["applied"]:
        print(f"norm_q_upper={norm_info['q_upper']}")
        print(f"norm_clip_upper={norm_info['clip_upper']}")
        print(f"norm_asinh_scale_used={norm_info['asinh_scale']}")
        print(f"norm_above_upper_count={norm_info['above_upper_count']}")
    print_distribution_summary("raw(sampled)", raw)
    print_distribution_summary("raw_abs_deriv_no_preproc(sampled)", raw_abs_deriv)
    print_distribution_summary("deriv(sampled)", deriv)
    print_distribution_summary("abs_deriv(sampled)", abs_deriv)
    print_distribution_summary("norm_diff(sampled)", norm_diff)
    print(f"saved: {dist_path}")


if __name__ == "__main__":
    main()
