#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from array import array

import numpy as np


def get_time_sec(row):
    keys_ns = [
        "clock_monotonic_ns",
        "monotonic_ns",
        "timestamp_ns",
        "time_ns",
    ]

    for k in keys_ns:
        if k in row and isinstance(row[k], (int, float)):
            return float(row[k]) / 1e9

    keys_sec = [
        "time_sec",
        "timestamp_sec",
        "t_sec",
    ]

    for k in keys_sec:
        if k in row and isinstance(row[k], (int, float)):
            return float(row[k])

    return None


def get_foot_force(row):
    if "msg" in row and isinstance(row["msg"], dict):
        msg = row["msg"]
        if "foot_force" in msg:
            ff = msg["foot_force"]
            if isinstance(ff, list) and len(ff) >= 4:
                return [float(ff[0]), float(ff[1]), float(ff[2]), float(ff[3])]

    if "foot_force" in row:
        ff = row["foot_force"]
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


def array_to_numpy(arr):
    if len(arr) == 0:
        return np.asarray([], dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def percentile(arr, ps):
    x = array_to_numpy(arr)

    if len(x) == 0:
        return {p: None for p in ps}

    return {p: float(np.percentile(x, p)) for p in ps}


def print_percentile_table(title, arr, ps):
    x = array_to_numpy(arr)

    print(title)
    print("-" * 80)

    if len(x) == 0:
        print("no data")
        print()
        return

    print(f"count : {len(x)}")
    print(f"min   : {float(np.min(x))}")
    print(f"max   : {float(np.max(x))}")
    print(f"mean  : {float(np.mean(x))}")
    print(f"std   : {float(np.std(x))}")

    for p in ps:
        print(f"p{p:<6}: {float(np.percentile(x, p))}")

    print()


def tanh_saturation_report(arr, s_values):
    x = array_to_numpy(arr)

    if len(x) == 0:
        return

    print("tanh saturation check for abs(tanh(dFdt / s))")
    print("-" * 80)
    print("s value      mean(abs(tanh))   >=0.50      >=0.80      >=0.95      >=0.99")

    for s in s_values:
        if s is None or s <= 0:
            continue

        y = np.abs(np.tanh(x / s))

        r50 = np.mean(y >= 0.50) * 100
        r80 = np.mean(y >= 0.80) * 100
        r95 = np.mean(y >= 0.95) * 100
        r99 = np.mean(y >= 0.99) * 100

        print(
            f"{s:<12.3f}"
            f"{float(np.mean(y)):<18.6f}"
            f"{r50:<11.3f}%"
            f"{r80:<11.3f}%"
            f"{r95:<11.3f}%"
            f"{r99:<11.3f}%"
        )

    print()


def analyze_file(path, args, stats):
    rows_t = []
    rows_f = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            stats["total_lines"] += 1

            try:
                row = json.loads(line)
            except Exception:
                stats["parse_errors"] += 1
                continue

            t = get_time_sec(row)
            ff = get_foot_force(row)

            if ff is None:
                continue

            stats["valid_foot_force"] += 1

            for k in range(4):
                v = ff[k]
                stats["raw_abs"].append(abs(v))

                if v < stats["raw_leg_min"][k]:
                    stats["raw_leg_min"][k] = v
                if v > stats["raw_leg_max"][k]:
                    stats["raw_leg_max"][k] = v

                if v < stats["raw_global_min"]:
                    stats["raw_global_min"] = v
                    stats["raw_global_min_loc"] = (str(path), line_no, ff)

                if v > stats["raw_global_max"]:
                    stats["raw_global_max"] = v
                    stats["raw_global_max_loc"] = (str(path), line_no, ff)

            if t is None or not math.isfinite(t):
                stats["missing_time"] += 1
                continue

            rows_t.append(t)
            rows_f.append(ff)

    if len(rows_t) < 2:
        return

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

    for i in range(1, len(t_arr)):
        dt = t_arr[i] - t_arr[i - 1]

        if not math.isfinite(dt) or dt <= 0:
            stats["bad_dt"] += 1
            continue

        diff = f_used[i] - f_used[i - 1]
        deriv = diff / dt

        stats["dt_values"].append(dt)

        for k in range(4):
            d = float(diff[k])
            dv = float(deriv[k])

            stats["diff_abs"].append(abs(d))
            stats["deriv_abs"].append(abs(dv))

            if d < stats["diff_leg_min"][k]:
                stats["diff_leg_min"][k] = d
            if d > stats["diff_leg_max"][k]:
                stats["diff_leg_max"][k] = d

            if dv < stats["deriv_leg_min"][k]:
                stats["deriv_leg_min"][k] = dv
            if dv > stats["deriv_leg_max"][k]:
                stats["deriv_leg_max"][k] = dv

            if dv < stats["deriv_global_min"]:
                stats["deriv_global_min"] = dv
                stats["deriv_global_min_loc"] = (
                    str(path),
                    i,
                    i + 1,
                    dt,
                    f_arr[i - 1].tolist(),
                    f_arr[i].tolist(),
                    f_used[i - 1].tolist(),
                    f_used[i].tolist(),
                    diff.tolist(),
                    deriv.tolist(),
                )

            if dv > stats["deriv_global_max"]:
                stats["deriv_global_max"] = dv
                stats["deriv_global_max_loc"] = (
                    str(path),
                    i,
                    i + 1,
                    dt,
                    f_arr[i - 1].tolist(),
                    f_arr[i].tolist(),
                    f_used[i - 1].tolist(),
                    f_used[i].tolist(),
                    diff.tolist(),
                    deriv.tolist(),
                )

            stats["valid_derivatives"] += 1


def print_location(title, loc):
    print(title)

    if loc is None:
        print("  none")
        return

    print(f"  file       : {loc[0]}")
    print(f"  lines      : {loc[1]} -> {loc[2]}")
    print(f"  dt         : {loc[3]}")
    print(f"  prev raw   : {loc[4]}")
    print(f"  curr raw   : {loc[5]}")
    print(f"  prev used  : {loc[6]}")
    print(f"  curr used  : {loc[7]}")
    print(f"  diff       : {loc[8]}")
    print(f"  dFdt       : {loc[9]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./noisy")
    parser.add_argument("--pattern", type=str, default="lowstate_segment.jsonl")
    parser.add_argument("--smooth-win", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob(args.pattern))

    if not files:
        raise FileNotFoundError(f"no files found: {root}/**/{args.pattern}")

    print("=" * 80)
    print("settings")
    print("=" * 80)
    print(f"root       : {root}")
    print(f"pattern    : {args.pattern}")
    print(f"files      : {len(files)}")
    print(f"smooth_win : {args.smooth_win}")
    print("dt         : actual timestamp difference only")
    print()

    stats = {
        "total_lines": 0,
        "valid_foot_force": 0,
        "valid_derivatives": 0,
        "missing_time": 0,
        "bad_dt": 0,
        "parse_errors": 0,

        "raw_abs": array("d"),
        "diff_abs": array("d"),
        "deriv_abs": array("d"),
        "dt_values": array("d"),

        "raw_global_min": float("inf"),
        "raw_global_max": -float("inf"),
        "raw_leg_min": [float("inf")] * 4,
        "raw_leg_max": [-float("inf")] * 4,
        "raw_global_min_loc": None,
        "raw_global_max_loc": None,

        "diff_leg_min": [float("inf")] * 4,
        "diff_leg_max": [-float("inf")] * 4,

        "deriv_global_min": float("inf"),
        "deriv_global_max": -float("inf"),
        "deriv_leg_min": [float("inf")] * 4,
        "deriv_leg_max": [-float("inf")] * 4,
        "deriv_global_min_loc": None,
        "deriv_global_max_loc": None,
    }

    for idx, path in enumerate(files, start=1):
        analyze_file(path, args, stats)

        if idx % args.log_every == 0 or idx == len(files):
            print(
                f"[{idx}/{len(files)}] "
                f"lines={stats['total_lines']} "
                f"valid_ff={stats['valid_foot_force']} "
                f"valid_diff={stats['valid_derivatives']}"
            )

    ps = [50, 75, 90, 95, 99, 99.5, 99.9, 99.95, 99.99]

    print()
    print("=" * 80)
    print("result")
    print("=" * 80)
    print(f"lowstate files      : {len(files)}")
    print(f"total lines         : {stats['total_lines']}")
    print(f"valid foot_force    : {stats['valid_foot_force']}")
    print(f"valid derivatives   : {stats['valid_derivatives']}")
    print(f"missing time count  : {stats['missing_time']}")
    print(f"bad dt count        : {stats['bad_dt']}")
    print(f"parse errors        : {stats['parse_errors']}")
    print()

    print("-" * 80)
    print(f"foot_force global min : {stats['raw_global_min']}")
    print(f"foot_force global max : {stats['raw_global_max']}")
    print()
    print("recommended raw denominator:")
    print(f"  F_norm = F / {stats['raw_global_max']}")
    print()

    for k in range(4):
        print(
            f"foot_force leg {k} min/max : "
            f"{stats['raw_leg_min'][k]} / {stats['raw_leg_max'][k]}"
        )

    print()
    print("-" * 80)
    print("1-sample diff min/max")
    for k in range(4):
        print(
            f"diff leg {k} min/max : "
            f"{stats['diff_leg_min'][k]} / {stats['diff_leg_max'][k]}"
        )

    print()
    print("-" * 80)
    print("1-sample derivative min/max with actual dt")
    print(f"derivative global min : {stats['deriv_global_min']}")
    print(f"derivative global max : {stats['deriv_global_max']}")

    for k in range(4):
        print(
            f"derivative leg {k} min/max : "
            f"{stats['deriv_leg_min'][k]} / {stats['deriv_leg_max'][k]}"
        )

    print()
    print("-" * 80)
    print("raw min location")
    if stats["raw_global_min_loc"] is not None:
        print(f"  file  : {stats['raw_global_min_loc'][0]}")
        print(f"  line  : {stats['raw_global_min_loc'][1]}")
        print(f"  value : {stats['raw_global_min_loc'][2]}")

    print("raw max location")
    if stats["raw_global_max_loc"] is not None:
        print(f"  file  : {stats['raw_global_max_loc'][0]}")
        print(f"  line  : {stats['raw_global_max_loc'][1]}")
        print(f"  value : {stats['raw_global_max_loc'][2]}")

    print()
    print("-" * 80)
    print_location("derivative global min location", stats["deriv_global_min_loc"])
    print_location("derivative global max location", stats["deriv_global_max_loc"])

    print()
    print_percentile_table("dt distribution", stats["dt_values"], ps)
    print_percentile_table("abs(raw foot_force) distribution", stats["raw_abs"], ps)
    print_percentile_table("abs(1-sample diff) distribution", stats["diff_abs"], ps)
    print_percentile_table("abs(1-sample derivative) distribution", stats["deriv_abs"], ps)

    deriv_p = percentile(stats["deriv_abs"], [95, 99, 99.5, 99.9, 99.95, 99.99])

    print("=" * 80)
    print("recommended s candidates for tanh(dFdt / s)")
    print("=" * 80)
    for p, v in deriv_p.items():
        print(f"s = p{p:<6} abs(dFdt) = {v}")

    s_values = []
    for v in deriv_p.values():
        if v is not None and v > 0:
            s_values.append(v)

    print()
    tanh_saturation_report(stats["deriv_abs"], s_values)

    print("=" * 80)
    print("final formula candidate")
    print("=" * 80)
    print(f"F_norm = F / {stats['raw_global_max']}")
    print("dFdt   = (F[t] - F[t-1]) / (t[t] - t[t-1])")
    print("D_norm = tanh(dFdt / s)")
    print()
    print("s는 위 percentile 중 p99 또는 p99.5부터 먼저 확인.")
    print("impact를 더 강하게 살리고 싶으면 p99.9 사용.")


if __name__ == "__main__":
    main()