#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "curriculum_analysis"
LABEL_ORDER = [
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
]
ALIGNED_EPOCHS = [5, 10, 15, 20]
STRATEGY_ORDER = ["endtoend", "curriculum"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join("---" for _ in headers) + " |"
    line_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([line1, line2, *line_rows])


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def infer_subset(sample_id: str) -> str:
    sample_id = sample_id.lower()
    if sample_id.endswith("__gnlos") or sample_id.endswith("__gnlos".lower()) or sample_id.endswith("__gnlos".upper().lower()):
        return "gnLOS"
    if sample_id.endswith("__gnlos") or "__gnlos" in sample_id:
        return "gnLOS"
    if sample_id.endswith("__glos") or "__glos" in sample_id:
        return "gLOS"
    if sample_id.endswith("__gnlos"):
        return "gnLOS"
    return "unknown"


def scan_metrics() -> list[dict]:
    rows: list[dict] = []

    end_root = ROOT / "01_endtoend"
    for epoch_dir in sorted(end_root.glob("epoch_*")):
        epoch = int(epoch_dir.name.split("_")[-1])
        metrics_path = epoch_dir / f"{epoch_dir.name}_metrics.json"
        decode_path = epoch_dir / f"{epoch_dir.name}_decode.jsonl"
        if not metrics_path.exists() or not decode_path.exists():
            continue
        data = load_json(metrics_path)
        rows.append(
            {
                "strategy": "endtoend",
                "epoch": epoch,
                "metrics_path": str(metrics_path),
                "decode_path": str(decode_path),
                "accuracy": data["accuracy"],
                "avg_loss": data["avg_loss"],
                "per_label_accuracy": data["per_label_accuracy"],
                "per_label_total": data["per_label_total"],
                "total_samples": data["total_samples"],
            }
        )

    cur_root = ROOT / "02_curriculum"
    for metrics_path in sorted(cur_root.glob("epoch_*_metrics.json")):
        epoch = int(metrics_path.stem.split("_")[1])
        decode_path = cur_root / f"epoch_{epoch:02d}_decode.jsonl"
        if not decode_path.exists():
            continue
        data = load_json(metrics_path)
        rows.append(
            {
                "strategy": "curriculum",
                "epoch": epoch,
                "metrics_path": str(metrics_path),
                "decode_path": str(decode_path),
                "accuracy": data["accuracy"],
                "avg_loss": data["avg_loss"],
                "per_label_accuracy": data["per_label_accuracy"],
                "per_label_total": data["per_label_total"],
                "total_samples": data["total_samples"],
            }
        )

    rows.sort(key=lambda row: (STRATEGY_ORDER.index(row["strategy"]), row["epoch"]))
    return rows


def enrich_with_decode(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    confusion_rows: list[dict] = []

    for row in rows:
        pred_counts = Counter()
        subset_total = Counter()
        subset_correct = Counter()
        pair_counts = Counter()
        errors = 0

        for item in iter_jsonl(Path(row["decode_path"])):
            pred_counts[item["pred_label"]] += 1
            subset = infer_subset(item["sample_id"])
            subset_total[subset] += 1
            subset_correct[subset] += int(bool(item["correct"]))
            if not item["correct"]:
                errors += 1
                pair_counts[(item["target_label"], item["pred_label"])] += 1

        front_share = pred_counts["front"] / row["total_samples"]
        front_front_right_share = (
            pred_counts["front"] + pred_counts["front-right"]
        ) / row["total_samples"]
        mean_nonfront_acc = mean(
            acc for label, acc in row["per_label_accuracy"].items() if label != "front"
        )

        row["subset_gLOS_acc"] = (
            subset_correct["gLOS"] / subset_total["gLOS"] if subset_total["gLOS"] else None
        )
        row["subset_gnLOS_acc"] = (
            subset_correct["gnLOS"] / subset_total["gnLOS"] if subset_total["gnLOS"] else None
        )
        row["front_share"] = front_share
        row["front_front_right_share"] = front_front_right_share
        row["mean_nonfront_acc"] = mean_nonfront_acc

        for rank, ((target, pred), count) in enumerate(pair_counts.most_common(5), start=1):
            confusion_rows.append(
                {
                    "strategy": row["strategy"],
                    "epoch": row["epoch"],
                    "rank": rank,
                    "target_label": target,
                    "pred_label": pred,
                    "count": count,
                    "error_share": (count / errors) if errors else 0.0,
                }
            )

    return rows, confusion_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    trajectory_rows = scan_metrics()
    trajectory_rows, confusion_rows = enrich_with_decode(trajectory_rows)

    by_strategy = {name: [] for name in STRATEGY_ORDER}
    for row in trajectory_rows:
        by_strategy[row["strategy"]].append(row)

    best_rows = {
        strategy: max(rows, key=lambda row: (row["accuracy"], -row["avg_loss"]))
        for strategy, rows in by_strategy.items()
    }
    final_rows = {
        strategy: max(rows, key=lambda row: row["epoch"]) for strategy, rows in by_strategy.items()
    }

    aligned_rows: list[dict] = []
    label_comparison_rows: list[dict] = []
    subset_rows: list[dict] = []

    row_lookup = {(row["strategy"], row["epoch"]): row for row in trajectory_rows}

    for epoch in ALIGNED_EPOCHS:
        end_row = row_lookup[("endtoend", epoch)]
        cur_row = row_lookup[("curriculum", epoch)]
        aligned_rows.append(
            {
                "epoch": epoch,
                "endtoend_accuracy": end_row["accuracy"],
                "curriculum_accuracy": cur_row["accuracy"],
                "delta_accuracy": cur_row["accuracy"] - end_row["accuracy"],
                "endtoend_loss": end_row["avg_loss"],
                "curriculum_loss": cur_row["avg_loss"],
                "delta_loss": cur_row["avg_loss"] - end_row["avg_loss"],
                "endtoend_front_share": end_row["front_share"],
                "curriculum_front_share": cur_row["front_share"],
                "delta_front_share": cur_row["front_share"] - end_row["front_share"],
                "endtoend_front_front_right_share": end_row["front_front_right_share"],
                "curriculum_front_front_right_share": cur_row["front_front_right_share"],
                "delta_front_front_right_share": cur_row["front_front_right_share"]
                - end_row["front_front_right_share"],
                "endtoend_gLOS_acc": end_row["subset_gLOS_acc"],
                "curriculum_gLOS_acc": cur_row["subset_gLOS_acc"],
                "delta_gLOS_acc": (cur_row["subset_gLOS_acc"] or 0.0)
                - (end_row["subset_gLOS_acc"] or 0.0),
                "endtoend_gnLOS_acc": end_row["subset_gnLOS_acc"],
                "curriculum_gnLOS_acc": cur_row["subset_gnLOS_acc"],
                "delta_gnLOS_acc": (cur_row["subset_gnLOS_acc"] or 0.0)
                - (end_row["subset_gnLOS_acc"] or 0.0),
            }
        )

        for label in LABEL_ORDER:
            label_comparison_rows.append(
                {
                    "scope": f"aligned_epoch_{epoch}",
                    "label": label,
                    "endtoend_acc": end_row["per_label_accuracy"][label],
                    "curriculum_acc": cur_row["per_label_accuracy"][label],
                    "delta_acc": cur_row["per_label_accuracy"][label]
                    - end_row["per_label_accuracy"][label],
                }
            )

        for strategy, row in [("endtoend", end_row), ("curriculum", cur_row)]:
            for subset_name, value in [("gLOS", row["subset_gLOS_acc"]), ("gnLOS", row["subset_gnLOS_acc"])]:
                subset_rows.append(
                    {
                        "scope": f"aligned_epoch_{epoch}",
                        "strategy": strategy,
                        "epoch": epoch,
                        "subset": subset_name,
                        "accuracy": value,
                    }
                )

    for label in LABEL_ORDER:
        label_comparison_rows.append(
            {
                "scope": "best_vs_best",
                "label": label,
                "endtoend_acc": best_rows["endtoend"]["per_label_accuracy"][label],
                "curriculum_acc": best_rows["curriculum"]["per_label_accuracy"][label],
                "delta_acc": best_rows["curriculum"]["per_label_accuracy"][label]
                - best_rows["endtoend"]["per_label_accuracy"][label],
            }
        )

    for strategy, row in best_rows.items():
        for subset_name, value in [("gLOS", row["subset_gLOS_acc"]), ("gnLOS", row["subset_gnLOS_acc"])]:
            subset_rows.append(
                {
                    "scope": "best_vs_best",
                    "strategy": strategy,
                    "epoch": row["epoch"],
                    "subset": subset_name,
                    "accuracy": value,
                }
            )

    trajectory_csv_rows = []
    for row in trajectory_rows:
        csv_row = {
            "strategy": row["strategy"],
            "epoch": row["epoch"],
            "accuracy": round(row["accuracy"], 6),
            "avg_loss": round(row["avg_loss"], 6),
            "front_share": round(row["front_share"], 6),
            "front_front_right_share": round(row["front_front_right_share"], 6),
            "front_acc": round(row["per_label_accuracy"]["front"], 6),
            "mean_nonfront_acc": round(row["mean_nonfront_acc"], 6),
            "subset_gLOS_acc": round(row["subset_gLOS_acc"], 6)
            if row["subset_gLOS_acc"] is not None
            else None,
            "subset_gnLOS_acc": round(row["subset_gnLOS_acc"], 6)
            if row["subset_gnLOS_acc"] is not None
            else None,
        }
        for label in LABEL_ORDER:
            csv_row[f"label_{label}"] = round(row["per_label_accuracy"][label], 6)
        trajectory_csv_rows.append(csv_row)

    aligned_csv_rows = []
    for row in aligned_rows:
        aligned_csv_rows.append(
            {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in row.items()
            }
        )

    label_csv_rows = []
    for row in label_comparison_rows:
        label_csv_rows.append(
            {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in row.items()
            }
        )

    confusion_csv_rows = []
    for row in confusion_rows:
        scope = "aligned" if row["epoch"] in ALIGNED_EPOCHS else "extra"
        if row["epoch"] == best_rows[row["strategy"]]["epoch"]:
            scope = f"{scope}+best"
        confusion_csv_rows.append(
            {
                **row,
                "scope": scope,
                "error_share": round(row["error_share"], 6),
            }
        )

    subset_csv_rows = []
    for row in subset_rows:
        subset_csv_rows.append(
            {
                "scope": row["scope"],
                "strategy": row["strategy"],
                "epoch": row["epoch"],
                "subset": row["subset"],
                "accuracy": round(row["accuracy"], 6) if row["accuracy"] is not None else None,
            }
        )

    write_csv(
        OUTPUT_DIR / "trajectory_summary.csv",
        trajectory_csv_rows,
        list(trajectory_csv_rows[0].keys()),
    )
    write_csv(
        OUTPUT_DIR / "aligned_comparison.csv",
        aligned_csv_rows,
        list(aligned_csv_rows[0].keys()),
    )
    write_csv(
        OUTPUT_DIR / "label_comparison.csv",
        label_csv_rows,
        list(label_csv_rows[0].keys()),
    )
    write_csv(
        OUTPUT_DIR / "top_confusions.csv",
        confusion_csv_rows,
        list(confusion_csv_rows[0].keys()),
    )
    write_csv(
        OUTPUT_DIR / "subset_comparison.csv",
        subset_csv_rows,
        list(subset_csv_rows[0].keys()),
    )

    best_end = best_rows["endtoend"]
    best_cur = best_rows["curriculum"]
    aligned_mean_end = mean(row["endtoend_accuracy"] for row in aligned_rows)
    aligned_mean_cur = mean(row["curriculum_accuracy"] for row in aligned_rows)
    best_gap = best_cur["accuracy"] - best_end["accuracy"]
    best_gap_rel = (best_cur["accuracy"] / best_end["accuracy"]) - 1.0

    overview_rows = []
    for strategy in STRATEGY_ORDER:
        best_row = best_rows[strategy]
        final_row = final_rows[strategy]
        overview_rows.append(
            [
                strategy,
                str(best_row["epoch"]),
                pct(best_row["accuracy"]),
                f"{best_row['avg_loss']:.4f}",
                str(final_row["epoch"]),
                pct(final_row["accuracy"]),
                pct(best_row["subset_gLOS_acc"]),
                pct(best_row["subset_gnLOS_acc"]),
                pct(best_row["front_share"]),
                pct(best_row["front_front_right_share"]),
            ]
        )

    aligned_md_rows = []
    for row in aligned_rows:
        aligned_md_rows.append(
            [
                str(row["epoch"]),
                pct(row["endtoend_accuracy"]),
                pct(row["curriculum_accuracy"]),
                f"{row['delta_accuracy'] * 100:+.2f}pt",
                f"{row['endtoend_loss']:.4f}",
                f"{row['curriculum_loss']:.4f}",
                f"{row['delta_loss']:+.4f}",
                pct(row["endtoend_front_share"]),
                pct(row["curriculum_front_share"]),
            ]
        )

    best_label_rows = []
    best_label_scope = [row for row in label_comparison_rows if row["scope"] == "best_vs_best"]
    best_label_scope.sort(key=lambda row: (row["delta_acc"], row["label"]))
    for row in best_label_scope:
        best_label_rows.append(
            [
                row["label"],
                pct(row["endtoend_acc"]),
                pct(row["curriculum_acc"]),
                f"{row['delta_acc'] * 100:+.2f}pt",
            ]
        )

    subset_md_rows = []
    for row in subset_csv_rows:
        if row["scope"] in {"aligned_epoch_5", "aligned_epoch_10", "aligned_epoch_15", "aligned_epoch_20", "best_vs_best"}:
            subset_md_rows.append(
                [
                    row["scope"],
                    row["strategy"],
                    str(row["epoch"]),
                    row["subset"],
                    pct(row["accuracy"]),
                ]
            )

    confusion_md_rows = []
    for strategy in STRATEGY_ORDER:
        best_epoch = best_rows[strategy]["epoch"]
        top_rows = [
            row
            for row in confusion_rows
            if row["strategy"] == strategy and row["epoch"] == best_epoch
        ]
        text = ", ".join(
            f"{row['target_label']} -> {row['pred_label']} ({row['count']})"
            for row in top_rows[:5]
        )
        confusion_md_rows.append([strategy, str(best_epoch), text])

    key_findings = [
        f"Best validation decode accuracy는 end-to-end가 epoch {best_end['epoch']}에서 {best_end['accuracy'] * 100:.2f}%, curriculum은 epoch {best_cur['epoch']}에서 {best_cur['accuracy'] * 100:.2f}%로, curriculum이 {abs(best_gap) * 100:.2f}pt 낮습니다.",
        f"Aligned epoch(5/10/15/20) 평균 accuracy는 end-to-end {aligned_mean_end * 100:.2f}%, curriculum {aligned_mean_cur * 100:.2f}%로, curriculum이 {abs(aligned_mean_cur - aligned_mean_end) * 100:.2f}pt 낮습니다.",
        f"Best checkpoint 기준 `front` accuracy는 curriculum이 더 높지만({best_cur['per_label_accuracy']['front'] * 100:.2f}% vs {best_end['per_label_accuracy']['front'] * 100:.2f}%), 나머지 7개 방향 평균은 end-to-end {best_end['mean_nonfront_acc'] * 100:.2f}%, curriculum {best_cur['mean_nonfront_acc'] * 100:.2f}%로 크게 뒤집니다.",
        f"Best checkpoint에서 `front + front-right` 예측 비중은 end-to-end {best_end['front_front_right_share'] * 100:.2f}%, curriculum {best_cur['front_front_right_share'] * 100:.2f}%라서, curriculum이 두 전방 클래스에 심하게 쏠려 있습니다.",
        f"Subset 기준 best 성능은 end-to-end가 gLOS {best_end['subset_gLOS_acc'] * 100:.2f}%, gnLOS {best_end['subset_gnLOS_acc'] * 100:.2f}%이고, curriculum은 gLOS {best_cur['subset_gLOS_acc'] * 100:.2f}%, gnLOS {best_cur['subset_gnLOS_acc'] * 100:.2f}%입니다.",
    ]

    recommendation_lines = [
        "- 현재 `02_curriculum`은 baseline 대체안이 아니라 debug 대상입니다.",
        "- 가장 큰 문제는 `front` / `front-right` 쏠림과 `back`, `back-left`, `back-right`, `front-left`, `left`, `right` 정체입니다.",
        "- 다음 실험은 curriculum 순서를 유지하더라도 stage transition 조건에 validation decode gate를 넣고, 각 stage 종료 시 non-front 7개 방향 최소 accuracy를 확인하는 쪽이 안전합니다.",
        "- `front` 단일 성능 상승을 curriculum 성공으로 보면 안 되고, `mean_nonfront_acc`와 `front+front-right` prediction share를 함께 추적해야 합니다.",
    ]

    report_lines = [
        "# Curriculum Baseline Comparison",
        "",
        "## Scope",
        "",
        "- Compared strategies: `01_endtoend`, `02_curriculum`",
        "- Validation set size: 400 samples",
        "- Direction classes: 8-way (`front-left`, `front`, `front-right`, `right`, `back-right`, `back`, `back-left`, `left`)",
        "- Outputs generated alongside this report:",
        "  - `trajectory_summary.csv`",
        "  - `aligned_comparison.csv`",
        "  - `label_comparison.csv`",
        "  - `top_confusions.csv`",
        "  - `subset_comparison.csv`",
        "",
        "## Strategy Overview",
        "",
        markdown_table(
            [
                "Strategy",
                "Best epoch",
                "Best acc",
                "Best loss",
                "Final epoch",
                "Final acc",
                "Best gLOS",
                "Best gnLOS",
                "Front share",
                "Front+FR share",
            ],
            overview_rows,
        ),
        "",
        "## Aligned Epoch Comparison",
        "",
        markdown_table(
            [
                "Epoch",
                "End acc",
                "Cur acc",
                "Delta acc",
                "End loss",
                "Cur loss",
                "Delta loss",
                "End front share",
                "Cur front share",
            ],
            aligned_md_rows,
        ),
        "",
        "## Key Findings",
        "",
        *[f"- {line}" for line in key_findings],
        "",
        "## Best Checkpoint Label Comparison",
        "",
        markdown_table(
            ["Label", "End-to-end", "Curriculum", "Delta"],
            best_label_rows,
        ),
        "",
        "## Subset Accuracy",
        "",
        markdown_table(
            ["Scope", "Strategy", "Epoch", "Subset", "Accuracy"],
            subset_md_rows,
        ),
        "",
        "## Dominant Confusions At Best Epoch",
        "",
        markdown_table(
            ["Strategy", "Best epoch", "Top confusions"],
            confusion_md_rows,
        ),
        "",
        "## Recommendations",
        "",
        *recommendation_lines,
        "",
        "## Reproduction",
        "",
        "```bash",
        f"python3 {ROOT / 'build_curriculum_analysis.py'}",
        "```",
        "",
    ]

    (OUTPUT_DIR / "curriculum_analysis.md").write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
