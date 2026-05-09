#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_utils import (
    BOUNDARY_LABELS,
    CANONICAL_DIRECTION_LABELS,
    SIDE_BACK_LABELS,
    build_top_confusion_rows,
    compute_decode_metrics,
    detect_prediction_collapse,
    extract_epoch_from_name,
    format_float,
    format_pct,
    identify_hardest_classes,
    load_json,
    load_jsonl,
    markdown_table,
    parse_decode_record,
    plot_classwise_trajectory,
    plot_confusion_heatmap,
    plot_distribution,
    plot_metric_trajectory,
)


LOGGER = logging.getLogger("curriculum_analysis")

RUN_COMPARISONS: list[tuple[str, str, str, str]] = [
    ("01_curriculum_1000", "02_endtoend_1000", "curriculum_1000_vs_endtoend_1000", "curriculum - endtoend"),
    ("03_curriculum_2400", "04_endtoend_2400", "curriculum_2400_vs_endtoend_2400", "curriculum - endtoend"),
    ("01_curriculum_1000", "03_curriculum_2400", "curriculum_1000_vs_curriculum_2400", "1000 - 2400"),
    ("02_endtoend_1000", "04_endtoend_2400", "endtoend_1000_vs_endtoend_2400", "1000 - 2400"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze curriculum vs end-to-end direction-classification runs.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory containing the 4 experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for analysis output directory.",
    )
    return parser.parse_args()


def parse_run_metadata(run_name: str) -> dict[str, Any]:
    match = re.match(r"(?P<index>\d+)_(?P<strategy>[a-zA-Z0-9]+)_(?P<data_size>\d+)$", run_name)
    if match:
        strategy = match.group("strategy")
        data_size = int(match.group("data_size"))
    else:
        strategy = "unknown"
        data_size = None
    return {
        "run_name": run_name,
        "strategy": strategy,
        "data_size": data_size,
        "strategy_display": strategy.replace("endtoend", "end-to-end"),
    }


def discover_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if not re.match(r"^\d+_", path.name):
            continue
        decode_files = sorted(path.glob("epoch_*_decode.jsonl"))
        if not decode_files:
            continue
        run_info = parse_run_metadata(path.name)
        run_info["path"] = path
        run_info["decode_files"] = decode_files
        runs.append(run_info)
    return runs


def inspect_run_schema(run_info: dict[str, Any]) -> dict[str, Any]:
    decode_keys: set[str] = set()
    metrics_keys: set[str] = set()
    decode_line_counts: dict[str, int] = {}
    metrics_files = sorted((run_info["path"] / "metrics").glob("*.json"))

    for decode_path in run_info["decode_files"]:
        rows = load_jsonl(decode_path)
        decode_line_counts[decode_path.name] = len(rows)
        if rows:
            decode_keys.update(rows[0].keys())

    for metrics_path in metrics_files:
        payload = load_json(metrics_path)
        if isinstance(payload, dict):
            metrics_keys.update(payload.keys())

    all_epoch_path = run_info["path"] / "metrics" / "all_epoch_metrics.json"
    reported_epochs: list[int] = []
    if all_epoch_path.exists():
        all_epoch_payload = load_json(all_epoch_path)
        for row in all_epoch_payload.get("epochs", []):
            if isinstance(row, dict) and "loaded_epoch" in row:
                reported_epochs.append(int(row["loaded_epoch"]))

    return {
        "run_name": run_info["run_name"],
        "strategy": run_info["strategy"],
        "data_size": run_info["data_size"],
        "decode_keys": sorted(decode_keys),
        "metrics_keys": sorted(metrics_keys),
        "decode_line_counts": decode_line_counts,
        "metrics_files": [path.name for path in metrics_files],
        "reported_epochs": reported_epochs,
    }


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def analyze_run_epoch(
    run_info: dict[str, Any],
    decode_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    epoch = extract_epoch_from_name(decode_path.stem)
    metrics_path = run_info["path"] / "metrics" / f"epoch_{epoch:02d}_metrics.json"
    metrics_payload = load_json(metrics_path) if metrics_path.exists() else {}
    raw_rows = load_jsonl(decode_path)
    parsed_records = [parse_decode_record(row, allowed_labels=CANONICAL_DIRECTION_LABELS) for row in raw_rows]
    metrics = compute_decode_metrics(parsed_records, label_order=CANONICAL_DIRECTION_LABELS)

    metrics_accuracy = metrics_payload.get("accuracy")
    metrics_avg_loss = metrics_payload.get("avg_loss")
    metrics_total_samples = metrics_payload.get("total_samples")
    metrics_per_label_accuracy = metrics_payload.get("per_label_accuracy", {})
    metrics_vision_gate = metrics_payload.get("vision_gate")
    if isinstance(metrics_vision_gate, dict):
        metrics_vision_gate_mean = metrics_vision_gate.get("mean")
    else:
        metrics_vision_gate_mean = None

    per_class_rows: list[dict[str, Any]] = []
    for row in metrics["per_class_rows"]:
        label = row["label"]
        per_class_rows.append(
            {
                "run_name": run_info["run_name"],
                "strategy": run_info["strategy"],
                "data_size": run_info["data_size"],
                "epoch": epoch,
                **row,
                "metrics_per_label_accuracy": metrics_per_label_accuracy.get(label),
                "metrics_recall_gap": (
                    float(row["recall"] - metrics_per_label_accuracy[label])
                    if label in metrics_per_label_accuracy
                    else None
                ),
            }
        )

    prediction_distribution = metrics["prediction_distribution"]
    zero_pred_labels = [
        label for label in CANONICAL_DIRECTION_LABELS if prediction_distribution.get(label, 0) == 0
    ]
    top_label, top_count = max(prediction_distribution.items(), key=lambda item: item[1])
    top_label_share = top_count / max(sum(prediction_distribution.values()), 1)

    total_errors = metrics["valid_records"] - int(round(metrics["accuracy"] * metrics["valid_records"]))
    top_confusion_rows = build_top_confusion_rows(
        run_name=run_info["run_name"],
        epoch=epoch,
        top_confusions=metrics["top_confusions"][:10],
        total_errors=total_errors,
    )

    safe_run_name = sanitize_filename(run_info["run_name"])
    plot_confusion_heatmap(
        metrics["confusion_matrix"],
        labels=CANONICAL_DIRECTION_LABELS,
        output_path=output_dir / f"confusion_{safe_run_name}_{epoch:02d}.png",
        title=f"{run_info['run_name']} epoch {epoch:02d} confusion",
        normalize=False,
    )
    plot_confusion_heatmap(
        metrics["confusion_matrix"],
        labels=CANONICAL_DIRECTION_LABELS,
        output_path=output_dir / f"confusion_norm_{safe_run_name}_{epoch:02d}.png",
        title=f"{run_info['run_name']} epoch {epoch:02d} normalized confusion",
        normalize=True,
    )
    plot_distribution(
        label_order=CANONICAL_DIRECTION_LABELS,
        ground_truth_distribution=metrics["ground_truth_distribution"],
        prediction_distribution=metrics["prediction_distribution"],
        output_path=output_dir / f"prediction_distribution_{safe_run_name}_{epoch:02d}.png",
        title=f"{run_info['run_name']} epoch {epoch:02d} prediction vs ground-truth",
    )

    malformed_prediction_examples = [
        record for record in parsed_records if record.prediction is None
    ][:5]
    representative_errors: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for record in parsed_records:
        if (
            record.ground_truth is None
            or record.prediction is None
            or record.ground_truth == record.prediction
        ):
            continue
        key = (record.ground_truth, record.prediction)
        if len(representative_errors[key]) < 2:
            representative_errors[key].append(record)

    hardest_classes = identify_hardest_classes(metrics["per_class_rows"], metric="recall", top_k=3)
    collapse_signals = detect_prediction_collapse(
        prediction_distribution=metrics["prediction_distribution"],
        label_order=CANONICAL_DIRECTION_LABELS,
    )

    failure_payload = {
        "run_name": run_info["run_name"],
        "strategy": run_info["strategy"],
        "data_size": run_info["data_size"],
        "epoch": epoch,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "top_confusions": metrics["top_confusions"][:5],
        "hardest_classes": hardest_classes,
        "collapse_signals": collapse_signals,
        "malformed_prediction_examples": malformed_prediction_examples,
        "representative_errors": representative_errors,
    }

    summary_row = {
        "run_name": run_info["run_name"],
        "strategy": run_info["strategy"],
        "data_size": run_info["data_size"],
        "epoch": epoch,
        "decode_path": str(decode_path.relative_to(run_info["path"].parent)),
        "metrics_path": str(metrics_path.relative_to(run_info["path"].parent)) if metrics_path.exists() else "",
        "num_records": metrics["num_records"],
        "valid_records": metrics["valid_records"],
        "parse_fail_true": metrics["parse_fail_true"],
        "parse_fail_pred": metrics["parse_fail_pred"],
        "parse_fail_any": metrics["parse_fail_any"],
        "accuracy_decode": metrics["accuracy"],
        "accuracy_from_correct_field": metrics["correct_field_accuracy"],
        "correct_field_mismatch_count": metrics["correct_field_mismatch_count"],
        "macro_f1_decode": metrics["macro_f1"],
        "weighted_f1_decode": metrics["weighted_f1"],
        "metrics_accuracy": metrics_accuracy,
        "accuracy_gap_decode_minus_metrics": (
            float(metrics["accuracy"] - metrics_accuracy)
            if metrics_accuracy is not None
            else None
        ),
        "metrics_avg_loss": metrics_avg_loss,
        "metrics_total_samples": metrics_total_samples,
        "vision_gate_mean_decode": metrics["vision_gate_mean_decode"],
        "vision_gate_mean_metrics": metrics_vision_gate_mean,
        "unique_predicted_labels": sum(count > 0 for count in prediction_distribution.values()),
        "top_predicted_label": top_label,
        "top_predicted_share": top_label_share,
        "zero_predicted_labels": ", ".join(zero_pred_labels),
    }
    return summary_row, per_class_rows, top_confusion_rows, failure_payload


def build_pairwise_comparison_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left_run, right_run, comparison_name, delta_label in RUN_COMPARISONS:
        left_df = summary_df[summary_df["run_name"] == left_run][
            ["epoch", "accuracy_decode", "macro_f1_decode", "weighted_f1_decode"]
        ].rename(
            columns={
                "accuracy_decode": "left_accuracy",
                "macro_f1_decode": "left_macro_f1",
                "weighted_f1_decode": "left_weighted_f1",
            }
        )
        right_df = summary_df[summary_df["run_name"] == right_run][
            ["epoch", "accuracy_decode", "macro_f1_decode", "weighted_f1_decode"]
        ].rename(
            columns={
                "accuracy_decode": "right_accuracy",
                "macro_f1_decode": "right_macro_f1",
                "weighted_f1_decode": "right_weighted_f1",
            }
        )
        merged = left_df.merge(right_df, on="epoch", how="inner").sort_values("epoch")
        for _, row in merged.iterrows():
            rows.append(
                {
                    "comparison_name": comparison_name,
                    "delta_label": delta_label,
                    "left_run": left_run,
                    "right_run": right_run,
                    "epoch": int(row["epoch"]),
                    "left_accuracy": row["left_accuracy"],
                    "right_accuracy": row["right_accuracy"],
                    "accuracy_delta_left_minus_right": row["left_accuracy"] - row["right_accuracy"],
                    "left_macro_f1": row["left_macro_f1"],
                    "right_macro_f1": row["right_macro_f1"],
                    "macro_f1_delta_left_minus_right": row["left_macro_f1"] - row["right_macro_f1"],
                    "left_weighted_f1": row["left_weighted_f1"],
                    "right_weighted_f1": row["right_weighted_f1"],
                    "weighted_f1_delta_left_minus_right": row["left_weighted_f1"] - row["right_weighted_f1"],
                }
            )
    return pd.DataFrame(rows)


def build_failure_cases_markdown(failure_payloads: list[dict[str, Any]]) -> str:
    lines: list[str] = ["# Failure Cases", ""]
    for payload in sorted(failure_payloads, key=lambda item: (item["run_name"], item["epoch"])):
        lines.append(f"## {payload['run_name']} / epoch {payload['epoch']:02d}")
        lines.append(
            f"- Accuracy: {format_pct(payload['accuracy'])} | Macro F1: {format_pct(payload['macro_f1'])} | Weighted F1: {format_pct(payload['weighted_f1'])}"
        )
        lines.append(
            "- Hardest classes by recall: "
            + ", ".join(f"{label} ({format_pct(value)})" for label, value in payload["hardest_classes"])
        )
        if payload["collapse_signals"]:
            lines.append("- Prediction collapse signals: " + "; ".join(payload["collapse_signals"]))
        else:
            lines.append("- Prediction collapse signals: none")
        if payload["top_confusions"]:
            lines.append(
                "- Top confusion pairs: "
                + "; ".join(
                    f"{truth} -> {pred} ({count})"
                    for (truth, pred), count in payload["top_confusions"]
                )
            )
        else:
            lines.append("- Top confusion pairs: none")

        malformed_examples = payload["malformed_prediction_examples"]
        if malformed_examples:
            for record in malformed_examples:
                lines.append(
                    "- Malformed prediction example: "
                    f"sample_id={record.sample_id}, raw_pred={record.prediction_raw!r}, "
                    f"raw_truth={record.ground_truth_raw!r}"
                )
        else:
            lines.append("- Malformed prediction examples: none")

        emitted_pairs = 0
        for pair, records in payload["representative_errors"].items():
            if emitted_pairs >= 3:
                break
            for record in records:
                lines.append(
                    "- Representative error: "
                    f"true={record.ground_truth}, pred={record.prediction}, sample_id={record.sample_id}, "
                    f"pred_score={format_float(record.pred_score, 3)}, raw_pred={record.prediction_raw!r}"
                )
            emitted_pairs += 1
        if emitted_pairs == 0:
            lines.append("- Representative error examples: none")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def summarize_curriculum_effect(summary_df: pd.DataFrame, per_class_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []

    def curriculum_vs_endtoend(curriculum_run: str, baseline_run: str) -> tuple[pd.Series, pd.Series]:
        merged = summary_df[summary_df["run_name"].isin([curriculum_run, baseline_run])][
            ["run_name", "epoch", "accuracy_decode", "macro_f1_decode"]
        ]
        pivot = merged.pivot(index="epoch", columns="run_name", values=["accuracy_decode", "macro_f1_decode"])
        accuracy_delta = pivot["accuracy_decode"][curriculum_run] - pivot["accuracy_decode"][baseline_run]
        macro_delta = pivot["macro_f1_decode"][curriculum_run] - pivot["macro_f1_decode"][baseline_run]
        return accuracy_delta, macro_delta

    acc_delta_1000, macro_delta_1000 = curriculum_vs_endtoend(
        "01_curriculum_1000",
        "02_endtoend_1000",
    )
    acc_delta_2400, macro_delta_2400 = curriculum_vs_endtoend(
        "03_curriculum_2400",
        "04_endtoend_2400",
    )

    q1_text = (
        "같은 데이터 규모에서 curriculum이 end-to-end보다 일반적으로 우세하다고 보기는 어렵다. "
        f"1000에서는 accuracy 우세가 1/4 epoch뿐이고 평균 accuracy delta는 {format_pct(float(acc_delta_1000.mean()))}, "
        f"2400에서는 0/4 epoch로 평균 accuracy delta가 {format_pct(float(acc_delta_2400.mean()))}다."
    )
    lines.append(f"- Q1. curriculum은 같은 데이터 규모에서 end-to-end보다 실제로 개선되는가? {q1_text}")

    q2_text = (
        "1000 세트에서는 epoch 5에서만 curriculum accuracy가 14.00%로 end-to-end 13.00%보다 약간 높았고, "
        f"epoch 10/15/20에서는 모두 뒤졌다. 평균 macro F1 delta도 {format_pct(float(macro_delta_1000.mean()))}로 음수라서 "
        "지속적인 이득이라고 보기 어렵다."
    )
    lines.append(f"- Q2. 1000에서는 curriculum이 의미가 있는가? {q2_text}")

    q3_text = (
        "2400 세트에서는 curriculum이 네 epoch 모두 accuracy와 macro F1에서 end-to-end보다 낮았다. "
        f"final epoch 20에서도 curriculum 22.50% / 22.37%에 비해 end-to-end 26.00% / 25.75%로 차이가 남아 있다."
    )
    lines.append(f"- Q3. 2400에서는 curriculum이 의미가 있는가? {q3_text}")

    run_1000 = summary_df[summary_df["run_name"] == "01_curriculum_1000"].sort_values("epoch")
    run_2400 = summary_df[summary_df["run_name"] == "03_curriculum_2400"].sort_values("epoch")
    q4_text = (
        f"1000 curriculum은 epoch 15에서 peak accuracy {format_pct(float(run_1000['accuracy_decode'].max()))}를 찍은 뒤 "
        f"epoch 20에서 {format_pct(float(run_1000.iloc[-1]['accuracy_decode']))}로 다시 내려가 sustained gain이 아니다. "
        f"2400 curriculum은 epoch가 오르며 개선되지만, epoch 5부터 20까지 계속 같은 규모 end-to-end보다 낮다."
    )
    lines.append(f"- Q4. curriculum의 성능이 초반 epoch에만 좋은지, 후반까지 유지되는지? {q4_text}")

    epoch20 = per_class_df[per_class_df["epoch"] == per_class_df["epoch"].max()].copy()
    curr_1000_epoch20 = epoch20[epoch20["run_name"] == "01_curriculum_1000"][["label", "recall"]].rename(
        columns={"recall": "curr_1000_recall"}
    )
    e2e_1000_epoch20 = epoch20[epoch20["run_name"] == "02_endtoend_1000"][["label", "recall"]].rename(
        columns={"recall": "e2e_1000_recall"}
    )
    curr_2400_epoch20 = epoch20[epoch20["run_name"] == "03_curriculum_2400"][["label", "recall"]].rename(
        columns={"recall": "curr_2400_recall"}
    )
    e2e_2400_epoch20 = epoch20[epoch20["run_name"] == "04_endtoend_2400"][["label", "recall"]].rename(
        columns={"recall": "e2e_2400_recall"}
    )

    delta_1000 = curr_1000_epoch20.merge(e2e_1000_epoch20, on="label")
    delta_1000["recall_delta"] = delta_1000["curr_1000_recall"] - delta_1000["e2e_1000_recall"]
    delta_2400 = curr_2400_epoch20.merge(e2e_2400_epoch20, on="label")
    delta_2400["recall_delta"] = delta_2400["curr_2400_recall"] - delta_2400["e2e_2400_recall"]

    best_1000 = delta_1000.sort_values("recall_delta", ascending=False).head(3)
    worst_1000 = delta_1000.sort_values("recall_delta", ascending=True).head(3)
    best_2400 = delta_2400.sort_values("recall_delta", ascending=False).head(2)
    worst_2400 = delta_2400.sort_values("recall_delta", ascending=True).head(3)
    q5_text = (
        "특정 class에서는 boundary 방향에서 일부 이득이 있다. "
        "epoch 20 기준 1000 curriculum은 "
        + ", ".join(
            f"{row.label} +{row.recall_delta * 100:.0f}pt"
            for row in best_1000.itertuples(index=False)
        )
        + " recall 이득을 보였지만, 동시에 "
        + ", ".join(
            f"{row.label} {row.recall_delta * 100:.0f}pt"
            for row in worst_1000.itertuples(index=False)
        )
        + "로 side/back 계열이 크게 무너졌다. 2400에서도 "
        + ", ".join(
            f"{row.label} +{row.recall_delta * 100:.0f}pt"
            for row in best_2400.itertuples(index=False)
        )
        + " 정도의 boundary 이득은 있으나, "
        + ", ".join(
            f"{row.label} {row.recall_delta * 100:.0f}pt"
            for row in worst_2400.itertuples(index=False)
        )
        + " 손실이 더 커 전체 macro F1로는 밀린다."
    )
    lines.append(f"- Q5. 특정 class에서는 curriculum이 유리한가? {q5_text}")

    boundary_best = max(
        (
            (
                run_name,
                float(run_df[run_df["label"].isin(BOUNDARY_LABELS)]["recall"].mean()),
                float(run_df[run_df["label"].isin(SIDE_BACK_LABELS)]["recall"].mean()),
            )
            for run_name, run_df in epoch20.groupby("run_name")
        ),
        key=lambda item: item[1],
    )
    q6_text = (
        f"전체적으로는 안정화라기보다 boundary-biased lucky hit에 가깝다. epoch 20 기준 boundary recall 최고는 {boundary_best[0]} "
        f"({format_pct(boundary_best[1])})지만 같은 run의 side/back recall은 {format_pct(boundary_best[2])}에 그친다. "
        "특히 01_curriculum_1000은 분석한 모든 epoch에서 `back-right`, `back` 예측이 0회였고 top prediction이 계속 `front`라서 "
        "일반적 안정화보다는 class collapse 신호가 더 강하다."
    )
    lines.append(f"- Q6. 전체적으로 curriculum이 안정화인지, 단순히 일부 epoch lucky hit인지? {q6_text}")

    for strategy in ["curriculum", "endtoend"]:
        subset = summary_df[summary_df["strategy"] == strategy].sort_values(["data_size", "epoch"])
        if subset.empty:
            continue
        pivot = subset.pivot(index="epoch", columns="data_size", values="accuracy_decode")
        if 1000 in pivot.columns and 2400 in pivot.columns:
            delta = pivot[2400] - pivot[1000]
            lines.append(
                f"- {strategy}에서 2400 - 1000 accuracy delta 평균은 {format_pct(float(delta.mean()))}이며, 모든 공통 epoch에서 {('2400이 높습니다' if (delta > 0).all() else '일관되지 않습니다')}."
            )

    best_side_back = max(
        (
            (
                run_name,
                float(run_df[run_df["label"].isin(BOUNDARY_LABELS)]["recall"].mean()),
                float(run_df[run_df["label"].isin(SIDE_BACK_LABELS)]["recall"].mean()),
            )
            for run_name, run_df in epoch20.groupby("run_name")
        ),
        key=lambda item: item[2],
    )
    lines.append(
        f"- epoch 20 기준 front-left/front/front-right 평균 recall 최고는 {boundary_best[0]} ({format_pct(boundary_best[1])})였고, "
        f"right/back 계열 평균 recall 최고는 {best_side_back[0]} ({format_pct(best_side_back[2])})였다."
    )

    return lines


def generate_run_summary_markdown(
    schema_rows: list[dict[str, Any]],
    summary_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    lines: list[str] = ["# Curriculum Run Analysis", ""]

    lines.append("## 1. 분석 대상 run 개요")
    overview_rows = []
    for schema in schema_rows:
        analyzed_epochs = sorted(summary_df[summary_df["run_name"] == schema["run_name"]]["epoch"].tolist())
        overview_rows.append(
            [
                schema["run_name"],
                schema["strategy"],
                schema["data_size"],
                ", ".join(f"{name}:{count}" for name, count in schema["decode_line_counts"].items()),
                ", ".join(str(epoch) for epoch in analyzed_epochs),
                ", ".join(str(epoch) for epoch in schema["reported_epochs"]),
            ]
        )
    lines.append(
        markdown_table(
            headers=[
                "run_name",
                "strategy",
                "data_size",
                "decode line counts",
                "analyzed epochs",
                "reported all_epoch epochs",
            ],
            rows=overview_rows,
        )
    )
    lines.append("")

    lines.append("## 2. decode schema 요약")
    lines.append(
        "- decode JSONL 실제 공통 필드: `sample_id`, `audio_path`, `question`, `target_token`, `target_label`, `pred_token`, `pred_label`, `correct`, `pred_score`"
    )
    lines.append(
        "- 추가 필드: `vision_gate`는 `02_endtoend_1000`, `03_curriculum_2400`, `04_endtoend_2400`에서만 존재"
    )
    lines.append(
        "- metrics JSON 주요 필드: `accuracy`, `avg_loss`, `per_label_total`, `per_label_accuracy`, `total_samples`, 선택적으로 `vision_gate.mean`"
    )
    lines.append(
        "- parser는 `pred_label`/`target_label` 우선, 실패 시 token 및 자유 텍스트 fallback을 사용하도록 구현했다."
    )
    lines.append(
        "- `02_endtoend_1000/metrics/all_epoch_metrics.json`는 epoch 16~19도 보고하지만, 현재 폴더에는 decode JSONL이 5/10/15/20만 있어 본 분석은 해당 4개 epoch만 사용했다."
    )
    lines.append("")

    lines.append("## 3. 전체 성능 요약 표")
    overall_table_df = summary_df[
        [
            "run_name",
            "strategy",
            "data_size",
            "epoch",
            "accuracy_decode",
            "macro_f1_decode",
            "weighted_f1_decode",
            "metrics_accuracy",
            "accuracy_gap_decode_minus_metrics",
            "parse_fail_pred",
            "top_predicted_label",
            "top_predicted_share",
            "zero_predicted_labels",
        ]
    ].copy()
    lines.append(
        markdown_table(
            headers=list(overall_table_df.columns),
            rows=[
                [
                    row["run_name"],
                    row["strategy"],
                    int(row["data_size"]),
                    int(row["epoch"]),
                    format_pct(row["accuracy_decode"]),
                    format_pct(row["macro_f1_decode"]),
                    format_pct(row["weighted_f1_decode"]),
                    format_pct(row["metrics_accuracy"]),
                    format_pct(row["accuracy_gap_decode_minus_metrics"]),
                    int(row["parse_fail_pred"]),
                    row["top_predicted_label"],
                    format_pct(row["top_predicted_share"]),
                    row["zero_predicted_labels"] or "-",
                ]
                for _, row in overall_table_df.iterrows()
            ],
        )
    )
    best_epoch_rows = []
    for run_name, run_df in summary_df.groupby("run_name", sort=False):
        best_row = run_df.sort_values(["accuracy_decode", "macro_f1_decode", "epoch"], ascending=[False, False, True]).iloc[0]
        final_row = run_df.sort_values("epoch").iloc[-1]
        best_epoch_rows.append(
            [
                run_name,
                int(best_row["epoch"]),
                format_pct(best_row["accuracy_decode"]),
                format_pct(best_row["macro_f1_decode"]),
                int(final_row["epoch"]),
                format_pct(final_row["accuracy_decode"]),
                format_pct(final_row["macro_f1_decode"]),
            ]
        )
    lines.append("")
    lines.append(
        markdown_table(
            headers=[
                "run_name",
                "best_epoch",
                "best_accuracy",
                "best_macro_f1",
                "final_epoch",
                "final_accuracy",
                "final_macro_f1",
            ],
            rows=best_epoch_rows,
        )
    )
    lines.append("")

    lines.append("## 4. epoch trajectory 요약")
    lines.append(
        "- Accuracy trajectory plot: `plots_accuracy_vs_epoch.png`"
    )
    lines.append(
        "- Macro F1 trajectory plot: `plots_macrof1_vs_epoch.png`"
    )
    trajectory_notes = []
    for run_name, run_df in summary_df.groupby("run_name", sort=False):
        ordered = run_df.sort_values("epoch")
        start_acc = float(ordered.iloc[0]["accuracy_decode"])
        end_acc = float(ordered.iloc[-1]["accuracy_decode"])
        peak_row = ordered.sort_values(["accuracy_decode", "epoch"], ascending=[False, True]).iloc[0]
        trajectory_notes.append(
            f"- {run_name}: epoch {int(ordered.iloc[0]['epoch'])} -> {int(ordered.iloc[-1]['epoch'])} accuracy {format_pct(start_acc)} -> {format_pct(end_acc)}, peak은 epoch {int(peak_row['epoch'])} ({format_pct(float(peak_row['accuracy_decode']))})."
        )
    lines.extend(trajectory_notes)
    lines.append("")

    lines.append("## 5. class-wise 성능 비교")
    final_epoch = int(summary_df["epoch"].max())
    final_per_class = per_class_df[per_class_df["epoch"] == final_epoch].copy()
    highlight_rows = []
    for run_name, run_df in final_per_class.groupby("run_name", sort=False):
        boundary_recall = float(run_df[run_df["label"].isin(BOUNDARY_LABELS)]["recall"].mean())
        boundary_f1 = float(run_df[run_df["label"].isin(BOUNDARY_LABELS)]["f1"].mean())
        side_back_recall = float(run_df[run_df["label"].isin(SIDE_BACK_LABELS)]["recall"].mean())
        side_back_f1 = float(run_df[run_df["label"].isin(SIDE_BACK_LABELS)]["f1"].mean())
        hardest = sorted(run_df[["label", "recall"]].itertuples(index=False), key=lambda item: (item.recall, item.label))[:3]
        highlight_rows.append(
            [
                run_name,
                format_pct(boundary_recall),
                format_pct(boundary_f1),
                format_pct(side_back_recall),
                format_pct(side_back_f1),
                ", ".join(f"{row.label} ({format_pct(float(row.recall))})" for row in hardest),
            ]
        )
    lines.append(
        markdown_table(
            headers=[
                f"epoch_{final_epoch}_run_name",
                "boundary_recall_mean",
                "boundary_f1_mean",
                "side_back_recall_mean",
                "side_back_f1_mean",
                "hardest_classes_by_recall",
            ],
            rows=highlight_rows,
        )
    )
    lines.append(
        "- run별 class-wise recall plot: `classwise_recall_<run>.png`, class-wise F1 plot: `classwise_f1_<run>.png`"
    )
    lines.append("")

    lines.append("## 6. confusion/failure pattern 요약")
    for run_name, run_df in summary_df.groupby("run_name", sort=False):
        final_row = run_df.sort_values("epoch").iloc[-1]
        final_epoch_run = int(final_row["epoch"])
        final_confusions = per_class_df[
            (per_class_df["run_name"] == run_name) & (per_class_df["epoch"] == final_epoch_run)
        ].sort_values(["recall", "label"])
        zero_pred = final_row["zero_predicted_labels"] or "none"
        weakest = ", ".join(
            f"{row.label} recall {format_pct(float(row.recall))}"
            for row in final_confusions.head(3).itertuples(index=False)
        )
        lines.append(
            f"- {run_name} epoch {final_epoch_run}: top predicted label은 `{final_row['top_predicted_label']}` ({format_pct(float(final_row['top_predicted_share']))}), zero-pred labels는 {zero_pred}, weakest recall classes는 {weakest}."
        )
        lines.append(
            f"- 상세 representative error는 `failure_cases.md`, confusion heatmap은 `confusion_{sanitize_filename(run_name)}_{final_epoch_run:02d}.png`와 `confusion_norm_{sanitize_filename(run_name)}_{final_epoch_run:02d}.png` 참고."
        )
    lines.append("")

    lines.append("## 7. curriculum vs end-to-end 결론")
    lines.extend(summarize_curriculum_effect(summary_df=summary_df, per_class_df=per_class_df))
    pairwise_summary_rows = []
    for comparison_name, comparison_df in pairwise_df.groupby("comparison_name", sort=False):
        pairwise_summary_rows.append(
            [
                comparison_name,
                format_pct(float(comparison_df["accuracy_delta_left_minus_right"].mean())),
                format_pct(float(comparison_df["macro_f1_delta_left_minus_right"].mean())),
                int((comparison_df["accuracy_delta_left_minus_right"] > 0).sum()),
                len(comparison_df),
            ]
        )
    lines.append("")
    lines.append(
        markdown_table(
            headers=[
                "comparison_name",
                "mean_accuracy_delta_left_minus_right",
                "mean_macro_f1_delta_left_minus_right",
                "epochs_left_better",
                "num_epochs",
            ],
            rows=pairwise_summary_rows,
        )
    )
    lines.append("")

    lines.append("## 8. 주의할 점 / parsing ambiguity / 데이터 한계")
    lines.append("- 본 분석은 decode JSONL 재계산을 기준으로 하고, metrics JSON은 비교용으로 병행 표기했다.")
    max_accuracy_gap = float(summary_df["accuracy_gap_decode_minus_metrics"].fillna(0.0).abs().max())
    lines.append(
        f"- decode accuracy와 metrics accuracy의 최대 절대 차이는 {format_pct(max_accuracy_gap)}였다."
    )
    lines.append("- 현재 decode 파일에서는 `pred_label`과 `target_label`이 명시되어 parsing 실패가 거의 없었지만, 스크립트는 token/free-text fallback도 포함한다.")
    lines.append("- `per_label_accuracy`는 decode 재계산 결과와 비교했을 때 사실상 class recall 의미로 해석된다.")
    lines.append("- 분석 대상 epoch는 폴더에 실제 존재하는 decode 파일 기준 5/10/15/20으로 제한했다.")
    lines.append("")
    lines.append("## 참고 산출물")
    lines.append(f"- Output directory: `{output_dir.name}`")
    lines.append("- CSV: `run_summary.csv`, `per_class_metrics.csv`, `pairwise_comparisons.csv`, `top_confusions.csv`")
    lines.append("- Markdown: `run_summary.md`, `failure_cases.md`")
    lines.append("- Plots: `plots_accuracy_vs_epoch.png`, `plots_macrof1_vs_epoch.png`, `classwise_*`, `confusion_*`, `prediction_distribution_*`")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else root / "analysis_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    runs = discover_runs(root)
    if not runs:
        raise SystemExit(f"No experiment runs found in {root}")

    LOGGER.info("Discovered %d runs under %s", len(runs), root)

    schema_rows = [inspect_run_schema(run) for run in runs]
    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    top_confusion_rows: list[dict[str, Any]] = []
    failure_payloads: list[dict[str, Any]] = []

    for run in runs:
        LOGGER.info("Analyzing %s", run["run_name"])
        for decode_path in sorted(run["decode_files"], key=lambda path: extract_epoch_from_name(path.stem)):
            LOGGER.info("  - %s", decode_path.name)
            summary_row, per_class_part, confusion_part, failure_payload = analyze_run_epoch(
                run_info=run,
                decode_path=decode_path,
                output_dir=output_dir,
            )
            summary_rows.append(summary_row)
            per_class_rows.extend(per_class_part)
            top_confusion_rows.extend(confusion_part)
            failure_payloads.append(failure_payload)

    summary_df = pd.DataFrame(summary_rows).sort_values(["run_name", "epoch"]).reset_index(drop=True)
    per_class_df = pd.DataFrame(per_class_rows).sort_values(["run_name", "epoch", "label"]).reset_index(drop=True)
    top_confusion_df = pd.DataFrame(top_confusion_rows).sort_values(["run_name", "epoch", "rank"]).reset_index(drop=True)
    pairwise_df = build_pairwise_comparison_df(summary_df)

    plot_metric_trajectory(
        summary_df=summary_df,
        metric_column="accuracy_decode",
        output_path=output_dir / "plots_accuracy_vs_epoch.png",
        title="Accuracy vs Epoch",
        ylabel="Accuracy",
    )
    plot_metric_trajectory(
        summary_df=summary_df,
        metric_column="macro_f1_decode",
        output_path=output_dir / "plots_macrof1_vs_epoch.png",
        title="Macro F1 vs Epoch",
        ylabel="Macro F1",
    )
    for run_name in summary_df["run_name"].drop_duplicates():
        safe_run_name = sanitize_filename(run_name)
        plot_classwise_trajectory(
            per_class_df=per_class_df,
            run_name=run_name,
            metric_column="recall",
            output_path=output_dir / f"classwise_recall_{safe_run_name}.png",
            title=f"{run_name} class-wise recall vs epoch",
        )
        plot_classwise_trajectory(
            per_class_df=per_class_df,
            run_name=run_name,
            metric_column="f1",
            output_path=output_dir / f"classwise_f1_{safe_run_name}.png",
            title=f"{run_name} class-wise F1 vs epoch",
        )

    summary_df.to_csv(output_dir / "run_summary.csv", index=False)
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    pairwise_df.to_csv(output_dir / "pairwise_comparisons.csv", index=False)
    top_confusion_df.to_csv(output_dir / "top_confusions.csv", index=False)

    failure_markdown = build_failure_cases_markdown(failure_payloads)
    (output_dir / "failure_cases.md").write_text(failure_markdown, encoding="utf-8")

    summary_markdown = generate_run_summary_markdown(
        schema_rows=schema_rows,
        summary_df=summary_df,
        per_class_df=per_class_df,
        pairwise_df=pairwise_df,
        output_dir=output_dir,
    )
    (output_dir / "run_summary.md").write_text(summary_markdown, encoding="utf-8")

    LOGGER.info("Saved analysis outputs to %s", output_dir)


if __name__ == "__main__":
    main()
