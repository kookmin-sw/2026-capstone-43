#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


CANONICAL_DIRECTION_LABELS: list[str] = [
    "front-left",
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
]
BOUNDARY_LABELS: list[str] = ["front-left", "front", "front-right"]
SIDE_BACK_LABELS: list[str] = ["right", "back-right", "back", "back-left", "left"]

TRUE_LABEL_KEYS: list[str] = [
    "target_label",
    "ground_truth",
    "groundtruth",
    "gold_label",
    "gt_label",
    "true_label",
    "target",
    "label",
    "answer",
    "reference",
    "expected_label",
]
TRUE_TOKEN_KEYS: list[str] = [
    "target_token",
    "ground_truth_token",
    "gt_token",
    "label_token",
]
PRED_LABEL_KEYS: list[str] = [
    "pred_label",
    "prediction",
    "predicted_label",
    "pred",
    "model_prediction",
    "model_output",
    "response",
    "output",
    "generated_text",
    "completion",
    "answer",
]
PRED_TOKEN_KEYS: list[str] = [
    "pred_token",
    "prediction_token",
    "output_token",
]
PROMPT_KEYS: list[str] = [
    "question",
    "prompt",
    "instruction",
    "input",
    "text",
]
SAMPLE_ID_KEYS: list[str] = ["sample_id", "id", "uid"]
AUDIO_PATH_KEYS: list[str] = ["audio_path", "audio", "wav_path"]
CORRECT_KEYS: list[str] = ["correct", "is_correct"]
PRED_SCORE_KEYS: list[str] = ["pred_score", "score", "confidence"]
VISION_GATE_KEYS: list[str] = ["vision_gate"]


def _build_label_aliases() -> dict[str, str]:
    alias_to_label: dict[str, str] = {}
    for label in CANONICAL_DIRECTION_LABELS:
        parts = label.split("-")
        variants = {
            label,
            label.replace("-", " "),
            label.replace("-", "_"),
            "".join(parts),
        }
        if len(parts) == 2:
            reverse_join = f"{parts[1]} {parts[0]}"
            variants.update(
                {
                    reverse_join,
                    reverse_join.replace(" ", "-"),
                    reverse_join.replace(" ", "_"),
                    "".join(reversed(parts)),
                }
            )
        for variant in variants:
            alias_to_label[variant] = label
    return alias_to_label


ALIAS_TO_LABEL: dict[str, str] = _build_label_aliases()
BIGRAM_TO_LABEL: dict[str, str] = {
    "front left": "front-left",
    "left front": "front-left",
    "front right": "front-right",
    "right front": "front-right",
    "back left": "back-left",
    "left back": "back-left",
    "back right": "back-right",
    "right back": "back-right",
}
UNIGRAM_TO_LABEL: dict[str, str] = {
    "front": "front",
    "right": "right",
    "back": "back",
    "left": "left",
}
TOKEN_TO_LABEL: dict[str, str] = {
    "DIR_H_FRONT_LEFT": "front-left",
    "DIR_H_FRONT": "front",
    "DIR_H_FRONT_RIGHT": "front-right",
    "DIR_H_RIGHT": "right",
    "DIR_H_BACK_RIGHT": "back-right",
    "DIR_H_BACK": "back",
    "DIR_H_BACK_LEFT": "back-left",
    "DIR_H_LEFT": "left",
}


@dataclass
class ParsedRecord:
    sample_id: str
    audio_path: str | None
    question: str | None
    ground_truth: str | None
    prediction: str | None
    ground_truth_raw: Any = None
    prediction_raw: Any = None
    ground_truth_source: str | None = None
    prediction_source: str | None = None
    correct_field: bool | None = None
    pred_score: float | None = None
    vision_gate: float | None = None
    notes: list[str] = field(default_factory=list)
    record: dict[str, Any] = field(default_factory=dict)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL {path} line {line_number}: {exc}") from exc
    return rows


def extract_epoch_from_name(name: str) -> int:
    match = re.search(r"epoch[_-]?(\d+)", name)
    if not match:
        raise ValueError(f"Could not extract epoch from {name}")
    return int(match.group(1))


def iter_named_values(payload: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, value
            yield from iter_named_values(value, path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]"
            yield path, value
            yield from iter_named_values(value, path)


def extract_candidate_value(
    record: Mapping[str, Any],
    candidates: Sequence[str],
) -> tuple[str | None, Any]:
    for candidate in candidates:
        if candidate in record and record[candidate] not in (None, ""):
            return candidate, record[candidate]

    flattened = list(iter_named_values(record))
    for candidate in candidates:
        for path, value in flattened:
            leaf = path.split(".")[-1]
            leaf = re.sub(r"\[\d+\]$", "", leaf)
            if leaf == candidate and value not in (None, ""):
                return path, value
    return None, None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def normalize_label_with_reason(
    raw_value: Any,
    allowed_labels: Sequence[str] | None = None,
) -> tuple[str | None, str]:
    if raw_value is None:
        return None, "missing"

    allowed = set(allowed_labels or CANONICAL_DIRECTION_LABELS)

    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        raw_text = str(raw_value)
    elif isinstance(raw_value, str):
        raw_text = raw_value
    else:
        raw_text = json.dumps(raw_value, ensure_ascii=False)

    text = raw_text.strip()
    if not text:
        return None, "empty"

    token_key = re.sub(r"[^A-Z]+", "_", text.upper()).strip("_")
    if token_key in TOKEN_TO_LABEL and TOKEN_TO_LABEL[token_key] in allowed:
        return TOKEN_TO_LABEL[token_key], "token"

    normalized_exact = re.sub(r"[\s_]+", " ", text.lower())
    normalized_exact = re.sub(r"\s+", " ", normalized_exact).strip(" .,:;!?\"'()[]{}")
    if normalized_exact in ALIAS_TO_LABEL and ALIAS_TO_LABEL[normalized_exact] in allowed:
        return ALIAS_TO_LABEL[normalized_exact], "alias_exact"

    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return None, "no_alpha_tokens"

    matches: list[tuple[int, str]] = []
    index = 0
    while index < len(words):
        if index + 1 < len(words):
            bigram = f"{words[index]} {words[index + 1]}"
            label = BIGRAM_TO_LABEL.get(bigram)
            if label in allowed:
                matches.append((index, label))
                index += 2
                continue
        unigram = UNIGRAM_TO_LABEL.get(words[index])
        if unigram in allowed:
            matches.append((index, unigram))
        index += 1

    if not matches:
        compact = "".join(words)
        label = ALIAS_TO_LABEL.get(compact)
        if label in allowed:
            return label, "alias_compact"
        return None, "no_label_match"

    distinct_labels = list(dict.fromkeys(label for _, label in matches))
    if len(distinct_labels) == 1:
        return distinct_labels[0], "text_single"

    lowered = text.lower()
    has_answer_cue = bool(
        re.search(
            r"(answer|prediction|predicted|direction|speaker direction|label)\s*(is|:|=)",
            lowered,
        )
    )
    has_option_list = "choose one of" in lowered or "options" in lowered
    if has_option_list and len(distinct_labels) > 2 and not has_answer_cue:
        return None, "ambiguous_option_list"

    if len(words) > 20 and len(distinct_labels) > 3 and not has_answer_cue:
        return None, "ambiguous_many_labels"

    return matches[-1][1], "text_last_match"


def normalize_label(
    raw_value: Any,
    allowed_labels: Sequence[str] | None = None,
) -> str | None:
    label, _ = normalize_label_with_reason(raw_value, allowed_labels=allowed_labels)
    return label


def parse_decode_record(
    record: Mapping[str, Any],
    allowed_labels: Sequence[str] | None = None,
) -> ParsedRecord:
    label_space = allowed_labels or CANONICAL_DIRECTION_LABELS

    gt_source, gt_raw = extract_candidate_value(record, TRUE_LABEL_KEYS)
    ground_truth, gt_reason = normalize_label_with_reason(gt_raw, allowed_labels=label_space)
    if ground_truth is None:
        gt_token_source, gt_token_raw = extract_candidate_value(record, TRUE_TOKEN_KEYS)
        gt_from_token, token_reason = normalize_label_with_reason(
            gt_token_raw,
            allowed_labels=label_space,
        )
        if gt_from_token is not None:
            ground_truth = gt_from_token
            gt_source = gt_token_source or gt_source
            gt_raw = gt_token_raw
            gt_reason = f"token_fallback:{token_reason}"

    pred_source, pred_raw = extract_candidate_value(record, PRED_LABEL_KEYS)
    prediction, pred_reason = normalize_label_with_reason(pred_raw, allowed_labels=label_space)
    if prediction is None:
        pred_token_source, pred_token_raw = extract_candidate_value(record, PRED_TOKEN_KEYS)
        pred_from_token, token_reason = normalize_label_with_reason(
            pred_token_raw,
            allowed_labels=label_space,
        )
        if pred_from_token is not None:
            prediction = pred_from_token
            pred_source = pred_token_source or pred_source
            pred_raw = pred_token_raw
            pred_reason = f"token_fallback:{token_reason}"

    sample_id_source, sample_id = extract_candidate_value(record, SAMPLE_ID_KEYS)
    audio_source, audio_path = extract_candidate_value(record, AUDIO_PATH_KEYS)
    prompt_source, question = extract_candidate_value(record, PROMPT_KEYS)
    correct_source, correct_raw = extract_candidate_value(record, CORRECT_KEYS)
    score_source, pred_score_raw = extract_candidate_value(record, PRED_SCORE_KEYS)
    gate_source, vision_gate_raw = extract_candidate_value(record, VISION_GATE_KEYS)

    notes: list[str] = [
        f"ground_truth_parse={gt_reason}",
        f"prediction_parse={pred_reason}",
    ]
    if gt_source is None:
        notes.append("ground_truth_source=missing")
    if pred_source is None:
        notes.append("prediction_source=missing")
    if sample_id_source is None:
        notes.append("sample_id_source=missing")
    if prompt_source is None:
        notes.append("prompt_source=missing")
    if audio_source is None:
        notes.append("audio_path_source=missing")
    if correct_source is None:
        notes.append("correct_source=missing")
    if score_source is None:
        notes.append("pred_score_source=missing")
    if gate_source is None:
        notes.append("vision_gate_source=missing")

    return ParsedRecord(
        sample_id=str(sample_id) if sample_id is not None else "unknown",
        audio_path=str(audio_path) if audio_path is not None else None,
        question=str(question) if question is not None else None,
        ground_truth=ground_truth,
        prediction=prediction,
        ground_truth_raw=gt_raw,
        prediction_raw=pred_raw,
        ground_truth_source=gt_source,
        prediction_source=pred_source,
        correct_field=_coerce_bool(correct_raw),
        pred_score=_coerce_float(pred_score_raw),
        vision_gate=_coerce_float(vision_gate_raw),
        notes=notes,
        record=dict(record),
    )


def guess_label_order(labels: Iterable[str]) -> list[str]:
    label_set = {label for label in labels if label}
    ordered = [label for label in CANONICAL_DIRECTION_LABELS if label in label_set]
    extras = sorted(label_set - set(ordered))
    return ordered + extras


def compute_decode_metrics(
    parsed_records: Sequence[ParsedRecord],
    label_order: Sequence[str],
) -> dict[str, Any]:
    valid_records = [
        record
        for record in parsed_records
        if record.ground_truth in label_order and record.prediction in label_order
    ]
    y_true = [record.ground_truth for record in valid_records]
    y_pred = [record.prediction for record in valid_records]

    if not y_true or not y_pred:
        raise ValueError("No valid parsed records available for metric computation.")

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0))
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=label_order, average="weighted", zero_division=0)
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_order,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_order)
    gt_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    per_class_rows: list[dict[str, Any]] = []
    for index, label in enumerate(label_order):
        per_class_rows.append(
            {
                "label": label,
                "support": int(support[index]),
                "ground_truth_count": int(gt_counts.get(label, 0)),
                "prediction_count": int(pred_counts.get(label, 0)),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
        )

    parse_fail_true = sum(record.ground_truth is None for record in parsed_records)
    parse_fail_pred = sum(record.prediction is None for record in parsed_records)
    parse_fail_any = sum(
        record.ground_truth is None or record.prediction is None for record in parsed_records
    )
    correct_field_values = [record.correct_field for record in valid_records if record.correct_field is not None]
    correct_field_accuracy = (
        float(sum(correct_field_values) / len(correct_field_values)) if correct_field_values else None
    )
    correct_field_mismatch_count = sum(
        record.correct_field is not None
        and record.correct_field != (record.ground_truth == record.prediction)
        for record in valid_records
    )

    vision_gate_values = [record.vision_gate for record in parsed_records if record.vision_gate is not None]
    prediction_distribution = {
        label: int(pred_counts.get(label, 0))
        for label in label_order
    }
    ground_truth_distribution = {
        label: int(gt_counts.get(label, 0))
        for label in label_order
    }
    top_confusions = Counter(
        (record.ground_truth, record.prediction)
        for record in valid_records
        if record.ground_truth != record.prediction
    ).most_common()

    return {
        "num_records": len(parsed_records),
        "valid_records": len(valid_records),
        "parse_fail_true": parse_fail_true,
        "parse_fail_pred": parse_fail_pred,
        "parse_fail_any": parse_fail_any,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "correct_field_accuracy": correct_field_accuracy,
        "correct_field_mismatch_count": int(correct_field_mismatch_count),
        "per_class_rows": per_class_rows,
        "confusion_matrix": matrix,
        "prediction_distribution": prediction_distribution,
        "ground_truth_distribution": ground_truth_distribution,
        "top_confusions": top_confusions,
        "y_true": y_true,
        "y_pred": y_pred,
        "vision_gate_mean_decode": (
            float(sum(vision_gate_values) / len(vision_gate_values)) if vision_gate_values else None
        ),
    }


def build_top_confusion_rows(
    run_name: str,
    epoch: int,
    top_confusions: Sequence[tuple[tuple[str, str], int]],
    total_errors: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, ((ground_truth, prediction), count) in enumerate(top_confusions, start=1):
        rows.append(
            {
                "run_name": run_name,
                "epoch": epoch,
                "rank": rank,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "count": int(count),
                "error_share": float(count / total_errors) if total_errors else 0.0,
            }
        )
    return rows


def identify_hardest_classes(
    per_class_rows: Sequence[dict[str, Any]],
    metric: str = "recall",
    top_k: int = 3,
) -> list[tuple[str, float]]:
    ordered = sorted(
        ((row["label"], float(row[metric])) for row in per_class_rows),
        key=lambda item: (item[1], item[0]),
    )
    return ordered[:top_k]


def detect_prediction_collapse(
    prediction_distribution: Mapping[str, int],
    label_order: Sequence[str],
    threshold: float = 0.35,
) -> list[str]:
    total = sum(prediction_distribution.values())
    if not total:
        return ["no_predictions"]

    collapse_signals: list[str] = []
    top_label, top_count = max(prediction_distribution.items(), key=lambda item: item[1])
    top_share = top_count / total
    if top_share >= threshold:
        collapse_signals.append(f"top_label={top_label} share={top_share:.1%}")

    zero_pred_labels = [label for label in label_order if prediction_distribution.get(label, 0) == 0]
    if zero_pred_labels:
        collapse_signals.append("zero_pred_labels=" + ", ".join(zero_pred_labels))

    active_labels = sum(count > 0 for count in prediction_distribution.values())
    if active_labels <= max(3, math.ceil(len(label_order) / 2)):
        collapse_signals.append(f"active_labels={active_labels}/{len(label_order)}")

    return collapse_signals


def plot_metric_trajectory(
    summary_df: pd.DataFrame,
    metric_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    for run_name, run_df in summary_df.groupby("run_name", sort=False):
        ordered = run_df.sort_values("epoch")
        ax.plot(
            ordered["epoch"],
            ordered[metric_column],
            marker="o",
            linewidth=2.0,
            label=run_name,
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(summary_df["epoch"].unique()))
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_classwise_trajectory(
    per_class_df: pd.DataFrame,
    run_name: str,
    metric_column: str,
    output_path: Path,
    title: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))

    run_df = per_class_df[per_class_df["run_name"] == run_name].copy()
    for label in CANONICAL_DIRECTION_LABELS:
        label_df = run_df[run_df["label"] == label].sort_values("epoch")
        if label_df.empty:
            continue
        ax.plot(
            label_df["epoch"],
            label_df[metric_column],
            marker="o",
            linewidth=1.8,
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_column.replace("_", " ").title())
    ax.set_xticks(sorted(run_df["epoch"].unique()))
    ax.set_ylim(0.0, 1.05)
    ax.legend(ncol=2, fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    output_path: Path,
    title: str,
    normalize: bool = False,
) -> None:
    plt.style.use("default")
    matrix_to_plot = matrix.astype(float)
    annotation_format = "d"
    if normalize:
        row_sums = matrix_to_plot.sum(axis=1, keepdims=True)
        matrix_to_plot = np.divide(
            matrix_to_plot,
            row_sums,
            out=np.zeros_like(matrix_to_plot),
            where=row_sums != 0,
        )
        annotation_format = ".2f"

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    image = ax.imshow(matrix_to_plot, cmap="YlOrRd", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Ground-truth label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    threshold = matrix_to_plot.max() / 2 if matrix_to_plot.size else 0.0
    for row in range(matrix_to_plot.shape[0]):
        for col in range(matrix_to_plot.shape[1]):
            value = matrix_to_plot[row, col]
            if normalize:
                text = format(value, annotation_format)
            else:
                text = str(int(value))
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_distribution(
    label_order: Sequence[str],
    ground_truth_distribution: Mapping[str, int],
    prediction_distribution: Mapping[str, int],
    output_path: Path,
    title: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    positions = np.arange(len(label_order))
    width = 0.38

    gt_values = [ground_truth_distribution.get(label, 0) for label in label_order]
    pred_values = [prediction_distribution.get(label, 0) for label in label_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(positions - width / 2, gt_values, width=width, label="Ground truth")
    ax.bar(positions + width / 2, pred_values, width=width, label="Prediction")
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xticks(positions)
    ax.set_xticklabels(label_order, rotation=45, ha="right")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        text = str(value)
        text = text.replace("|", "\\|")
        return text.replace("\n", " ")

    header_line = "| " + " | ".join(clean(value) for value in headers) + " |"
    divider_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider_line, *body_lines])


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value * 100:.{digits}f}%"


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"
