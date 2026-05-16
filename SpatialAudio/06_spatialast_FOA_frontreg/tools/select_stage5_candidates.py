import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STAGE4_ROOT = ROOT / "outputs_stage4"


def load_stage4_best(run_name):
    metrics_path = STAGE4_ROOT / run_name / "metrics_summary.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return {
        "run_name": run_name,
        "config": metrics.get("config", {}),
        "best": metrics.get("best", {}),
    }


def angular_delta(run_a, run_b):
    if run_a is None or run_b is None:
        return None
    return run_a["best"]["val_angular_error"] - run_b["best"]["val_angular_error"]


def build_selected_candidates():
    foa_last4 = load_stage4_best("foa_last4")
    logmel_last4 = load_stage4_best("logmel_last4")
    foa_stage3_last2 = load_stage4_best("foa_stage3_last2")
    foa_patch_last4 = load_stage4_best("foa_patch_last4")
    logmel_patch_last4 = load_stage4_best("logmel_patch_last4")
    foa_last4_cosine = load_stage4_best("foa_last4_cosine")
    foa_last4_longer = load_stage4_best("foa_last4_longer")
    foa_last4_lower_head_lr = load_stage4_best("foa_last4_lower_head_lr")

    longer_gain = angular_delta(foa_last4_longer, foa_last4)
    lower_head_gain = angular_delta(foa_last4_lower_head_lr, foa_last4)
    cosine_gain = angular_delta(foa_last4_cosine, foa_last4)
    patch_last4_penalty = angular_delta(foa_patch_last4, foa_last4)
    logmel_patch_penalty = angular_delta(logmel_patch_last4, logmel_last4)
    last2_vs_last4 = angular_delta(foa_stage3_last2, foa_last4)

    return [
        {
            "priority": 1,
            "run_name": "logmel_last4",
            "script_name": "train_stage5_2400_logmel_last4.sh",
            "reason": "Anchor baseline for logmel-only comparisons.",
        },
        {
            "priority": 2,
            "run_name": "foa_last4",
            "script_name": "train_stage5_2400_foa_last4.sh",
            "reason": "Anchor baseline for FOA-native recipe and unfreeze comparisons.",
        },
        {
            "priority": 3,
            "run_name": "foa_last6",
            "script_name": "train_stage5_2400_foa_last6.sh",
            "reason": "Highest-value new unfreeze probe above last4 without changing architecture.",
        },
        {
            "priority": 4,
            "run_name": "foa_patch_last6",
            "script_name": "train_stage5_2400_foa_patch_last6.sh",
            "reason": (
                "Patch-plus-last4 already underperformed by "
                f"{patch_last4_penalty:+.4f} angular error, so keep only one higher-adaptation patch probe."
            ),
        },
        {
            "priority": 5,
            "run_name": "foa_last4_longer",
            "script_name": "train_stage5_2400_foa_last4_longer.sh",
            "reason": (
                "Strongest Stage-4 single-seed recipe gain: "
                f"{longer_gain:+.4f} angular error vs foa_last4."
            ),
        },
        {
            "priority": 6,
            "run_name": "foa_last4_lower_head_lr",
            "script_name": "train_stage5_2400_foa_last4_lower_head_lr.sh",
            "reason": (
                "Second-best Stage-4 recipe gain: "
                f"{lower_head_gain:+.4f} angular error vs foa_last4."
            ),
        },
        {
            "priority": 7,
            "run_name": "foa_last4_lower_stem_lr",
            "script_name": "train_stage5_2400_foa_last4_lower_stem_lr.sh",
            "reason": "New targeted LR-group probe to test whether the FOA stem is over-updating.",
        },
        {
            "priority": 8,
            "run_name": "logmel_last6",
            "script_name": "train_stage5_2400_logmel_last6.sh",
            "reason": "Fair comparison run for larger-unfreeze effects on the logmel baseline.",
        },
    ], [
        {
            "run_name": "foa_last2",
            "reason": (
                "Stage-4 already has an equivalent lower-bound reference in foa_stage3_last2 "
                f"({last2_vs_last4:+.4f} angular vs foa_last4), so Stage-5 compute is better spent on last6."
            ),
        },
        {
            "run_name": "foa_patch_last4",
            "reason": (
                "Down-ranked because patch_plus_last4 already missed foa_last4 by "
                f"{patch_last4_penalty:+.4f} angular error."
            ),
        },
        {
            "run_name": "foa_last4_cosine",
            "reason": (
                "Already tested in Stage-4; it improved by "
                f"{cosine_gain:+.4f} angular error, but weaker than longer/lower_head and less informative than lower_stem."
            ),
        },
        {
            "run_name": "logmel_patch_last4",
            "reason": (
                "Down-ranked because patch_plus_last4 already missed logmel_last4 by "
                f"{logmel_patch_penalty:+.4f} angular error."
            ),
        },
        {
            "run_name": "logmel_last4_longer",
            "reason": "Deferred until a stronger FOA-native candidate emerges that clearly needs a longer logmel control.",
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-names", action="store_true")
    args = parser.parse_args()

    selected, deferred = build_selected_candidates()
    if args.run_names:
        for item in selected:
            print(item["run_name"])
        return

    print("Stage-5 candidate shortlist")
    print()
    for item in selected:
        print(
            f"{item['priority']}. {item['run_name']} "
            f"({item['script_name']}): {item['reason']}"
        )

    print()
    print("Deferred for now")
    for item in deferred:
        print(f"- {item['run_name']}: {item['reason']}")


if __name__ == "__main__":
    main()
