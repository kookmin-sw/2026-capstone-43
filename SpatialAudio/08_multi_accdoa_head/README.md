# Multi-ACCDOA Source-Slot Head Experiment

This directory is an isolated experiment for a Multi-ACCDOA source-slot head.
It does not modify or depend on the existing SpatialAST training code.

The goal is to make the head/loss interface easy to attach later to a real FOA
encoder or SpatialAST encoder. Before that integration, the modules can be
tested with random slot tokens or the included toy model.

## Representation

`Kmax = 3` source slots by default. One slot is one source hypothesis.

Each slot predicts:

- `accdoa`: `[3]`
- `class_logits`: `[C]`
- `distance`: `[1]`

Output shapes:

```python
{
    "accdoa":        [B, Kmax, 3],
    "class_logits":  [B, Kmax, C],
    "distance":      [B, Kmax, 1],
}
```

The implementation uses a joint head:

```python
slot_tokens = encoder_output_slots  # [B, Kmax, D]
joint_out = joint_head(slot_tokens)  # [B, Kmax, 3 + C + 1]
```

The split is:

```python
accdoa       = joint_out[..., 0:3]
class_logits = joint_out[..., 3:3+C]
distance     = joint_out[..., 3+C:3+C+1]
```

## Design Rules

- `slot = one source hypothesis`.
- A slot jointly owns `accdoa + class + distance`.
- Class and direction are coupled by the same slot index.
- `||accdoa||` is activity/presence.
- Distance is not encoded into the ACCDOA norm.
- Inactive slot target is `accdoa = [0, 0, 0]`.
- Class/distance losses are computed only for active matched slots.
- Inactive class uses `ignore_index = -100`.
- Multi-source order is permutation-invariant, so PIT matching is required.
- Because `Kmax` is small, this experiment uses exhaustive `itertools.permutations`
  instead of SciPy/Hungarian matching.

## Distance Target Policy

Predicted `distance` is interpreted as log-distance by default.

- `distance_target_is_log=False`: target distance is raw meters, and PIT/loss
  convert it internally with `log(distance + eps)`.
- `distance_target_is_log=True`: target distance is already log-distance, so no
  additional log transform is applied.
- `matched["distance"]` stores the distance target in the same space used by the
  loss. In the default `distance_is_log=True` mode, this means log-distance.

This shared policy is implemented through `prepare_distance_target(...)` so PIT
matching and final distance loss do not accidentally apply `log()` twice.

## ACCDOA Active/Inactive Loss

Active slots are trained toward unit direction vectors. Inactive slots are
trained toward `[0, 0, 0]`, because ACCDOA norm is the activity/presence score.

`MultiACCDOALoss` exposes:

- `lambda_acc_active`: vector loss weight for active matched slots.
- `lambda_acc_inactive`: zero-vector penalty weight for inactive slots.
- `lambda_acc_vec`: outer weight applied after combining active/inactive vector
  losses.

Class and distance losses are still computed only for active slots.

## Validation Metrics

`threshold_sweep_metrics(...)` evaluates active/inactive decisions by sweeping
the ACCDOA norm threshold. The default threshold commonly used for decoding is
`0.5`, but validation should sweep thresholds when activity calibration matters.

For fixed source count experiments, `topk_matched_metrics(...)` selects the top-K
slots by ACCDOA norm. This helps separate direction quality from activity
threshold calibration. For example, in a 2-source-only experiment, use `k=2` to
check whether the two strongest slots point in the right directions even when
their norms are under the threshold.

## Files

- `src/heads.py`: `JointMultiSourceHead` and `decode_accdoa`.
- `src/pit.py`: exhaustive PIT matching and matched target construction.
- `src/losses.py`: `MultiACCDOALoss`.
- `src/metrics.py`: angular, distance, class, and activity metrics.
- `src/synthetic_data.py`: random source target generation for tests.
- `src/toy_model.py`: small MLP + learnable slot query model.
- `tests/`: fast pytest coverage for shapes, PIT assignment, and backward.
- `scripts/run_pit_demo.py`: human-readable swapped-slot PIT demo.
- `scripts/run_sanity_train.py`: toy overfit sanity training loop.

## Install

```bash
cd /home/yu/Project_git/SpatialAudio/08_multi_accdoa_head
pip install -r requirements.txt
```

## Test

```bash
cd /home/yu/Project_git/SpatialAudio/08_multi_accdoa_head
python -m pytest -q
```

## PIT Demo

```bash
cd /home/yu/Project_git/SpatialAudio/08_multi_accdoa_head
python scripts/run_pit_demo.py
```

Expected logical case:

- GT source A = front
- GT source B = left
- pred slot0 = left
- pred slot1 = inactive
- pred slot2 = front

Expected PIT assignment:

```text
slot0 -> source B
slot2 -> source A
slot1 -> inactive
```

## Sanity Training

```bash
cd /home/yu/Project_git/SpatialAudio/08_multi_accdoa_head
python scripts/run_sanity_train.py
```

This script overfits a fixed synthetic batch. It is only intended to verify that:

- PIT matching is differentiable around the selected assignment.
- ACCDOA inactive slots are pulled toward zero.
- Class/distance losses operate on active slots.
- The whole head/loss pipeline supports `loss.backward()`.

## Future Encoder Integration

The only required encoder interface is:

```python
slot_tokens = encoder(waveform)  # [B, Kmax, D]
pred = JointMultiSourceHead(hidden_dim=D, num_classes=C, kmax=Kmax)(slot_tokens)
loss_out = MultiACCDOALoss(num_classes=C)(pred, targets)
loss = loss_out["loss"]
```

Threshold sweep example:

```python
from src.metrics import threshold_sweep_metrics

metrics = threshold_sweep_metrics(
    pred,
    targets,
    thresholds=[0.3, 0.5, 0.7],
    kmax=Kmax,
)
```

Top-K fixed-source example:

```python
from src.metrics import topk_matched_metrics

metrics = topk_matched_metrics(pred, targets, k=2)
```

Targets stay a list because each sample can have a different number of sources:

```python
targets = [
    {
        "accdoa": Tensor[N_i, 3],     # unit directions
        "class": Tensor[N_i],         # class ids
        "distance": Tensor[N_i, 1],   # raw meters by default
    },
    ...
]
```

By default, predicted `distance` is interpreted as log-distance. Raw meter
targets are transformed internally with `log(distance + eps)`.

## GitHub Cleanup Notes

This GitHub-ready copy was renumbered from `17_multi_accdoa_head` to `08_multi_accdoa_head`.
Generated experiment artifacts, logs, Python caches, and model weights were removed from this copy. The useful result metadata is summarized below so the repository stays lightweight.

### Removed Artifacts
- `src/__pycache__/`: 5 files, 27.0 KB

### Result Summary
- No experiment result JSON or model weight files were present in the selected original folder; only generated Python caches were excluded.
