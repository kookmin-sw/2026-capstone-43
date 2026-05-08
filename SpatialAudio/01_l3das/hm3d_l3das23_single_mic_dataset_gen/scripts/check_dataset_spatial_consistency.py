#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hm3d_l3das23_single_mic.geometry import world_to_local  # noqa: E402
from hm3d_l3das23_single_mic.manifest_io import iter_dataset_rows  # noqa: E402
from hm3d_l3das23_single_mic.spatial_conventions import (  # noqa: E402
    AZIMUTH_CONVENTION,
    AZIMUTH_REFERENCE,
    FOA_CANONICAL_AXES,
    FOA_CANONICAL_CHANNEL_ORDER,
    FOA_RAW_CHANNEL_ORDER,
    LOCAL_COORDINATE_FRAME,
    ambix_unit_vector_xyz,
    azimuth_raw_deg,
    direction_8way_from_azimuth,
    local_angles_from_relative_xyz,
    local_unit_vector_right_front_up,
    normalize_azimuth_deg,
)


CONVENTION_FIELDS = {
    "local_coordinate_frame": LOCAL_COORDINATE_FRAME,
    "azimuth_reference": AZIMUTH_REFERENCE,
    "azimuth_convention": AZIMUTH_CONVENTION,
    "audio_channel_order": FOA_RAW_CHANNEL_ORDER,
    "foa_raw_channel_order": FOA_RAW_CHANNEL_ORDER,
    "foa_canonical_channel_order": FOA_CANONICAL_CHANNEL_ORDER,
    "foa_canonical_axes": FOA_CANONICAL_AXES,
}


def _as_float_list(value: Any, *, length: int) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    return values


def _relative_xyz(row: dict[str, Any]) -> Optional[list[float]]:
    for key in ("source_mic_relative_position", "source_pose_local_xyz"):
        values = _as_float_list(row.get(key), length=3)
        if values is not None:
            return values
    sources = row.get("sources")
    if isinstance(sources, list) and sources:
        first = sources[0]
        if isinstance(first, dict):
            return _as_float_list(first.get("source_mic_relative_position"), length=3)
    return None


def _source_world(row: dict[str, Any]) -> Optional[list[float]]:
    for key in ("source_world_position", "source_pose_world", "speaker_proxy_reference_world"):
        values = _as_float_list(row.get(key), length=3)
        if values is not None:
            return values
    return None


def _mic_pose(row: dict[str, Any]) -> tuple[Optional[list[float]], Optional[float]]:
    pose = row.get("mic_pose_world")
    if isinstance(pose, dict):
        position = _as_float_list(pose.get("position_xyz"), length=3)
        yaw = pose.get("yaw_rad")
        if position is not None and isinstance(yaw, (int, float)) and math.isfinite(float(yaw)):
            return position, float(yaw)

    position = _as_float_list(row.get("mic_world_position"), length=3)
    yaw = row.get("mic_yaw_rad")
    if position is not None and isinstance(yaw, (int, float)) and math.isfinite(float(yaw)):
        return position, float(yaw)
    return None, None


def _circular_error_deg(actual: float, expected: float) -> float:
    return abs(normalize_azimuth_deg(float(actual) - float(expected)))


def _check_close(
    errors: list[str],
    *,
    sample_id: str,
    field: str,
    actual: Any,
    expected: float,
    tolerance: float,
    circular: bool = False,
) -> None:
    if actual is None:
        return
    if not isinstance(actual, (int, float)) or not math.isfinite(float(actual)):
        errors.append(f"{sample_id}: {field} is not finite numeric: {actual!r}")
        return
    delta = _circular_error_deg(float(actual), expected) if circular else abs(float(actual) - float(expected))
    if delta > tolerance:
        errors.append(
            f"{sample_id}: {field} mismatch actual={float(actual):.6f} expected={expected:.6f} delta={delta:.6f}"
        )


def _check_vector_close(
    errors: list[str],
    *,
    sample_id: str,
    field: str,
    actual: Any,
    expected: Iterable[float],
    tolerance: float,
) -> None:
    actual_values = _as_float_list(actual, length=3)
    if actual_values is None:
        return
    expected_values = [float(value) for value in expected]
    max_delta = max(abs(a - b) for a, b in zip(actual_values, expected_values))
    if max_delta > tolerance:
        errors.append(
            f"{sample_id}: {field} mismatch actual={actual_values} expected={expected_values} max_delta={max_delta:.6f}"
        )


def _check_row(
    row: dict[str, Any],
    *,
    tolerance: float,
    require_new_fields: bool,
) -> tuple[list[str], list[str], dict[str, Any]]:
    sample_id = str(row.get("sample_id", "<unknown>"))
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    relative = _relative_xyz(row)
    if relative is None:
        return [f"{sample_id}: missing source_mic_relative_position/source_pose_local_xyz"], warnings, stats

    azimuth_deg, elevation_deg, distance_m = local_angles_from_relative_xyz(relative)
    direction = direction_8way_from_azimuth(azimuth_deg)
    stats["direction_8way"] = direction
    modulo_10 = abs(normalize_azimuth_deg(azimuth_deg)) % 10.0
    stats["azimuth_nearest_10deg_residual"] = round(min(modulo_10, 10.0 - modulo_10), 6)

    for field in ("continuous_azimuth_deg", "azimuth_deg"):
        _check_close(
            errors,
            sample_id=sample_id,
            field=field,
            actual=row.get(field),
            expected=azimuth_deg,
            tolerance=tolerance,
            circular=True,
        )
    _check_close(
        errors,
        sample_id=sample_id,
        field="azimuth_continuous_raw_deg",
        actual=row.get("azimuth_continuous_raw_deg"),
        expected=azimuth_raw_deg(azimuth_deg),
        tolerance=tolerance,
        circular=True,
    )
    for field in ("continuous_elevation_deg", "elevation_deg"):
        _check_close(
            errors,
            sample_id=sample_id,
            field=field,
            actual=row.get(field),
            expected=elevation_deg,
            tolerance=tolerance,
        )
    for field in ("distance_to_mic", "source_distance", "euclidean_distance", "direct_path_length"):
        _check_close(
            errors,
            sample_id=sample_id,
            field=field,
            actual=row.get(field),
            expected=distance_m,
            tolerance=tolerance,
        )

    spherical = row.get("source_pose_spherical")
    if isinstance(spherical, dict):
        _check_close(
            errors,
            sample_id=sample_id,
            field="source_pose_spherical.distance",
            actual=spherical.get("distance"),
            expected=distance_m,
            tolerance=tolerance,
        )
        _check_close(
            errors,
            sample_id=sample_id,
            field="source_pose_spherical.azimuth_deg",
            actual=spherical.get("azimuth_deg"),
            expected=azimuth_deg,
            tolerance=tolerance,
            circular=True,
        )
        _check_close(
            errors,
            sample_id=sample_id,
            field="source_pose_spherical.elevation_deg",
            actual=spherical.get("elevation_deg"),
            expected=elevation_deg,
            tolerance=tolerance,
        )

    for field in ("direction_8way", "label_8way"):
        actual = row.get(field)
        if actual is None:
            if require_new_fields:
                errors.append(f"{sample_id}: missing {field}")
            else:
                warnings.append(f"{sample_id}: missing {field}")
        elif str(actual) != direction:
            errors.append(f"{sample_id}: {field} mismatch actual={actual!r} expected={direction!r}")

    for field, expected in CONVENTION_FIELDS.items():
        actual = row.get(field)
        if actual is None:
            if require_new_fields:
                errors.append(f"{sample_id}: missing {field}")
            else:
                warnings.append(f"{sample_id}: missing {field}")
        elif str(actual) != expected:
            errors.append(f"{sample_id}: {field} mismatch actual={actual!r} expected={expected!r}")

    _check_vector_close(
        errors,
        sample_id=sample_id,
        field="local_unit_vector_right_front_up",
        actual=row.get("local_unit_vector_right_front_up"),
        expected=local_unit_vector_right_front_up(relative),
        tolerance=tolerance,
    )
    _check_vector_close(
        errors,
        sample_id=sample_id,
        field="ambix_unit_vector_xyz",
        actual=row.get("ambix_unit_vector_xyz"),
        expected=ambix_unit_vector_xyz(relative),
        tolerance=tolerance,
    )

    mic_position, mic_yaw_rad = _mic_pose(row)
    source_world = _source_world(row)
    if mic_position is not None and mic_yaw_rad is not None and source_world is not None:
        recomputed_relative = world_to_local(mic_position, mic_yaw_rad, source_world).tolist()
        _check_vector_close(
            errors,
            sample_id=sample_id,
            field="world_to_local(source_world_position)",
            actual=relative,
            expected=recomputed_relative,
            tolerance=max(tolerance, 1.0e-5),
        )

    proxy_reference = _as_float_list(row.get("speaker_proxy_reference_world"), length=3)
    explicit_source_world = _as_float_list(row.get("source_world_position"), length=3)
    if proxy_reference is not None and explicit_source_world is not None:
        _check_vector_close(
            errors,
            sample_id=sample_id,
            field="source_world_position_vs_speaker_proxy_reference_world",
            actual=explicit_source_world,
            expected=proxy_reference,
            tolerance=max(tolerance, 1.0e-5),
        )

    return errors, warnings, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check generated HM3D/L3DAS spatial labels against mic-local coordinates."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument("--require-new-fields", action="store_true")
    parser.add_argument("--max-error-preview", type=int, default=30)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    direction_counts: Counter[str] = Counter()
    azimuth_nearest_10deg_residual_counts: Counter[str] = Counter()
    checked = 0

    for row in iter_dataset_rows(dataset_root):
        if args.limit is not None and checked >= int(args.limit):
            break
        row_errors, row_warnings, row_stats = _check_row(
            row,
            tolerance=float(args.tolerance),
            require_new_fields=bool(args.require_new_fields),
        )
        errors.extend(row_errors)
        warnings.extend(row_warnings)
        direction = row_stats.get("direction_8way")
        if isinstance(direction, str):
            direction_counts[direction] += 1
        residual = row_stats.get("azimuth_nearest_10deg_residual")
        if isinstance(residual, (int, float)):
            bucket = f"{round(float(residual), 3):.3f}"
            azimuth_nearest_10deg_residual_counts[bucket] += 1
        checked += 1

    summary = {
        "dataset_root": str(dataset_root),
        "checked_rows": checked,
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "direction_8way_counts": dict(sorted(direction_counts.items())),
        "azimuth_nearest_10deg_residual_counts": dict(
            sorted(azimuth_nearest_10deg_residual_counts.items())
        ),
        "error_preview": errors[: int(args.max_error_preview)],
        "warning_preview": warnings[: int(args.max_error_preview)],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
