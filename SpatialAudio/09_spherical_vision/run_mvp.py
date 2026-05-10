from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PipelineConfig, run_pipeline


def _load_yaml_defaults(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(payload)!r}.")
    return payload


def build_parser(defaults: dict[str, Any], default_config_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Structured RGB -> ZoeDepth -> Point Cloud -> Vision Sphere export pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=default_config_path, help="YAML config file.")
    parser.add_argument("--input", type=Path, required="input" not in defaults, help="Input image path or directory.")
    parser.add_argument("--output_dir", type=Path, required="output_dir" not in defaults, help="Output directory.")
    parser.add_argument("--zoe_model_path", type=Path, default=defaults.get("zoe_model_path"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=defaults.get("device", "auto"))
    parser.add_argument("--fx", type=float, default=defaults.get("fx"))
    parser.add_argument("--fy", type=float, default=defaults.get("fy"))
    parser.add_argument("--cx", type=float, default=defaults.get("cx"))
    parser.add_argument("--cy", type=float, default=defaults.get("cy"))
    parser.add_argument("--hfov_deg", type=float, default=defaults.get("hfov_deg", 69.0))
    parser.add_argument("--num_az_bins", type=int, default=defaults.get("num_az_bins", 24))
    parser.add_argument("--num_el_bins", type=int, default=defaults.get("num_el_bins", 8))
    parser.add_argument(
        "--use_elevation",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("use_elevation", True),
        help="Retained for compatibility. The structured export always uses the full 2D spherical grid.",
    )
    parser.add_argument("--point_stride", type=int, default=defaults.get("point_stride", 1))
    parser.add_argument("--max_points", type=int, default=defaults.get("max_points", 200_000))
    parser.add_argument("--plot_max_points", type=int, default=defaults.get("plot_max_points", 40_000))
    parser.add_argument("--depth_clip_min", type=float, default=defaults.get("depth_clip_min"))
    parser.add_argument("--depth_clip_max", type=float, default=defaults.get("depth_clip_max"))
    parser.add_argument(
        "--depth_clip_percentile_low",
        type=float,
        default=defaults.get("depth_clip_percentile_low"),
        help="Optional lower percentile bound for filtering raw depth before point cloud generation.",
    )
    parser.add_argument(
        "--depth_clip_percentile_high",
        type=float,
        default=defaults.get("depth_clip_percentile_high"),
        help="Optional upper percentile bound for filtering raw depth before point cloud generation.",
    )
    parser.add_argument(
        "--include_extra_depth_channels",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("include_extra_depth_channels", True),
        help="Include inverse/log depth transform channels in the exported tensor.",
    )
    parser.add_argument(
        "--export_pt",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("export_pt", False),
        help="Also export vision_sphere.pt if torch is available.",
    )
    parser.add_argument(
        "--pooling_mode",
        choices=["mean", "max", "both"],
        default=defaults.get("pooling_mode", "mean"),
        help="8-way pooling reduction for azimuth features.",
    )
    parser.add_argument(
        "--pooling_mapping",
        choices=["sector", "nearest"],
        default=defaults.get("pooling_mapping", "sector"),
        help="How azimuth bin centers map to the 8 coarse sectors.",
    )
    parser.add_argument(
        "--save_globe",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("save_globe", False),
        help="Save optional 3D globe visualizations for debugging.",
    )
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 0))
    parser.add_argument("--log_level", default=defaults.get("log_level", "INFO"))
    return parser


def parse_args() -> argparse.Namespace:
    default_config_path = PROJECT_ROOT / "configs" / "default.yaml"
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=default_config_path)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = _load_yaml_defaults(pre_args.config)
    parser = build_parser(defaults, pre_args.config)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig(
        input_path=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        project_root=PROJECT_ROOT,
        zoe_model_path=args.zoe_model_path,
        device=args.device,
        hfov_deg=float(args.hfov_deg),
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        num_az_bins=int(args.num_az_bins),
        num_el_bins=int(args.num_el_bins),
        use_elevation=bool(args.use_elevation),
        point_stride=int(args.point_stride),
        max_points=None if args.max_points is None or int(args.max_points) <= 0 else int(args.max_points),
        plot_max_points=int(args.plot_max_points),
        depth_clip_min=args.depth_clip_min,
        depth_clip_max=args.depth_clip_max,
        depth_clip_percentile_low=args.depth_clip_percentile_low,
        depth_clip_percentile_high=args.depth_clip_percentile_high,
        include_extra_depth_channels=bool(args.include_extra_depth_channels),
        export_pt=bool(args.export_pt),
        pooling_mode=str(args.pooling_mode),
        pooling_mapping=str(args.pooling_mapping),
        save_globe=bool(args.save_globe),
        seed=int(args.seed),
        log_level=str(args.log_level),
    )

    summary = run_pipeline(config)
    print(
        f"Processed {summary['processed_count']}/{summary['num_images']} image(s). "
        f"Outputs saved under: {config.output_dir}"
    )
    if summary["failure_count"]:
        print(f"Failed images: {summary['failure_count']}. See error.json files inside the output directory.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
