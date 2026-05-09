from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_yaml_config
from src.pipeline import AudioPipelineConfig, run_pipeline


def _config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def build_arg_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FOA wav to learning-ready spherical audio tensor A_sphere.")
    parser.add_argument("--input", required=True, help="Path to a 4-channel FOA wav file or directory of wav files.")
    parser.add_argument("--output_dir", default=_config_value(defaults, "output_dir", "outputs/demo"))
    parser.add_argument("--channel_order", default=_config_value(defaults, "channel_order", "WXYZ"))
    parser.add_argument("--target_sr", type=int, default=_config_value(defaults, "target_sr", None))
    parser.add_argument("--normalize_audio", action="store_true", default=bool(_config_value(defaults, "normalize_audio", False)))
    parser.add_argument("--num_az_bins", type=int, default=int(_config_value(defaults, "num_az_bins", 24)))
    parser.add_argument("--num_el_bins", type=int, default=int(_config_value(defaults, "num_el_bins", 8)))
    parser.add_argument("--window_sec", type=float, default=float(_config_value(defaults, "window_sec", 2.0)))
    parser.add_argument("--hop_sec", type=float, default=float(_config_value(defaults, "hop_sec", 1.0)))
    parser.add_argument("--aggregation", choices=["mean", "max", "both"], default=_config_value(defaults, "aggregation", "both"))
    parser.add_argument("--pooling_mode", choices=["mean", "max", "both"], default=_config_value(defaults, "pooling_mode", "both"))
    parser.add_argument("--pooling_mapping", choices=["sector", "nearest"], default=_config_value(defaults, "pooling_mapping", "sector"))
    parser.add_argument("--n_fft", type=int, default=int(_config_value(defaults, "n_fft", 1024)))
    parser.add_argument("--stft_hop_length", type=int, default=_config_value(defaults, "stft_hop_length", None))
    parser.add_argument("--stft_window", default=_config_value(defaults, "stft_window", "hann"))
    parser.add_argument("--aiv_sign", type=float, default=float(_config_value(defaults, "aiv_sign", 1.0)))
    parser.add_argument("--export_pt", action="store_true", default=bool(_config_value(defaults, "export_pt", False)))
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    known, _ = pre_parser.parse_known_args()
    defaults = read_yaml_config(known.config)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    full_parser = build_arg_parser(defaults)
    for action in full_parser._actions:
        if not any(existing.dest == action.dest for existing in parser._actions):
            parser._add_action(action)
    args = parser.parse_args()

    config = AudioPipelineConfig(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        channel_order=args.channel_order,
        target_sr=args.target_sr,
        normalize_audio=args.normalize_audio,
        num_az_bins=args.num_az_bins,
        num_el_bins=args.num_el_bins,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        aggregation=args.aggregation,
        pooling_mode=args.pooling_mode,
        pooling_mapping=args.pooling_mapping,
        n_fft=args.n_fft,
        stft_hop_length=args.stft_hop_length,
        stft_window=args.stft_window,
        aiv_sign=args.aiv_sign,
        export_pt=args.export_pt,
    )
    result = run_pipeline(config)
    print(f"Output directory: {result['output_dir']}")


if __name__ == "__main__":
    main()

