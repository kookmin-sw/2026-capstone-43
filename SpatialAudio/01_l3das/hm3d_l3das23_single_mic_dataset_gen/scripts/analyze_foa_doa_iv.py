#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hm3d_l3das23_single_mic.foa_doa_iv_analysis import AnalysisOptions, run_analysis


def _parse_sample_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    sample_ids = {token.strip() for token in value.split(",") if token.strip()}
    return sample_ids or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze FOA DOA/IV stability from generated HM3D samples.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root that contains manifests/ and scenes/.")
    parser.add_argument("--config", type=Path, required=True, help="Generator config used to discover scenes for sanity rerendering.")
    parser.add_argument("--mode", choices=("existing", "sanity", "both"), default="both")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit-los", type=int, default=32)
    parser.add_argument("--limit-nlos", type=int, default=16)
    parser.add_argument("--sample-ids", type=str, default=None, help="Optional comma-separated sample ids.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stft-win", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--energy-db-below-peak", type=float, default=20.0)
    parser.add_argument("--diffuseness-max", type=float, default=0.5)
    parser.add_argument("--beam-az-step", type=float, default=2.0)
    parser.add_argument("--beam-el-step", type=float, default=5.0)
    parser.add_argument("--probe-signals", type=str, default="white,pink,chirp")
    parser.add_argument("--save-rendered-probes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    options = AnalysisOptions(
        dataset_root=args.dataset_root.resolve(),
        config_path=args.config.resolve(),
        mode=str(args.mode),
        split=str(args.split),
        limit_los=int(args.limit_los),
        limit_nlos=int(args.limit_nlos),
        sample_ids=_parse_sample_ids(args.sample_ids),
        out_dir=args.out_dir.resolve(),
        stft_win=int(args.stft_win),
        hop=int(args.hop),
        nfft=int(args.nfft),
        energy_db_below_peak=float(args.energy_db_below_peak),
        diffuseness_max=float(args.diffuseness_max),
        beam_az_step=float(args.beam_az_step),
        beam_el_step=float(args.beam_el_step),
        probe_signals=tuple(token.strip() for token in str(args.probe_signals).split(",") if token.strip()),
        save_rendered_probes=bool(args.save_rendered_probes),
    )
    summary = run_analysis(options)
    print(f"selected_mapping: {summary.get('selected_mapping', '-')}")
    print(f"mapping_selection_source: {summary.get('mapping_selection_source', '-')}")
    print(f"num_items_analyzed: {summary.get('num_items_analyzed', 0)}")
    print(f"aggregate_summary_json: {options.out_dir / 'aggregate_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
