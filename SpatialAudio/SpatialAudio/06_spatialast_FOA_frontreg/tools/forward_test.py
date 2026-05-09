import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spatial_ast


def reorder_raw_foa_wyzx_to_wxyz(x):
    """Raw FOA order is WYZX; internally reordered to WXYZ."""
    if x.ndim == 3:
        assert x.shape[1] == 4, f"Expected [B, 4, T], got {tuple(x.shape)}"
        return x[:, [0, 3, 1, 2], :]
    if x.ndim == 2:
        assert x.shape[0] == 4, f"Expected [4, T], got {tuple(x.shape)}"
        return x[[0, 3, 1, 2], :]
    raise ValueError(f"Unsupported tensor rank for FOA reorder: {x.ndim}")


def build_synthetic_raw_foa(batch_size, sample_rate, seconds):
    total_samples = sample_rate * seconds
    t = torch.linspace(0.0, 1.0, total_samples)

    # Synthetic raw WYZX channels with deliberately different patterns.
    w = 0.1 * torch.sin(2 * torch.pi * 220.0 * t)
    y = 0.2 * torch.sin(2 * torch.pi * 330.0 * t + 0.1)
    z = 0.3 * torch.sin(2 * torch.pi * 440.0 * t + 0.2)
    x = 0.4 * torch.sin(2 * torch.pi * 550.0 * t + 0.3)

    raw = torch.stack([w, y, z, x], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--seconds", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    raw_wyzx = build_synthetic_raw_foa(args.batch_size, args.sample_rate, args.seconds)
    canonical_wxyz = reorder_raw_foa_wyzx_to_wxyz(raw_wyzx)

    print(f"raw_wyzx shape: {tuple(raw_wyzx.shape)}")
    print("raw channel order: WYZX")
    print(f"canonical_wxyz shape: {tuple(canonical_wxyz.shape)}")
    print("internal channel order: WXYZ")

    model = spatial_ast.build_AST(
        num_classes=args.num_classes,
        reverb_type="foa",
        foa_stem_type="foa_native",
    ).to(device)
    model.eval()

    waveforms = canonical_wxyz.to(device)
    reverbs = torch.zeros(args.batch_size, 1, 1, device=device)

    with torch.no_grad():
        outputs = model(waveforms, reverbs)

    assert len(outputs) == 5, f"Expected 5 outputs, got {len(outputs)}"
    debug_shapes = model.get_debug_shapes()

    print(f"classifier shape: {tuple(outputs[0].shape)}")
    print(f"distance shape: {tuple(outputs[1].shape)}")
    print(f"azimuth shape: {tuple(outputs[2].shape)}")
    print(f"elevation shape: {tuple(outputs[3].shape)}")
    print(f"vector shape: {tuple(outputs[4].shape)}")
    print(f"debug shapes: {debug_shapes}")
    print("forward_test: PASS")


if __name__ == "__main__":
    main()
