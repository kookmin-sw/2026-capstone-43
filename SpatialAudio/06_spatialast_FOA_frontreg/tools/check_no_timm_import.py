import builtins
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


_real_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    top_level = name.split(".")[0]
    if top_level == "timm":
        raise RuntimeError("timm import attempted, but 03_spatialast_FOA should not depend on timm.")
    return _real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded_import


def main():
    from model import build_model

    model = build_model(
        num_classes=10,
        reverb_type="foa",
        foa_stem_type="foa_native",
    )
    model.eval()

    waveforms = torch.randn(1, 4, 32000)
    reverbs = torch.zeros(1, 1, 1)

    with torch.no_grad():
        outputs = model(waveforms, reverbs=reverbs)

    print("check_no_timm_import: PASS")
    print({key: tuple(value.shape) for key, value in outputs.items()})


if __name__ == "__main__":
    main()
