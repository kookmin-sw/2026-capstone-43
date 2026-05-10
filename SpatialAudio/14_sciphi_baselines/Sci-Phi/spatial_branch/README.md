# Spatial Branch (SELDNet) for Sci-Phi

This directory isolates SELD spatial-encoder code from `phi4mm_clean` so you can extend the model without editing vendor files.

## Included modules

- `utils/input.py`
  - `FOASpatialFeatureExtractor`: builds FOA spatial maps (`4 mel + 3 intensity-vector`).
  - `feature_maps_to_seld_tensor`: converts `[frames, 7, mel]` to `[1, 7, frames, mel]`.
- `models/spatial_encoder.py`
  - `SeldNetSpatialEncoder`: SELD backbone (`Conv + BiGRU + MHSA`) returning `[B, T, D]`.
  - `load_from_dcase_checkpoint(...)`: loads matching backbone weights from DCASE checkpoints.
- `projector.py`
  - `SpatialFusionProjector`: paper-style branch fusion module.
    - frozen mono-audio projector path (existing Phi projector)
    - trainable spatial projector path (2-layer linear to 3072)
    - additive fusion in LLM embedding space
- `model.py`
  - `attach_spatial_branch(...)`: runtime injection into loaded Phi model.
  - `set_spatial_features(...)`: pass spatial tensor for the next forward/generation call.
  - `setup_spatial_lora(...)`: add rank-320 spatial LoRA using mono-audio LoRA config.

## Quick usage

```python
import soundfile as sf
from spatial_branch import (
    FOASpatialFeatureExtractor,
    SeldNetSpatialEncoder,
    attach_spatial_branch,
    feature_maps_to_seld_tensor,
    set_spatial_features,
)

# 1) Build and load spatial encoder
spatial_encoder = SeldNetSpatialEncoder()
spatial_encoder.load_from_dcase_checkpoint("../DCASE2024_seld_baseline/3_1_dev_split0_multiaccdoa_foa_model.h5")

# 2) Attach to already loaded phi model (no phi4mm_clean edit)
attach_spatial_branch(model, spatial_encoder, freeze_phi=True, freeze_conformer=True, train_spatial=True)

# 3) Prepare FOA spatial features
audio_4ch, sr = sf.read("your_foa.wav")  # [samples, 4]
extractor = FOASpatialFeatureExtractor(sample_rate=24000, nb_mel_bins=64)
maps = extractor.extract(audio_4ch, sr)  # [frames, 7, mel]
spatial_tensor = feature_maps_to_seld_tensor(maps, device=model.device)

# 4) Inject and run generation
set_spatial_features(model, spatial_tensor)
out = model.generate(**inputs)
```

Optional spatial LoRA (rank 320):

```python
from spatial_branch import setup_spatial_lora
setup_spatial_lora(model, rank=320, adapter_name="spatial", freeze_mono_audio_lora=True)
# by default keeps mono `speech` + new `spatial` adapters both active
```

## Test Convolution (mono source x FOA RIR)

```bash
python spatial_branch/tools/convolve_mono_with_foa_rir.py \
  --dry_wav ./phi4mm_clean/examples/what_is_shown_in_this_image.wav \
  --rir_wav ../rir_IDX0_T21_05_23.wav \
  --out_foa_wav ./spatial_branch/test_data/convolved_foa.wav \
  --out_mono_wav ./spatial_branch/test_data/convolved_mono.wav \
  --target_sr 24000
```

Then run inference with spatial branch using the convolved FOA file:

```bash
FORCE_CPU=1 USE_SPATIAL_BRANCH=1 \
AUDIO_PATH=./spatial_branch/test_data/convolved_foa.wav \
SPATIAL_AUDIO_PATH=./spatial_branch/test_data/convolved_foa.wav \
USE_SPATIAL_LORA=1 SPATIAL_LORA_RANK=320 \
MAX_NEW_TOKENS=64 \
python inference_test.py
```

## Training (Spatial Modules Only)

`train_spatial.py` trains only:
- spatial encoder (SELDNet)
- spatial projector
- spatial LoRA (default rank=320)

Frozen by default:
- Phi backbone
- conformer audio encoder
- mono-audio projector
- mono speech LoRA

### Train JSONL format

```json
{"question":"Where is the sound source?","answer":"A source is on the left side.","audio_path":"./path/input.wav","spatial_audio_path":"./path/input_foa.wav"}
```

- `audio_path`: waveform for conformer path (mono is recommended; multichannel is averaged to mono)
- `spatial_audio_path`: FOA waveform for spatial path (4ch expected)

If `answer` is omitted and `sources` list exists, you can pass `--sort_sources_by_loudness` to serialize sources by descending loudness.

### Train command

```bash
cd /home/yu/Project_git/Sci-Phi
python train_spatial.py \
  --train_jsonl ./your_train.jsonl \
  --output_dir ./spatial_branch/checkpoints \
  --model_dir ./phi4mm_clean \
  --adapter_dir ./phi4mm_clean/speech-lora \
  --spatial_ckpt ../DCASE2024_seld_baseline/3_1_dev_split0_multiaccdoa_foa_model.h5 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --num_epochs 5 \
  --global_batch_size 24 \
  --per_device_batch_size 1 \
  --spatial_lora_rank 320 \
  --dtype bfloat16
```

For multi-GPU:

```bash
torchrun --nproc_per_node=8 train_spatial.py \
  --train_jsonl ./your_train.jsonl \
  --output_dir ./spatial_branch/checkpoints \
  --global_batch_size 24 \
  --per_device_batch_size 1
```

## Notes

- The spatial branch is consumed only when `set_spatial_features(...)` is called before forward/generation.
- Conformer encoder and mono-audio projector are frozen by default.
- Spatial encoder and spatial projector are trainable.
