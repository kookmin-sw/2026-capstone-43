# LRDSE

Robot noise speech enhancement 실험용 코드입니다. Clean speech와 quadruped robot noise가 섞인 noisy speech를 paired manifest로 학습하며, 기본 학습 경로는 `train_sgmse.py`의 SGMSE 기반 모델입니다. Noisy sample 폴더에 robot `anchor` / `lowstate` 파일이 있으면 foot force 8ch auxiliary condition도 함께 사용할 수 있습니다.

## 폴더 구조

```text
LRDSE/
├── train_sgmse.py              # SGMSE speech enhancement 학습 / 검증 / 샘플 저장
├── denoise_sgmse.py            # 학습된 SGMSE checkpoint로 noisy wav denoising
├── train_rddm.py               # RDDM 실험 코드
├── dataset.py                  # paired clean/noisy dataset, condition 로딩
├── src/
│   ├── audio/preprocess.py     # wav crop, STFT, spec transform, inverse transform
│   ├── check/                  # dataset/model/condition sanity check 유틸
│   ├── condition/preprocess.py # foot_force condition token 생성
│   ├── models/                 # SGMSE / RDDM / condition encoder model wrapper
│   ├── plot/                   # STFT/loss/condition 시각화 유틸
│   └── prepare/                # manifest / noisy data 생성 유틸
├── data/
│   ├── manifest.csv            # 현재 paired manifest
│   └── noisy/                  # noisy wav와 condition segment 파일
├── checkpoints/                # 학습 checkpoint 저장 위치
└── outputs/                    # debug/plot output 저장 위치
```

## 환경 준비

이 저장소에는 별도 `requirements.txt`가 없으므로, 사용하는 환경에 맞춰 주요 패키지를 설치합니다.

```bash
cd /home/jaewoo/MAIR/ryu/robot_denoising/2026-capstone-43/LRDSE

python3 -m pip install torch torchaudio numpy scipy soundfile matplotlib
```

CUDA를 쓸 경우 PyTorch는 로컬 CUDA 버전에 맞는 wheel로 설치해야 합니다.

## 데이터 형식

`SpeechEnhancementDataset`은 manifest CSV에서 `valid=1`인 row만 사용합니다. 핵심 컬럼은 아래와 같습니다.

- `id`, `speaker_id`, `book_id`, `source_id`
- `noisy_wav`: noisy wav/flac 경로
- `clean_wav`: clean wav/flac 경로
- `valid`: 학습 사용 가능 여부

Aux condition을 쓰는 경우 noisy wav의 parent directory를 `run_dir`로 추론합니다. 해당 폴더에는 아래 파일 중 인식 가능한 이름이 있어야 합니다.

```text
<run_dir>/
├── <source_id>.wav
├── anchor_segment.json      # 또는 anchors.json, anchor.json
├── lowstate_segment.jsonl   # 또는 lowstate.jsonl 등
├── highstate_segment.jsonl
└── segment_meta.json
```

기본 audio preprocess 설정은 16 kHz mono 기준입니다.

- `target_sr=16000`
- `n_fft=510`
- `win_length=510`
- `hop_length=128`
- `num_frames=256`
- `target_length=(num_frames - 1) * hop_length = 32640`

`train_sgmse.py`는 `win_length == n_fft`와 위 `target_length` 관계를 검사합니다.

## Manifest 생성

이미 `data/manifest.csv`가 있으면 바로 학습할 수 있습니다. 새로 만들 때는 noisy root와 clean root를 지정합니다.

```bash
python3 -m src.prepare.build_manifest \
  --noisy-root ./data/noisy \
  --clean-root /path/to/clean \
  --out ./data/manifest.csv \
  --target-sr 16000 \
  --val-ratio 0.1 \
  --split-by speaker
```

기본 출력:

- `data/manifest.csv`: 전체 row
- `data/manifest_train.csv`: train split
- `data/manifest_val.csv`: validation split

## SGMSE 빠른 확인

작은 샘플로 loss forward/backward가 도는지 먼저 확인할 때:

```bash
python3 train_sgmse.py \
  --manifest ./data/manifest.csv \
  --save-dir ./checkpoints/sgmse_debug \
  --device cuda \
  --overfit-samples 8 \
  --batch-size 2 \
  --max-steps 20 \
  --log-every 1 \
  --disable-checkpoint-save
```

CUDA가 없으면 `--device cpu`로 실행합니다.

## SGMSE 학습

Validation manifest가 있는 일반 학습 예시:

```bash
python3 train_sgmse.py \
  --manifest ./data/manifest_train.csv \
  --val-manifest ./data/manifest_val.csv \
  --save-dir ./checkpoints/sgmse_se \
  --device cuda \
  --batch-size 4 \
  --num-workers 2 \
  --lr 1e-4 \
  --max-epochs 300 \
  --save-every-epochs 5 \
  --sample-every-epochs 50 \
  --num-sample-wavs 1
```

Validation set 없이 현재 `data/manifest.csv`만 사용할 때:

```bash
python3 train_sgmse.py \
  --manifest ./data/manifest.csv \
  --save-dir ./checkpoints/sgmse_se \
  --device cuda \
  --batch-size 4 \
  --max-epochs 300
```

`--sample-every` 또는 `--sample-every-epochs`를 켜면 `--val-manifest`가 필요합니다.

## Temp Contact Condition 학습

Foot force contact channel을 쓰려면 `--use-temp-condition`을 추가합니다. 현재 SGMSE path는 `backbone=ncsnpp_v2` 입력을 `[x.real, x.imag, y.real, y.imag, foot0, foot1, foot2, foot3]` 8채널로 구성합니다.
각 foot channel은 audio frame의 `CLOCK_MONOTONIC` time window 안에서 해당 foot force가 `50`을 넘으면 `1`, 아니면 `0`입니다. 기본값은 foot force timestamp를 `+58.5ms` 늦춘 뒤 정렬합니다.

```bash
python3 train_sgmse.py \
  --manifest ./data/manifest_train.csv \
  --val-manifest ./data/manifest_val.csv \
  --save-dir ./checkpoints/sgmse_temp_contact \
  --device cuda \
  --batch-size 4 \
  --max-epochs 300 \
  --use-temp-condition
```

threshold와 lag는 `--temp-contact-threshold`, `--temp-contact-lag-ms`로 조정할 수 있습니다.

## Step 1: Condition Encoder

`train_condition_encoder.py`는 noisy speech가 아니라 robot noise-only audio와 foot force만으로 noise-only STFT magnitude prior를 예측하는 condition encoder 학습 코드입니다. 입력 feature는 STFT hop인 8 ms frame마다 4 legs × `[mean, max, std, p95, dmean, dmax_abs] = 24ch`로 만들고, target은 `log(1 + |STFT(noise)|)` `[256, 256]`입니다.

실시간 denoising까지 고려해 기본 condition encoder는 causal depthwise-separable TCN입니다. 기본값은 `hidden=64`, `layers=4`, `kernel=3`, `max_dilation=8`이며, encoder receptive field는 31 frames, 약 248 ms의 과거 force history입니다. 온라인 추론에서는 future look-ahead 없이 history buffer만 유지하면 됩니다.

먼저 GO2 noise-only run 폴더를 manifest로 만듭니다. 현재 기본 noise root는 `./data/noise`이며, 이 경로는 `/home/jaewoo/Downloads/output/go2_train/` symlink입니다. 각 run은 `0001/audio.wav`, `0001/anchor.json`, `0001/lowstate.jsonl`, `0001/highstate.jsonl` 형태를 기대합니다. `contaminated/` 아래 run은 기본 제외됩니다.

```bash
python3 -m src.prepare.build_noise_only_manifest \
  --noise-root ./data/noise \
  --out ./data/noise_only_manifest.csv \
  --target-sr 16000 \
  --val-ratio 0.1
```

`contaminated/`까지 포함해서 manifest를 만들고 싶으면 `--recursive --include-contaminated`를 추가합니다.

기존 paired/noisy manifest는 Step 1의 권장 입력이 아닙니다. 디버그 호환을 위해 `segment_meta.json`에서 원본 robot noise crop을 역추적하는 경로는 남겨두었지만, 실제 Step 1 실험은 `noise_only_manifest_train.csv` / `noise_only_manifest_val.csv`를 사용합니다.

기본 L1 magnitude loss로 시작:

```bash
python3 train_condition_encoder.py \
  --manifest ./data/noise_only_manifest_train.csv \
  --val-manifest ./data/noise_only_manifest_val.csv \
  --save-dir ./checkpoints/condition_encoder \
  --device cuda \
  --batch-size 16 \
  --num-workers 2 \
  --max-epochs 3 \
  --eval-every-epochs 1 \
  --save-every-epochs 1 \
  --auto-delay
```

위 설정은 총 3 epoch를 돌고, 매 epoch 끝에서 validation을 수행한 뒤 `latest.pt`와 개선된 경우 `best.pt`를 저장합니다.

예전의 무거운 dense Conv1d TCN으로 비교하려면 `--encoder-conv-type standard --hidden-channels 256 --num-layers 8 --kernel-size 5 --max-dilation 16`을 추가합니다.

학습이 안정화된 뒤 band/event loss를 추가:

```bash
python3 train_condition_encoder.py \
  --manifest ./data/noise_only_manifest_train.csv \
  --val-manifest ./data/noise_only_manifest_val.csv \
  --save-dir ./checkpoints/condition_encoder_band_event \
  --device cuda \
  --batch-size 16 \
  --auto-delay \
  --band-weight 0.5 \
  --event-weight 0.1 \
  --event-percentile 85
```

Sanity check:

```bash
python3 -m src.check.check_condition_encoder_sanity \
  --checkpoint ./checkpoints/condition_encoder/latest.pt \
  --manifest ./data/noise_only_manifest_val.csv \
  --out-dir ./outputs/condition_encoder_sanity \
  --device cuda \
  --batch-size 8
```

이 스크립트는 force derivative magnitude와 noise energy의 cross-correlation delay, matched vs shuffled loss ratio, time-shift loss curve, `target M / predicted M_hat / error / shuffled-force prediction` heatmap을 저장합니다. 기대 기준은 `shuffled_target_ratio > 1.2`이며, time-shift curve는 추정 delay 근처에 valley가 생기는지 확인합니다.

## Checkpoint와 Resume

기본 저장 파일:

- `args.json`: 실행 설정
- `latest.pt`: 최근 checkpoint
- `best.pt`: validation loss가 있으면 best validation, 없으면 저장 시점 train loss 기준
- `epoch_losses.csv`: epoch 평균 train/validation loss
- `samples/step_*/`: validation sample에서 저장한 noisy/clean/enhanced wav

Resume:

```bash
python3 train_sgmse.py \
  --manifest ./data/manifest_train.csv \
  --val-manifest ./data/manifest_val.csv \
  --save-dir ./checkpoints/sgmse_se \
  --resume ./checkpoints/sgmse_se/latest.pt
```

Step별 checkpoint도 남기려면 `--save-step-checkpoints`를 추가합니다.

## SGMSE Denoising 추론

학습된 `latest.pt` 또는 `best.pt`를 불러와 특정 noisy wav를 denoising할 때:

```bash
python3 denoise_sgmse.py \
  --checkpoint ./checkpoints/sgmse_se/latest.pt \
  --noisy-wav /path/to/noisy.wav \
  --out ./outputs/denoise_sgmse/noisy_enhanced.wav \
  --device cuda
```

출력 경로를 생략하면 `./outputs/denoise_sgmse/<파일명>_enhanced.wav`로 저장합니다.

```bash
python3 denoise_sgmse.py \
  --checkpoint ./checkpoints/sgmse_se/best.pt \
  --noisy-wav ./data/noisy/.../sample.wav \
  --sampling-N 30
```

Aux condition으로 학습한 checkpoint라면 noisy wav의 parent directory를 `run_dir`로 자동 사용합니다. 별도 위치를 쓰려면 `--run-dir /path/to/run_dir`를 지정합니다.

## Loss 시각화

```bash
python3 -m src.plot.plot_epoch_loss \
  --csv ./checkpoints/sgmse_se/epoch_losses.csv \
  --out ./outputs/plots/epoch_loss.png \
  --smooth-window 3
```

## Noisy 데이터 합성

Clean source audio와 robot noise run을 섞어 `data/noisy` 구조를 만들 때 사용합니다.

```bash
python3 -m src.prepare.make_noisy_data \
  --source-root /path/to/source_speech_root \
  --noise-root /path/to/robot_noise_root \
  --out-root ./data/noisy \
  --snr-min -5 \
  --snr-max 5 \
  --seed 0
```

Noise run 폴더에는 `audio.wav`, `anchor.json`, `lowstate.jsonl`, `highstate.jsonl`이 필요합니다. 출력 sample 폴더에는 noisy wav와 `anchor_segment.json`, `lowstate_segment.jsonl`, `highstate_segment.jsonl`, `segment_meta.json`이 생성됩니다.

## 유용한 스크립트

```bash
# dataset 로딩과 STFT inverse check
python3 -m src.check.check_dataset \
  --manifest ./data/manifest.csv \
  --num-samples 4 \
  --save-debug

# RDDM forward/backward smoke test
python3 -m src.check.check_model_forward \
  --manifest ./data/manifest.csv \
  --batch-size 1 \
  --device cpu \
  --dim 8 \
  --dim-mults '(1, 2, 4)' \
  --timesteps 10 \
  --sampling-timesteps 2

# wav의 STFT plot 저장
python3 -m src.plot.plot_stft --wav ./data/noisy/.../sample.wav --out ./outputs/debug/stft_plot.png

# data/noisy 아래 wav duration 확인
python3 -m src.check.check_duration --root ./data/noisy

# lowstate foot force 통계 확인
python3 -m src.check.check_max_foot_force --root ./data/noisy --pattern lowstate_segment.jsonl
```

## RDDM 실험

`train_rddm.py`도 현재 디렉터리에서 바로 실행할 수 있습니다.

```bash
python3 train_rddm.py \
  --manifest ./data/manifest.csv \
  --save-dir ./checkpoints/rddm_se \
  --device cuda \
  --batch-size 2 \
  --max-steps 1000
```

모든 실행 예시는 현재 환경에 맞춰 `python3` 기준으로 작성했습니다.
