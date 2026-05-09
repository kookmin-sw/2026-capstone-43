# LRDSE

Robot noise speech enhancement 실험용 코드입니다. Clean speech와 quadruped robot noise가 섞인 noisy speech를 paired manifest로 학습하며, 기본 학습 경로는 `train_sgmse.py`의 SGMSE 기반 모델입니다. Noisy sample 폴더에 robot `anchor` / `lowstate` 파일이 있으면 foot force 8ch auxiliary condition도 함께 사용할 수 있습니다.

## 폴더 구조

```text
LRDSE/
├── train_sgmse.py              # SGMSE speech enhancement 학습 / 검증 / 샘플 저장
├── train_rddm.py               # RDDM 실험 코드
├── dataset.py                  # paired clean/noisy dataset, condition 로딩
├── src/
│   ├── audio/preprocess.py     # wav crop, STFT, spec transform, inverse transform
│   ├── condition/preprocess.py # foot_force condition token 생성
│   └── models/                 # SGMSE / RDDM model wrapper
├── scripts/
│   ├── build_manifest.py       # noisy/clean pair manifest 생성 및 train/val split
│   ├── plot_epoch_loss.py      # epoch loss CSV plot
│   ├── plot_stft.py            # wav STFT 시각화
│   └── ...
├── data/
│   ├── make_noisy_data.py      # clean speech + robot noise 합성
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
python3 scripts/build_manifest.py \
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

## Aux Condition 학습

Foot force condition을 쓰려면 `--use-aux-cond`를 추가합니다. 현재 SGMSE aux path는 `backbone=ncsnpp_v2`, `aux_cond_dim=8` 조합을 전제로 합니다.

```bash
python3 train_sgmse.py \
  --manifest ./data/manifest_train.csv \
  --val-manifest ./data/manifest_val.csv \
  --save-dir ./checkpoints/sgmse_aux \
  --device cuda \
  --batch-size 4 \
  --max-epochs 300 \
  --use-aux-cond \
  --aux-cond-dim 8
```

Condition preprocessing은 `lowstate`에서 4ch raw foot force와 4ch derivative를 만들고, crop 구간에 맞춰 `[8, 1024]` token으로 패딩합니다.

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

## Loss 시각화

```bash
python3 scripts/plot_epoch_loss.py \
  --csv ./checkpoints/sgmse_se/epoch_losses.csv \
  --out ./outputs/plots/epoch_loss.png \
  --smooth-window 3
```

## Noisy 데이터 합성

Clean source audio와 robot noise run을 섞어 `data/noisy` 구조를 만들 때 사용합니다.

```bash
python3 data/make_noisy_data.py \
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
python3 scripts/check_dataset.py \
  --manifest ./data/manifest.csv \
  --num-samples 4 \
  --save-debug

# RDDM forward/backward smoke test
python3 scripts/check_model_forward.py \
  --manifest ./data/manifest.csv \
  --batch-size 1 \
  --device cpu \
  --dim 8 \
  --dim-mults '(1, 2, 4)' \
  --timesteps 10 \
  --sampling-timesteps 2

# wav의 STFT plot 저장
python3 scripts/plot_stft.py --wav ./data/noisy/.../sample.wav --out ./outputs/debug/stft_plot.png

# data/noisy 아래 wav duration 확인
python3 data/check_duration.py --root ./data/noisy

# lowstate foot force 통계 확인
python3 data/check_max_foot_force.py --root ./data/noisy --pattern lowstate_segment.jsonl
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
