# LRDSE

**LRDSE**는 *Legged Robot Diffusion Speech Enhancement*를 위한 연구 코드입니다.  
목표는 사족보행 로봇이 이동하거나 발을 지면에 접촉할 때 발생하는 불규칙적이고 non-stationary한 robot noise를 줄이고, 사람 음성을 더 명확하게 복원하는 것입니다.

기본 학습 경로는 `train_sgmse.py`입니다.  
noisy sample 폴더에 `anchor` / `lowstate` 파일이 있으면 `foot_force`와 그 derivative를 8ch auxiliary condition으로 사용할 수 있습니다.

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

## 1. 연구 배경

일반적인 speech enhancement는 배경 noise, 실내 환경 noise, 잡음이 섞인 음성 등을 제거하는 데 초점을 둡니다. 하지만 legged robot 환경에서는 robot 자체의 움직임이 noise 발생 원인이 됩니다.

특히 Unitree GO2와 같은 사족보행 로봇은 발이 지면에 반복적으로 닿고 떨어지며, 이 과정에서 순간적이고 불규칙적인 contact noise가 발생할 수 있습니다. 이 noise는 단순히 일정하게 유지되는 stationary noise가 아니라, 보행 패턴, 속도, 자세, 회전, 지면 상태에 따라 계속 변하는 non-stationary noise입니다.

따라서 본 프로젝트는 speech enhancement 문제를 단순 audio-only denoising으로 보지 않고, robot state를 함께 사용하는 conditional speech enhancement 문제로 다룹니다.

---

## 2. 핵심 아이디어

LRDSE의 핵심 아이디어는 robot noise가 발 접촉과 강하게 관련될 수 있다는 점을 이용하는 것입니다.

`lowstate`에서 얻은 `foot_force`는 각 발의 접촉 상태를 설명할 수 있는 cue로 사용할 수 있습니다. 이 프로젝트에서는 다음 8ch 값을 auxiliary condition으로 사용합니다.

```text
4ch foot_force
4ch foot_force derivative
```

`foot_force`는 현재 발 접촉의 크기 정보를 제공하고, derivative는 접촉 변화가 급격히 발생하는 순간을 강조합니다. 이를 diffusion speech enhancement model에 condition으로 제공하면, model이 speech 성분과 foot-contact 기반 robot noise를 더 잘 구분할 수 있을 것으로 기대합니다.

단, 현재 구현에서 `foot_force`는 speech enhancement를 위한 auxiliary cue로 사용됩니다. 정확한 물리 단위의 Ground Reaction Force를 직접 추정하거나 보정하는 것이 목표는 아닙니다.

---

## 3. 전체 pipeline

```text
Clean speech
    +
Robot noise recording
    +
Robot state logs
    |
    v
Noisy data generation
    |
    ├── noisy wav
    ├── anchor_segment.json
    ├── lowstate_segment.jsonl
    ├── highstate_segment.jsonl
    └── segment_meta.json
    |
    v
Manifest
    |
    v
STFT preprocessing
    |
    ├── noisy complex spectrogram
    ├── clean complex spectrogram
    └── foot_force auxiliary condition
    |
    v
SGMSE-based diffusion speech enhancement
    |
    v
Enhanced speech
```

---

```bash
python3 -m src.prepare.build_manifest \
  --noisy-root ./data/noisy \
  --clean-root /path/to/clean \
  --out ./data/manifest.csv \
  --target-sr 16000 \
  --val-ratio 0.1 \
  --split-by speaker
```

단순히 train loss만 비교하기보다 validation loss, enhanced wav, contact noise가 강한 구간의 spectrogram, 실제 청감 결과를 함께 확인하는 것이 좋습니다.

---

## 5. 실행 위치

모든 명령은 `LRDSE` 디렉터리에서 실행합니다.

```bash
cd /home/jaewoo/MAIR/ryu/robot_denoising/2026-capstone-43/LRDSE
```

필요 패키지는 환경에 맞게 설치합니다.

```bash
python3 -m pip install torch torchaudio numpy scipy soundfile matplotlib
```

---

## 6. SGMSE 학습

### 6.1 No condition 학습

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

### 6.2 foot_force condition 학습

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
  --num-workers 2 \
  --lr 1e-4 \
  --max-epochs 300 \
  --save-every-epochs 5 \
  --sample-every-epochs 50 \
  --num-sample-wavs 1 \
  --use-aux-cond \
  --aux-cond-dim 8
```

학습 결과는 `--save-dir` 아래에 저장됩니다.

```text
checkpoints/sgmse_aux/
├── latest.pt
├── best.pt
├── args.json
├── epoch_losses.csv
└── samples/
```

---

## 7. 학습된 가중치로 denoising

현재 코드는 별도 `denoise.py`가 아니라 `train_sgmse.py`의 sample 저장 기능으로 denoising 결과를 생성합니다.

`--resume`에 학습된 checkpoint를 넣고, `--sample-every 1`을 사용하면 validation manifest의 sample에 대해 enhanced wav가 저장됩니다.

### 7.1 No condition checkpoint로 denoising

```bash
python3 train_sgmse.py \
  --manifest ./data/manifest_train.csv \
  --val-manifest ./data/manifest_val.csv \
  --save-dir ./outputs/denoise_sgmse \
  --resume ./checkpoints/sgmse_se/best.pt \
  --device cuda \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 1 \
  --lr 0 \
  --sample-every 1 \
  --num-sample-wavs 5 \
  --sample-max-sec 0 \
  --disable-checkpoint-save
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

결과 wav는 아래 위치에 저장됩니다.

```bash
python3 -m src.prepare.make_noisy_data \
  --source-root /path/to/source_speech_root \
  --noise-root /path/to/robot_noise_root \
  --out-root ./data/noisy \
  --snr-min -5 \
  --snr-max 5 \
  --seed 0
```

`*_enhanced_full.wav`가 denoising 결과입니다.

---

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

## 9. References

- Simon Welker, Julius Richter, Timo Gerkmann, “Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain”, Interspeech 2022.  
  https://arxiv.org/abs/2203.17004

- Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Timo Gerkmann, “Speech Enhancement and Dereverberation with Diffusion-Based Generative Models”, IEEE/ACM TASLP 2023.  
  https://arxiv.org/abs/2208.05830

- Jiawei Liu, Qiang Wang, Huijie Fan, Yinong Wang, Yandong Tang, Liangqiong Qu, “Residual Denoising Diffusion Models”, CVPR 2024.  
  https://arxiv.org/abs/2308.13712

- Unitree Robotics, Go2 SDK / LowState interface documentation.  
  https://support.unitree.com/home/en/developer/Basic_services
