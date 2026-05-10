# LRDSE

**LRDSE**는 *Legged Robot Diffusion Speech Enhancement*를 위한 연구 코드입니다.  
목표는 사족보행 로봇이 이동하거나 발을 지면에 접촉할 때 발생하는 불규칙적이고 non-stationary한 robot noise를 줄이고, 사람 음성을 더 명확하게 복원하는 것입니다.

기본 학습 경로는 `train_sgmse.py`입니다.  
noisy sample 폴더에 `anchor` / `lowstate` 파일이 있으면 `foot_force`와 그 derivative를 8ch auxiliary condition으로 사용할 수 있습니다.

---

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

## 4. 주요 실험 관점

비교해야 하는 기본 실험은 다음과 같습니다.

```text
No condition baseline
    - audio만 사용한 speech enhancement

Foot force condition model
    - audio + foot_force condition 사용
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

`lowstate_segment.jsonl`과 `anchor_segment.json`이 noisy sample 폴더에 있을 때 사용합니다.

```bash
python3 train_sgmse.py \
  --manifest ./data/manifest_train.csv \
  --val-manifest ./data/manifest_val.csv \
  --save-dir ./checkpoints/sgmse_aux \
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

### 7.2 foot_force condition checkpoint로 denoising

```bash
python3 train_sgmse.py \
  --manifest ./data/manifest_train.csv \
  --val-manifest ./data/manifest_val.csv \
  --save-dir ./outputs/denoise_sgmse_aux \
  --resume ./checkpoints/sgmse_aux/best.pt \
  --device cuda \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 1 \
  --lr 0 \
  --sample-every 1 \
  --num-sample-wavs 5 \
  --sample-max-sec 0 \
  --disable-checkpoint-save \
  --use-aux-cond \
  --aux-cond-dim 8
```

결과 wav는 아래 위치에 저장됩니다.

```text
outputs/denoise_sgmse/samples/step_*/
├── *_noisy_full.wav
├── *_clean_full.wav
└── *_enhanced_full.wav
```

`*_enhanced_full.wav`가 denoising 결과입니다.

---

## 8. 참고 연구 흐름

LRDSE는 다음 흐름을 참고합니다.

- SGMSE: complex STFT domain에서 score-based generative model을 사용하는 speech enhancement
- Diffusion-based speech enhancement: noisy speech에서 clean speech distribution으로 복원하는 generative approach
- RDDM: restoration 문제에서 residual diffusion과 noise diffusion을 분리해 해석하는 관점
- Robot state-conditioned enhancement: audio 외부의 robot state를 condition으로 사용해 robot self-generated noise를 제거하는 방향

---

## 9. References

- Simon Welker, Julius Richter, Timo Gerkmann, “Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain”, Interspeech 2022.  
  https://arxiv.org/abs/2203.17004

- Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Timo Gerkmann, “Speech Enhancement and Dereverberation with Diffusion-Based Generative Models”, IEEE/ACM TASLP 2023.  
  https://arxiv.org/abs/2208.05830

- Jiawei Liu, Qiang Wang, Huijie Fan, Yinong Wang, Yandong Tang, Liangqiong Qu, “Residual Denoising Diffusion Models”, CVPR 2024.  
  https://arxiv.org/abs/2308.13712

- Unitree Robotics, Go2 SDK / LowState interface documentation.  
  https://support.unitree.com/home/en/developer/Basic_services
