# 03_spatialast_FOA

FOA 입력을 위한 SpatialAST 기본 학습 패키지입니다.  
backbone/head/loss/train 코드가 분리되어 있어 이후 ablation(04~07)의 기준점 역할을 합니다.

## 핵심 아이디어

- 기본 입력: `WXYZ log-mel(4ch) + IV_x/IV_y/IV_z(3ch)` = 총 7채널
- 원본 채널 순서가 `WYZX`일 때 내부에서 `WXYZ`로 정렬해 사용
- multitask head(방위각/고도/벡터 등)를 모듈식으로 결합

## 주요 파일

- `backbone.py`: FOA stem + SpatialAST transformer
- `heads.py`: 예측 head 모음
- `losses.py`: multitask loss
- `dataset.py`: FOA dataset loader
- `train.py`: 학습 엔트리포인트
- `tools/forward_test.py`, `tools/train_smoke_test.py`: 점검 스크립트

## 빠른 점검

```bash
cd 03_spatialast_FOA
pip install -r requirements.txt

python tools/forward_test.py
python tools/train_smoke_test.py
python tools/check_no_timm_import.py
```

## 학습 예시

```bash
python train.py \
  --train_json /path/to/train.json \
  --val_json /path/to/val.json \
  --audio_path_root /path/to/audio_root \
  --num_classes 355 \
  --batch_size 4 \
  --epochs 10 \
  --audio_normalize \
  --azimuth_loss_weight 2.0 \
  --elevation_loss_weight 2.0 \
  --vector_loss_weight 0.5
```

## 참고

- `scripts/`에는 stage별 실험 실행 스크립트가 정리되어 있습니다.
- `tools/compare_stage*_runs.py`로 실험 간 비교 리포트를 만들 수 있습니다.
