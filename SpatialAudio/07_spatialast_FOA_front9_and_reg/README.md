# 07_spatialast_FOA_front9_and_reg

Stage-14를 확장해 front9 분류, front 회귀, full360 sin-cos 회귀를 함께 다루는 Stage-15/16/17 실험 디렉토리입니다.

## 핵심 포인트

- 방위각 모드:
  - `front9_classification`
  - `front_regression`
  - `full360_sincos_regression`
- manifest 기반 front9 라벨 매핑
- patch embedding pretrained init + unfreeze 학습 정책

## 주요 파일

- `heads.py`, `losses.py`, `train.py`: 방위각 모드별 학습 로직
- `tools/build_stage15_subset.py`: Stage-15 subset 구성
- `tools/build_hm3d_losnlos_*_manifests.py`: full360 계열 manifest 생성
- `scripts/train_stage16_*`, `scripts/train_stage17_*`: 후속 확장 실험 실행 스크립트

## 실행 예시

```bash
cd 07_spatialast_FOA_front9_and_reg
pip install -r requirements.txt

python tools/build_stage15_subset.py
bash scripts/train_stage15_subset_foa_baseline_front9_patchunfreeze.sh
bash scripts/train_stage15_subset_foa_baseline_reg_patchunfreeze.sh
python tools/compare_stage15_runs.py
```

full360 staged 학습:

```bash
bash scripts/train_stage15_hm3d_losnlos_full360_staged_sincos.sh
```

## 참고

이 폴더는 `03~06`의 누적 실험 위에, full360 방향 학습과 대규모 manifest 실험이 추가된 현재 주력 실험 브랜치입니다.
