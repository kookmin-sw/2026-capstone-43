# 06_spatialast_FOA_frontreg

Stage-13 구조를 유지하면서, 방위각 supervision을 front-cone 회귀로 확장한 Stage-14 실험 디렉토리입니다.

## 핵심 변경점

- 방위각 head 모드 추가:
  - `full360_classification`: 기존 360-way 분류
  - `front_regression`: `[-45, 45]` 범위 연속 회귀
- front-cone 라벨셋(예: `320,330,...,40`)을 signed angle로 매핑해 학습 가능

## 주요 파일

- `heads.py`: 분류/회귀 방위각 head
- `losses.py`: mode별 target/loss 라우팅
- `train.py`: Stage-14 지표 로깅
- `tools/build_stage14_subset.py`: subset 빌드
- `tools/compare_stage14_runs.py`: 성능 비교

## 실행 예시

```bash
cd 06_spatialast_FOA_frontreg
pip install -r requirements.txt

python tools/build_stage14_subset.py
bash scripts/train_stage14_subset_foa_baseline_cls_slow.sh
bash scripts/train_stage14_subset_foa_baseline_reg_slow.sh
python tools/compare_stage14_runs.py
```

## 참고

stem/transformer 관련 코드는 05와 동일하고, 방위각 supervision 방식 비교가 이 디렉토리의 중심입니다.
