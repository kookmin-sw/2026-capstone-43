# 05_spatialast_FOA_conv64x2

`04_spatialast_FOA_conv`에서 stem 폭 실험을 확장해, stem 깊이(다층 구성)까지 비교하는 Stage-13 실험 디렉토리입니다.

## 추가된 stem variant

- `conv32_32_out8`: `7 -> 32 -> 32 -> 8`
- `conv64_64_out8`: `7 -> 64 -> 64 -> 8`
- `conv64_64_out16`: `7 -> 64 -> 64 -> 16`

기존 `baseline/conv32/conv64` 계열 variant도 함께 유지됩니다.

## 주요 파일

- `backbone.py`: 2-stage/3-stage stem 구성 지원
- `scripts/train_stage13_*`: Stage-13 학습 스크립트
- `tools/build_stage13_subset.py`: subset manifest 생성
- `tools/compare_stage13_runs.py`: Stage-13 비교 리포트

## 실행 예시

```bash
cd 05_spatialast_FOA_conv64x2
pip install -r requirements.txt

python tools/build_stage13_subset.py
bash scripts/train_stage13_subset_foa_baseline_slow.sh
bash scripts/train_stage13_subset_foa_conv64_64_out8_slow.sh
python tools/compare_stage13_runs.py
```

## 참고

학습 프레임워크 자체는 03/04와 동일하고, Stage-13에서 stem 깊이 비교 실험이 추가된 버전입니다.
