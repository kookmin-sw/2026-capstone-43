# 04_spatialast_FOA_conv

`03_spatialast_FOA`를 기반으로 FOA stem 폭을 바꿔보는 ablation 디렉토리입니다.  
목표는 초기 stem 압축 비율이 방향 단서 보존에 미치는 영향을 검증하는 것입니다.

## 추가된 실험 포인트

- `baseline`: `7 -> 16 -> 1`
- `conv32_out4`: `7 -> 32 -> 4`
- `conv32_out8`: `7 -> 32 -> 8`
- `conv64_out8`: `7 -> 64 -> 8`
- `conv64_out16`: `7 -> 64 -> 16`

## 주요 파일

- `backbone.py`: stem variant 분기 포함
- `scripts/train_stage12_*`: Stage-12 학습 스크립트
- `tools/build_stage12_subset.py`: subset manifest 생성
- `tools/compare_stage12_runs.py`: 실험 결과 비교

## 실행 예시

```bash
cd 04_spatialast_FOA_conv
pip install -r requirements.txt

python tools/build_stage12_subset.py
bash scripts/train_stage12_subset_foa_baseline_slow.sh
bash scripts/train_stage12_subset_foa_conv64_out8_slow.sh
python tools/compare_stage12_runs.py
```

## 참고

기본 데이터 로딩/학습 루프는 `03_spatialast_FOA`와 동일하며, 차이는 stem 설정과 Stage-12 실험 스크립트입니다.
