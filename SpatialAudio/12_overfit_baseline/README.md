# 12_overfit_baseline

audio vs AV overfit baseline 비교 결과를 정리한 디렉토리입니다.  
원본 decode/metrics 산출물은 정리되어 있고, 시각화 재생성 스크립트를 중심으로 유지됩니다.

## 구성

- `overfit_01_*` ~ `overfit_08_*`: 실험 폴더 자리
- `analysis_outputs/generate_pairwise_visuals.py`: pairwise 대시보드 생성 스크립트

## 스크립트 실행

```bash
cd 12_overfit_baseline
python analysis_outputs/generate_pairwise_visuals.py
```

## 주의사항

- 스크립트의 `ROOT`가 `12_overfit_baseline_rerun/analysis_outputs`로 하드코딩되어 있습니다.
- 현재 경로/파일명에 맞게 `ROOT`, `PAIR_CSV`, `VAL_CSV`, `DECODE_CSV`를 수정해야 합니다.
- 입력 CSV가 없으면 그림을 다시 생성할 수 없습니다.
