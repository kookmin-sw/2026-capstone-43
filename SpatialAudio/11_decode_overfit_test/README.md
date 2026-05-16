# 11_decode_overfit_test

decode overfit 실험 결과를 분석하기 위한 보관/분석 디렉토리입니다.  
원본 대용량 결과 파일은 제거되었고, 분석 스크립트와 요약 정보 중심으로 유지됩니다.

## 구성

- `01_decoded_3way` ~ `10_decode_glos_gnlos_8way`: 실험 폴더 자리
- `analysis_20260416_182603/analyze_decode_results.py`: 비교 분석 스크립트

## 스크립트 실행

```bash
cd 11_decode_overfit_test
python analysis_20260416_182603/analyze_decode_results.py
```

## 주의사항

- 분석 스크립트 상단 `ROOT`, `ANALYSIS_DIR`, `DATASET_ROOT`가 절대경로로 하드코딩되어 있습니다.
- 현재 저장소 경로에 맞게 해당 상수를 먼저 수정해야 정상 실행됩니다.
- raw `epoch_*_decode.jsonl`이 없는 상태에서는 전체 재분석이 불가능할 수 있습니다.
