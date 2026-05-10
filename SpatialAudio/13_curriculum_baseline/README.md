# 14_curriculum_baseline

curriculum 학습과 end-to-end 학습을 비교 분석하는 디렉토리입니다.  
분석용 파이썬 스크립트는 유지되어 있으며, 원본 대량 결과 파일은 정리된 상태입니다.

## 주요 파일

- `analyze_curriculum_runs.py`: run 디렉토리 자동 스캔 후 성능 분석
- `build_curriculum_analysis.py`: curriculum/end-to-end 정렬 비교 리포트 생성
- `analysis_utils.py`: 공통 파싱/유틸 함수

## 실행 예시

```bash
cd 14_curriculum_baseline

python analyze_curriculum_runs.py \
  --root . \
  --output-dir analysis_outputs

python build_curriculum_analysis.py
```

## 주의사항

- 일부 스크립트는 특정 폴더 구조(`epoch_*`, `metrics/*.json`, `*_decode.jsonl`)를 전제로 작성되어 있습니다.
- 현재 디렉토리에 raw decode/metrics가 없으면 전체 분석 재생성은 제한됩니다.
- 필요 시 스크립트 내부의 run 경로 가정(예: `01_endtoend`, `02_curriculum`)을 현재 구조에 맞게 수정해야 합니다.
