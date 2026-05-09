# 08_multi_accdoa_head

Multi-ACCDOA source-slot head를 독립적으로 검증하는 실험 디렉토리입니다.  
기존 SpatialAST 코드와 분리되어 있어 head/loss/PIT 매칭을 단독 테스트하기 좋습니다.

## 설계 요약

- 기본 슬롯 수: `Kmax=3`
- 슬롯별 출력:
  - `accdoa`: `[3]`
  - `class_logits`: `[C]`
  - `distance`: `[1]`
- 출력 텐서:
  - `accdoa`: `[B, Kmax, 3]`
  - `class_logits`: `[B, Kmax, C]`
  - `distance`: `[B, Kmax, 1]`

## 주요 파일

- `src/heads.py`: joint multi-source head
- `src/pit.py`: exhaustive permutation PIT 매칭
- `src/losses.py`: active/inactive 분리 loss
- `src/metrics.py`: threshold sweep/top-k 지표
- `scripts/run_pit_demo.py`: 슬롯 교환 PIT 예시
- `scripts/run_sanity_train.py`: toy overfit 학습

## 실행

```bash
cd 08_multi_accdoa_head
pip install -r requirements.txt
python -m pytest -q
python scripts/run_pit_demo.py
python scripts/run_sanity_train.py
```

## 참고

향후 실제 encoder 출력(`slot_tokens: [B, Kmax, D]`)에 연결할 때 head/loss 인터페이스를 그대로 재사용할 수 있도록 구성되어 있습니다.
