# 15_sciphi_baselines

`99_archive`에 있던 Sci-Phi 계열 구현에서 **모델 구현 코드만** 분리해 가져온 디렉토리입니다.

## 포함한 소스

- `99_archive/Sci-Phi/spatial_branch`
- `99_archive/Sci-Phi-DINOv2-B/spatial_branch`

## 제외한 항목

- `Sci-Phi-DINOv2-B/phi4mm_clean` 전체
- 학습/추론/평가 스크립트(`train_*`, `inference_*`, `eval_*`, `inspect_*` 등)
- `__pycache__`
- `test_data`

## 현재 구조

```text
15_sciphi_baselines/
├── Sci-Phi/
│   └── spatial_branch/
└── Sci-Phi-DINOv2-B/
    └── spatial_branch/
```
