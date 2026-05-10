# 16_model_architectures

이 디렉토리는 SpatialAudio 내 구현 모델들의 아키텍처를 문서화/도식화하기 위한 공간입니다.

## 작성 규칙

- 각 파일의 `mermaid` 블록을 실제 구조에 맞게 채워 넣습니다.
- 입력/출력 텐서 shape를 가능한 한 명시합니다.
- 학습 loss 연결 지점(예: azimuth/elevation/vector, PIT, ACCDOA)을 다이어그램에 포함합니다.

## 모델별 템플릿

- [03_spatialast_FOA.md](./03_spatialast_FOA.md)
- [04_spatialast_FOA_conv.md](./04_spatialast_FOA_conv.md)
- [05_spatialast_FOA_conv64x2.md](./05_spatialast_FOA_conv64x2.md)
- [06_spatialast_FOA_frontreg.md](./06_spatialast_FOA_frontreg.md)
- [07_spatialast_FOA_front9_and_reg.md](./07_spatialast_FOA_front9_and_reg.md)
- [08_multi_accdoa_head.md](./08_multi_accdoa_head.md)
- [15_sciphi.md](./15_sciphi.md)
- [15_sciphi_dinov2_b.md](./15_sciphi_dinov2_b.md)
