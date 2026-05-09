# 09_spherical_vision

단일 RGB 입력을 depth/point cloud를 거쳐 학습용 구면 표현 `V_sphere`로 변환하는 파이프라인입니다.

## 목적

- 시각 정보를 방향-정렬된 구면 텐서로 정규화
- 이후 `10_spherical_audio`의 `A_sphere`와 각도 그리드 정합을 가능하게 함

## 파이프라인

1. RGB 입력 로드
2. ZoeDepth로 depth 추정
3. depth -> point cloud 변환
4. point cloud를 구면 bin(`azimuth/elevation`)으로 투영
5. `vision_sphere.npy` 및 메타데이터 저장
6. 8-way 방향 풀링(`vision_8way_pooled.npy`)

## 주요 파일

- `run_mvp.py`: 실행 엔트리포인트
- `src/pipeline.py`: end-to-end 처리
- `src/spherical_projection.py`: 구면 격자/투영
- `src/feature_export.py`: tensor/metadata 저장
- `src/pooling_utils.py`: 8-way 풀링

## 실행 예시

```bash
cd 09_spherical_vision
pip install -r requirements.txt

python run_mvp.py \
  --input /path/to/image_or_dir \
  --output_dir outputs/demo \
  --hfov_deg 69 \
  --num_az_bins 24 \
  --num_el_bins 8
```

## 출력

- `vision_sphere.npy`: `[E, A, C]`
- `vision_sphere_azimuth.npy`: `[A, C]`
- `vision_sphere_meta.json`, `vision_sphere_channels.json`
- `vision_8way_pooled.npy`, `vision_8way_meta.json`
- 디버그 이미지(`summary_panel.png` 등)
