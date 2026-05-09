# 10_spherical_audio

4채널 FOA wav를 방향 기반 구면 텐서 `A_sphere`로 변환하는 오디오 파이프라인입니다.

## 목적

- FOA 신호에서 해석 가능한 방향 특성(beam/AIV/diffuseness 등) 추출
- `09_spherical_vision`과 같은 각도 규약으로 정렬 가능한 오디오 표현 생성

## 파이프라인

1. FOA wav 로드 및 채널 정규화(`WXYZ`)
2. STFT 계산
3. 방향 특징 계산(beam power, active intensity 등)
4. 구면 그리드 투영
5. `audio_sphere.npy` 저장
6. 8-way 풀링(`audio_8way_pooled.npy`)

## 주요 파일

- `run_audio_mvp.py`: 실행 엔트리포인트
- `src/foa_utils.py`: 채널 순서 정규화/입출력
- `src/stft_utils.py`: windowed STFT
- `src/directional_features.py`: 방향 특징 추출
- `src/spherical_projection.py`: 구면 투영
- `src/pooling_utils.py`: 8-way pooling

## 실행 예시

```bash
cd 10_spherical_audio
pip install -r requirements.txt

python run_audio_mvp.py \
  --input /path/to/foa.wav \
  --output_dir outputs/demo \
  --channel_order WYZX \
  --num_az_bins 24 \
  --num_el_bins 8 \
  --aggregation both
```

## 출력

- `audio_sphere.npy`: `[E, A, C]`
- `audio_sphere_azimuth.npy`: `[A, C]`
- `audio_sphere_meta.json`, `audio_sphere_channels.json`
- `audio_8way_pooled.npy`, `audio_8way_meta.json`
- 디버그 시각화(`summary_panel.png`, `beam_power_map.png` 등)
