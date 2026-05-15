# Online Gaussian Splatting 등록 파이프라인

이 문서는 Piper 카메라 수집 결과를 online Gaussian Splatting SLAM prototype에 등록하는 방법을 정리한다.

## 전체 구조

```text
ROS collector
  /camera/color/image_raw
  /camera/aligned_depth_to_color/image_raw
  /camera/color/camera_info
  /robot/odom + base_link -> camera TF
    -> rgb/*.png
    -> depth/*.png
    -> poses.csv
    -> camera_info.json

online_gs_slam
  LiveRgbPoseDataset
    -> Frame dataclass
    -> GaussianTracker
    -> OnlineMapper
    -> GaussianInserter
    -> GaussianMap checkpoint
```

## 1. 3Hz RGB-D + Pose 수집 시작

권장 방식은 로봇에서 image topic을 3Hz로 throttle하고, 노트북에서 collector를 실행하는 것이다.

로봇 `192.168.0.2`에서:

```bash
cd ~/catkin_ws
THROTTLE_DEPTH=True ./scripts/start_robot_image_throttle.sh
```

노트북 `192.168.0.100`에서:

```bash
cd ~/catkin_ws
OUTPUT_DIR=/home/harudev/rgb_pose_dataset_01 \
SAVE_DEPTH=True \
./scripts/start_laptop_rgb_pose_collection.sh
```

수동 실행은 아래와 같다.

로봇:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.4:11311
export ROS_HOSTNAME=192.168.0.2

rosrun topic_tools throttle messages \
  /camera/color/image_raw 3.0 /camera/color/image_raw_3hz
```

노트북:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.0.4:11311
export ROS_IP=192.168.0.100
unset ROS_HOSTNAME

roslaunch uni_navigation collect_rgb_pose.launch \
  image_topic:=/camera/color/image_raw_3hz \
  depth_topic:=/camera/aligned_depth_to_color/image_raw_3hz \
  save_depth:=true \
  output_dir:=/home/harudev/rgb_pose_dataset_01
```

기본 설정:

```text
min_interval: 0.333333
min_translation: 0.0
min_rotation: 0.0
```

즉 약 1초에 3장을 저장한다.

## 1.5. COLMAP 없이 Depth 기반 sparse point cloud 만들기

RealSense aligned depth가 있으면 COLMAP/SfM을 돌리지 않고도 Gaussian Splatting 초기 point cloud를 만들 수 있다.

```bash
cd ~/catkin_ws
DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
KEEP_EVERY=3 \
INCLUDE_DEPTH=True \
GENERATE_POINT_CLOUD=True \
POINT_STRIDE=6 \
MAX_POINTS_PER_FRAME=12000 \
MAX_TOTAL_POINTS=1500000 \
VOXEL_SIZE=0.01 \
./scripts/export_nerfstudio_dataset.sh
```

생성 파일:

```text
/home/harudev/rgb_pose_dataset_01/transforms.json
/home/harudev/rgb_pose_dataset_01/sparse_pc.ply
```

동작:

```text
1. aligned depth pixel을 camera intrinsics로 3D back-projection
2. poses.csv의 camera pose로 world frame에 변환
3. RGB image에서 같은 pixel color를 가져와 point color로 저장
4. voxel downsample 후 sparse_pc.ply 생성
5. transforms.json에 ply_file_path를 기록해서 Nerfstudio splatfacto 초기점으로 사용
```

시각화:

```bash
python3 scripts/view_depth_point_cloud.py \
  /home/harudev/rgb_pose_dataset_01/sparse_pc.ply
```

standalone으로 SfM-style point만 만들고 싶으면:

```bash
python3 scripts/depth_pose_to_sfm_points.py \
  /home/harudev/rgb_pose_dataset_01 \
  --keep-every 3 \
  --point-stride 6 \
  --voxel-size 0.01 \
  --output /home/harudev/rgb_pose_dataset_01/sparse_pc.ply \
  --colmap-points3d-output /home/harudev/rgb_pose_dataset_01/points3D.txt
```

이 `points3D.txt`는 COLMAP text 형식과 비슷한 debug/export 파일이다. 실제 Nerfstudio `splatfacto` 초기화에는 `sparse_pc.ply`와 `transforms.json`의 `ply_file_path`를 쓰는 흐름이 더 직접적이다.

## 1.6. RTAB-Map DB에서 Nerfstudio dataset 만들기

RTAB-Map을 사용해 RGB-D SLAM을 돌린 경우, 저장된 database에서 image, camera pose, assembled point cloud를 바로 export할 수 있다. 이 흐름은 COLMAP을 다시 돌리지 않고 RTAB-Map의 optimized camera trajectory와 depth 기반 point cloud를 `splatfacto` 초기값으로 쓰기 위한 것이다.

RTAB-Map 실행 후 DB는 기본적으로 아래에 저장된다.

```text
~/.ros/rtabmap.db
```

Nerfstudio dataset으로 변환:

```bash
cd ~/catkin_ws

python3 scripts/export_rtabmap_db_to_nerfstudio.py \
  ~/.ros/rtabmap.db \
  --output-dir /home/harudev/rtabmap_nerfstudio_dataset_01
```

생성 파일:

```text
/home/harudev/rtabmap_nerfstudio_dataset_01/
  transforms.json
  images/
  sparse_pc.ply
```

`transforms.json`에는 RTAB-Map camera pose가 Nerfstudio/OpenGL camera convention으로 변환되어 저장된다. `sparse_pc.ply`는 RTAB-Map이 depth image와 optimized poses로 조립한 initial point cloud다.

point cloud를 먼저 확인:

```bash
python3 scripts/view_depth_point_cloud.py \
  /home/harudev/rtabmap_nerfstudio_dataset_01/sparse_pc.ply
```

Nerfstudio `splatfacto` 실행:

```bash
source ~/miniconda3/bin/activate ns310

ns-train splatfacto \
  --data /home/harudev/rtabmap_nerfstudio_dataset_01 \
  --viewer.websocket-host 0.0.0.0 \
  --viewer.make-share-url True \
  --pipeline.datamanager.camera-res-scale-factor 0.35 \
  --pipeline.datamanager.images-on-gpu False \
  --pipeline.datamanager.cache-images cpu
```

> [!WARNING]
> RTAB-Map odometry가 자주 실패한 DB는 camera pose 자체가 불안정할 수 있다. 이 경우 `sparse_pc.ply`가 먼저 휘거나 겹쳐 보이고, `splatfacto` 결과도 흐리게 나온다. Gaussian 학습 전에 point cloud와 camera trajectory를 먼저 확인하는 것이 좋다.

## 1.7. Gaussian label을 4D hash grid field로 학습하기

VLM/Grounded-SAM 등으로 얻은 2D segmentation mask를 여러 view에서 Gaussian으로 누적하면, 각 Gaussian 또는 point에 semantic label supervision을 만들 수 있다. 이 supervision을 discrete Gaussian attribute로만 저장하지 않고 `(x, y, z, t)`를 입력으로 받는 4D multi-scale hash grid field로 학습할 수 있다.

이 방향은 LEGS(Language-Embedded Gaussian Splats)의 language-embedded Gaussian representation과 잘 맞는다. LEGS는 mobile robot으로 room-scale Gaussian splat을 incremental하게 만들고, language feature를 Gaussian map에 결합한다. 여기서는 우선 VLM mask에서 얻은 Gaussian/point label을 4D hash grid로 distill하는 lightweight prototype으로 시작한다.

### 1.7.1. splatfacto Gaussian을 VLM으로 labeling하기

학습된 `splatfacto` checkpoint를 Gaussian PLY로 export한다.

```bash
source ~/miniconda3/bin/activate ns310
cd ~/catkin_ws

ns-export gaussian-splat \
  --load-config outputs/rtabmap_nerfstudio_dataset_01/splatfacto/2026-05-13_163403/config.yml \
  --output-dir outputs/rtabmap_nerfstudio_dataset_01/exported_gaussians \
  --output-filename splat_rgb.ply \
  --ply-color-mode rgb
```

CLIPSeg dependency를 설치한다.

```bash
python -m pip install transformers
```

Nerfstudio `transforms.json`의 camera views에 CLIPSeg mask를 만들고, Gaussian centers를 각 view에 project해서 label vote를 누적한다.

```bash
PYTHONPATH=catkin_ws python3 catkin_ws/scripts/label_gaussians_with_clipseg.py \
  --gaussian-ply outputs/rtabmap_nerfstudio_dataset_01/exported_gaussians/splat_rgb.ply \
  --transforms /home/harudev/rtabmap_nerfstudio_dataset_01/transforms.json \
  --data-dir /home/harudev/rtabmap_nerfstudio_dataset_01 \
  --prompts floor table chair wall robot \
  --output outputs/hash_grid/semantic_points_clipseg.npz \
  --preview-ply outputs/hash_grid/semantic_points_clipseg_preview.ply \
  --max-gaussians 120000 \
  --max-frames 24 \
  --mask-threshold 0.45
```

> [!NOTE]
> CLIPSeg는 Grounded-SAM보다 가볍고 설치가 쉽지만 mask 품질은 제한적이다. 이 스크립트의 출력 형식은 `semantic_points.npz`로 고정되어 있으므로, 이후 Grounded-SAM/LEGS-style language feature로 바꿔도 4D hash grid 학습 코드는 그대로 쓸 수 있다.

입력 supervision 파일 형식:

```text
semantic_points.npz
  xyz: float32 [N, 3]
  labels: int64 [N]
  time: float32 [N]              # optional
  weights: float32 [N]           # optional
  class_names: str [C]           # optional
```

학습:

```bash
cd ~/catkin_ws

python3 scripts/train_4d_hash_grid_field.py \
  --samples outputs/hash_grid/semantic_points_clipseg.npz \
  --output outputs/hash_grid/semantic_hash_grid.pt \
  --preview-ply outputs/hash_grid/semantic_preview.ply \
  --steps 2000 \
  --batch-size 8192
```

출력:

```text
outputs/hash_grid/semantic_hash_grid.pt
outputs/hash_grid/semantic_preview.ply
```

`semantic_hash_grid.pt`는 4D hash grid + MLP head checkpoint다. `semantic_preview.ply`는 학습된 field가 각 supervision point에 예측한 semantic label을 색으로 칠한 point cloud라 Open3D/MeshLab/CloudCompare로 확인할 수 있다.

> [!NOTE]
> 현재 구현은 `tiny-cuda-nn` 없이 PyTorch만 사용하는 research prototype이다. 느리지만 구조를 직접 바꾸기 쉽다. 최적화가 필요해지면 같은 API를 유지하고 backend만 fused CUDA/tiny-cuda-nn으로 교체하면 된다.

## 2. Online GS 등록 실행

Ubuntu 20.04의 기본 Python은 3.8이므로 최신 PyTorch/typing-extensions가 안 맞을 수 있다. 아래처럼 Python 3.8 호환 버전을 설치한다.

```bash
python3 -m pip install --user --upgrade "pip<25" wheel
python3 -m pip install --user "typing_extensions<4.14"
python3 -m pip install --user torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install --user gsplat==1.5.3
python3 scripts/check_gsplat_backend.py
```

주의: `gsplat`은 PyTorch CUDA runtime만으로는 부족하고 CUDA toolkit의 `nvcc`가 필요하다. `gsplat: No CUDA toolkit found. gsplat will be disabled.`가 뜨면 CUDA toolkit 12.1 설치와 `PATH` 설정이 필요하다.

`nvcc --version`이 CUDA 10.1처럼 오래된 버전이면 gsplat 빌드가 실패한다.

```text
nvcc fatal: Unknown option '-generate-dependencies-with-compile'
```

이 경우 CUDA toolkit 12.1을 설치하고 아래 환경변수를 잡는다.

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$HOME/.local/bin:$PATH
nvcc --version
python3 scripts/check_gsplat_backend.py
```

그 다음 실행:

```bash
python run_online_gs_slam.py \
  --data_dir /home/harudev/rgb_pose_dataset_01 \
  --config configs/online_gs_slam.yaml \
  --output_dir outputs/scene01
```

짧게 테스트:

```bash
python run_online_gs_slam.py \
  --data_dir /home/harudev/rgb_pose_dataset_01 \
  --config configs/online_gs_slam.yaml \
  --output_dir outputs/scene01 \
  --max_frames 30 \
  --idle_timeout 5
```

## 3. 현재 등록 방식

현재 prototype은 gsplat renderer를 아직 붙이지 않은 최소 구조다.

프레임이 새로 들어오면:

```text
1. poses.csv의 새 row 감지
2. rgb/000xxx.png 로드
3. camera_info.json에서 intrinsics 로드
4. ROS optical pose를 OpenGL/Nerfstudio camera-to-world로 변환
5. Frame dataclass 생성
6. Tracker는 초기 pose를 그대로 사용
7. Mapper가 visible Gaussian observation_count/opacity 갱신
8. RGB-only pseudo-depth ray sampling으로 새 Gaussian 삽입
9. gsplat renderer로 현재 Gaussian map 렌더링
10. rendered RGB와 입력 RGB의 L1 loss로 local Gaussian parameter 최적화
11. uncertainty 업데이트
12. checkpoint/trajectory 저장
```

RGB-D depth가 연결되면 `GaussianInserter.propose_from_frame()`에서 depth backprojection으로 바꾸면 된다.

## 4. Output

```text
outputs/scene01/
  checkpoints/
    gaussians_000009.pt
    gaussians_latest.pt
  trajectory.json
  summary.json
  gaussians_latest.ply
  debug/
    compare_latest.png
    compare_000009.png
  high_uncertainty_centers.npy
  uncertainty_debug.png
```

## 5. 시각화

현재 prototype은 `gsplat` 렌더링과 간단한 키보드 viewer를 지원하지만, Nerfstudio만큼 보기 편한 web viewer는 아니다. 빠른 확인은 debug 이미지 또는 PLY로 하고, 편하게 둘러보려면 아래 Nerfstudio/Splatfacto 흐름을 사용한다.

Online GS 실행 중 아래 파일이 계속 갱신된다.

```text
outputs/scene01/gaussians_latest.ply
```

간단히 파일 갱신을 watch:

```bash
PLY_PATH=/home/harudev/catkin_ws/outputs/scene01/gaussians_latest.ply \
./scripts/watch_gaussian_map.sh
```

볼 수 있는 프로그램:

```text
MeshLab
CloudCompare
Open3D
```

Open3D가 있으면:

```bash
python3 -c "import open3d as o3d; p=o3d.io.read_point_cloud('outputs/scene01/gaussians_latest.ply'); o3d.visualization.draw_geometries([p])"
```

이 PLY는 Gaussian의 center와 RGB color만 보여준다. Nerfstudio viewer용 Gaussian splat 파일이 아니라 단순 확인용 point cloud다.

실제 Gaussian Splatting 렌더 결과는 debug comparison 이미지로 확인한다.

```text
outputs/scene01/debug/compare_latest.png
```

이미지는 좌에서 우로:

```text
input RGB | gsplat rendered RGB | absolute error
```

간단한 `gsplat` checkpoint viewer:

```bash
python3 scripts/view_gsplat_checkpoint.py \
  --output_dir outputs/scene01 \
  --data_dir /home/harudev/rgb_pose_dataset_01 \
  --checkpoint outputs/scene01/checkpoints/gaussians_001939.pt
```

## 6. Nerfstudio / Splatfacto로 보기

직접 만든 viewer가 보기 불편하면 수집한 `rgb + pose`를 Nerfstudio dataset으로 변환해서 `splatfacto`를 돌리는 것이 가장 편하다.

변환:

```bash
cd ~/catkin_ws
DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
KEEP_EVERY=5 \
INCLUDE_DEPTH=True \
GENERATE_POINT_CLOUD=True \
./scripts/export_nerfstudio_dataset.sh
```

생성 파일:

```text
/home/harudev/rgb_pose_dataset_01/transforms.json
/home/harudev/rgb_pose_dataset_01/sparse_pc.ply
```

Nerfstudio 설치 후 학습:

```bash
DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
KEEP_EVERY=5 \
MAX_NUM_ITERATIONS=7000 \
./scripts/train_nerfstudio_splatfacto.sh
```

`ns-train splatfacto`는 Nerfstudio web viewer 주소를 터미널에 출력한다. 브라우저에서 그 주소를 열면 마우스로 카메라를 조종하면서 Gaussian Splatting 결과를 볼 수 있다.

주의:

- 현재 repo의 `gaussians_latest.ply`는 Nerfstudio splat viewer용 파일이 아니다.
- Nerfstudio에는 `rgb/` 이미지와 `transforms.json`을 넘겨서 `splatfacto`가 다시 학습하게 하는 흐름이 더 안정적이다.
- 6739프레임 전체를 바로 넣으면 오래 걸릴 수 있으니 처음에는 `KEEP_EVERY=5` 또는 `KEEP_EVERY=10`으로 시작한다.

확인:

```bash
xdg-open outputs/scene01/debug/compare_latest.png
```

카메라를 직접 움직이면서 실제 gsplat 렌더를 보고 싶으면:

```bash
python3 scripts/view_gsplat_checkpoint.py \
  --output_dir outputs/scene01 \
  --data_dir /home/harudev/rgb_pose_dataset_01
```

키:

```text
W/S: forward/back
A/D: left/right
Q/E: down/up
Arrow keys: yaw/pitch
R: reset pose
P: save snapshot
ESC: quit
```

한 장만 렌더해서 저장:

```bash
python3 scripts/view_gsplat_checkpoint.py \
  --output_dir outputs/scene01 \
  --data_dir /home/harudev/rgb_pose_dataset_01 \
  --once
```

## 6. 다음 구현 단계

우선순위:

```text
1. gsplat backend renderer 구현
2. rendered RGB vs frame RGB residual 계산
3. SE(3) pose refinement loop 구현
4. residual/high-coverage 기반 Gaussian insertion
5. local keyframe window optimization
6. uncertainty-guided exploration target 출력
7. tactile embedding update 연결
```

Renderer만 `NullGaussianRenderer`에서 gsplat wrapper로 교체하면 tracking/mapping 코드는 최대한 유지되도록 설계했다.
