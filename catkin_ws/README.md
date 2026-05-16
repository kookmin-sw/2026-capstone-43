# Piper RGB-Pose Collection and Continual Gaussian Splatting Prototype

This repository contains the ROS tools, dataset conversion scripts, and early research code used to collect RGB images with robot/camera poses from Piper and train Gaussian Splatting models on a laptop GPU.

The long-term goal is an incremental / continual 3D Gaussian Splatting SLAM framework for robotics:

- online RGB or RGB-D frame ingestion
- camera pose tracking
- Gaussian map update and insertion
- local optimization
- uncertainty-aware exploration
- future tactile/material-aware Gaussian representation

> [!WARNING]
> This is a research prototype, not a production SLAM system. The Nerfstudio path is useful for visualization and offline/staged experiments, while the custom `online_gs_slam` package is where true continual Gaussian SLAM should be developed.

## Current Status

The repository currently supports three main workflows.

1. Collect RGB images and robot poses from ROS at about 3 Hz.
2. Convert the collected dataset to Nerfstudio format and train `splatfacto`.
3. Run an early custom `online_gs_slam` prototype with frame-by-frame Gaussian registration hooks.

The practical setup used here is:

- Robot IP: `192.168.0.2`
- Laptop wired IP: `192.168.0.100`
- Camera topic: `/camera/color/image_raw`
- Throttled camera topic: `/camera/color/image_raw_3hz`
- Pose source: `/robot/odom`
- Laptop GPU: NVIDIA RTX 3060 Laptop GPU
- Conda env: `~/miniconda3/envs/ns310`

## Repository Layout

```text
src/uni_navigation/
  launch/
    collect_rgb_pose.launch
    uni_navigation_demo.launch
  scripts/
    collect_rgb_pose.py
    rgb_pose_to_nerfstudio.py

online_gs_slam/
  data/
  tracking/
  mapping/
  rendering/
  material/
  utils/

scripts/
  start_laptop_rgb_pose_collection.sh
  start_robot_image_throttle.sh
  export_nerfstudio_dataset.sh
  train_nerfstudio_splatfacto.sh
  train_nerfstudio_splatfacto_ns310.sh
  train_nerfstudio_continual_stages.sh
  check_gsplat_backend.py
  view_gsplat_checkpoint.py

configs/
  online_gs_slam.yaml

docs/
  server_time_sync.md
  mapping_and_navigation.md
  piper_rgb_pose_collection.md
  online_gs_slam_pipeline.md
```

## Robot Network and Time Sync

The robot has no direct GPU, so the robot publishes ROS data and the laptop performs dataset logging and Gaussian training.

See:

- `docs/server_time_sync.md`
- `docs/mapping_and_navigation.md`

In this setup, the robot keeps `192.168.0.2` fixed and the laptop uses `192.168.0.100` on the wired interface. Time sync was handled with `ntpdate` because `chrony.service` was masked on the robot.

## RGB + Pose Collection

Start the throttled image stream on the robot:

```bash
cd ~/catkin_ws
./scripts/start_robot_image_throttle.sh
```

On the laptop, collect RGB images and poses:

```bash
cd ~/catkin_ws
DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
./scripts/start_laptop_rgb_pose_collection.sh
```

The dataset writer saves:

```text
rgb_pose_dataset_01/
  rgb/
    000000.png
    000001.png
    ...
  poses.csv
  poses_tum.txt
  camera_info.json
  metadata.json
```

The default collection rate is about 3 Hz.

## Nerfstudio Dataset Export

Convert the collected ROS dataset into Nerfstudio format:

```bash
cd ~/catkin_ws
DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
OUTPUT=/home/harudev/catkin_ws/outputs/nerfstudio_data/piper \
KEEP_EVERY=3 \
START_INDEX=1000 \
./scripts/export_nerfstudio_dataset.sh
```

Sampling options:

- `KEEP_EVERY=3` means export one frame every 3 original frames.
- `START_INDEX=1000` skips the first 1000 collected frames.
- `MAX_FRAMES=512` caps the exported dataset size.
- If `MAX_FRAMES` is unset, all sampled frames after `START_INDEX` are exported.

> [!NOTE]
> `KEEP_EVERY=3` does not mean "add 3 images at once"; it means frame stride 3.

## Nerfstudio / gsplat Environment

Activate the Nerfstudio environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ns310
```

Check CUDA, PyTorch, and gsplat:

```bash
python -c "import torch, gsplat; print(torch.__version__, torch.cuda.is_available(), gsplat.__version__)"
```

Expected environment:

```text
torch 2.4.1+cu121
cuda available True
gsplat 1.4.0+pt24cu121
nerfstudio 1.1.5
```

The prebuilt `gsplat` wheel is preferred over JIT compilation because local CUDA compilation was slow and memory-heavy.

## Offline Nerfstudio Splatfacto Training

Train a Nerfstudio `splatfacto` baseline:

```bash
cd ~/catkin_ws
unset MAX_FRAMES
DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
START_INDEX=1000 \
KEEP_EVERY=3 \
CAMERA_RES_SCALE_FACTOR=0.35 \
MAX_NUM_ITERATIONS=10000 \
MAKE_SHARE_URL=True \
VIEWER_MAX_NUM_DISPLAY_IMAGES=512 \
./scripts/train_nerfstudio_splatfacto_ns310.sh
```

Useful options:

- `CAMERA_RES_SCALE_FACTOR=0.35` reduces image resolution for 16 GB RAM systems.
- `VIEWER_MAX_NUM_DISPLAY_IMAGES=512` controls how many training cameras/images the viewer displays.
- `MAKE_SHARE_URL=True` creates a temporary Viser share URL.
- `MAX_NUM_ITERATIONS=10000` controls training length.

> [!IMPORTANT]
> Nerfstudio reads the dataset at startup. If new images are added to the folder while training is running, `ns-train splatfacto` does not automatically ingest them as new training frames.

## Staged Continual Nerfstudio Experiment

For a continual-learning-like experiment, use cumulative stages:

1. Build a subset of frames.
2. Train one stage.
3. Save checkpoint.
4. Add more frames.
5. Resume from the previous checkpoint.

Stable staged command:

```bash
cd ~/catkin_ws
SOURCE_DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
EXPERIMENT_NAME=piper_continual_stride20_stable \
SAMPLE_STRIDE=20 \
INITIAL_SAMPLED_FRAMES=40 \
ADD_SAMPLED_FRAMES=40 \
NUM_STAGES=9 \
ITERATIONS_PER_STAGE=1000 \
VIS=tensorboard \
MAKE_SHARE_URL=False \
CAMERA_RES_SCALE_FACTOR=0.35 \
REFINE_EVERY=1000000 \
RESET_ALPHA_EVERY=1000000 \
STOP_SPLIT_AT=0 \
STOP_SCREEN_SIZE_AT=0 \
CULL_ALPHA_THRESH=0.0 \
CULL_SCALE_THRESH=1000000.0 \
DENSIFY_GRAD_THRESH=1000000.0 \
./scripts/train_nerfstudio_continual_stages.sh
```

> [!WARNING]
> This is a Nerfstudio staged-resume workaround, not true online continual Gaussian Splatting. Enabling densify/prune can improve quality, but it caused CUDA device-side assert failures in `gsplat/strategy/default.py::_prune_gs` during checkpoint resume. The stable command disables most densify/prune behavior to avoid those crashes, but the result can become blurry because new Gaussians are not aggressively inserted.

## Custom Online GS SLAM Prototype

The `online_gs_slam` package is the research backbone for true continual GS:

- `Frame` and `CameraIntrinsics` dataclasses
- live RGB-pose dataset reader
- `GaussianMap` with means, scales, rotations, opacity, color, observation count, uncertainty, and material features
- Gaussian insertion module
- local mapper
- keyframe manager
- uncertainty utilities
- tactile/material extension hooks
- `gsplat` renderer abstraction

Run the prototype:

```bash
cd ~/catkin_ws
DATA_DIR=/home/harudev/rgb_pose_dataset_01 \
OUTPUT_DIR=/home/harudev/catkin_ws/outputs/scene01 \
./scripts/start_online_gs_registration.sh
```

Outputs:

```text
outputs/scene01/
  checkpoints/
    gaussians_latest.pt
  gaussians_latest.ply
  trajectory.json
  summary.json
  debug/
    compare_latest.png
```

> [!NOTE]
> `gaussians_latest.ply` is mainly a point-cloud-style export of Gaussian centers/colors. It is not the same as an interactive Nerfstudio Gaussian viewer export.

## Visualizing Custom Checkpoints

Use the lightweight OpenCV viewer:

```bash
cd ~/catkin_ws
python3 scripts/view_gsplat_checkpoint.py \
  --output_dir outputs/scene01 \
  --data_dir /home/harudev/rgb_pose_dataset_01 \
  --checkpoint outputs/scene01/checkpoints/gaussians_latest.pt
```

Controls:

- `W/S`: move forward/back
- `A/D`: move left/right
- `Q/E`: move down/up
- arrow keys: yaw/pitch
- `R`: reset camera
- `P`: save snapshot
- `ESC`: quit

## Mapping and Navigation Notes

For 2D navigation:

1. Run SLAM such as gmapping.
2. Drive the robot around to build the occupancy map.
3. Save the map while gmapping is still publishing `/map`.
4. Stop gmapping.
5. Launch localization/navigation with the saved map.

See `docs/mapping_and_navigation.md` for the exact commands and known `map_server` pitfalls.

## Known Issues

- Nerfstudio `splatfacto` does not support live image ingestion during one running training process.
- Staged Nerfstudio resume can crash if densify/prune is enabled across stages.
- The custom `online_gs_slam` implementation is still early and needs stronger residual-based insertion, RGB-D backprojection, and local bundle-style optimization.
- RGB-only Gaussian insertion currently relies on rough placeholder depth assumptions.
- Camera poses are mostly trusted from odometry; full photometric pose tracking is a future target.
- Large image sets can freeze a 16 GB RAM laptop if full-resolution caching or too many viewer images are enabled.

## Research Direction

The intended next step is to move beyond staged Nerfstudio training and implement true continual Gaussian Splatting inside `online_gs_slam`:

- ingest frames one by one from ROS or a live dataset folder
- optimize only a local keyframe window
- insert new Gaussians from unexplained/high-residual regions
- estimate uncertainty per Gaussian
- select high-uncertainty regions for robot exploration
- integrate tactile feedback as material embeddings on nearby Gaussians

## Related Work

Language-Embedded Gaussian Splats (LEGS) is a close reference for this project:

- Paper: "Language-Embedded Gaussian Splats (LEGS): Incrementally Building Room-Scale Representations with a Mobile Robot"
- Code: https://github.com/uynitsuj/LEGS
- Project: https://berkeleyautomation.github.io/LEGS/

LEGS is useful here because it connects three ideas that match this repository's direction:

- incremental room-scale Gaussian map construction with a mobile robot
- language/semantic embedding attached to Gaussian representations
- Nerfstudio-style custom method integration for training and visualization

This project currently keeps the implementation lighter and more modular: RTAB-Map/Nerfstudio are used for practical data export and visualization, while `online_gs_slam` is the experimental backbone for continual RGB-D Gaussian insertion, uncertainty, and later tactile/material updates.

Future work:

- RGB-D depth backprojection insertion
- monocular depth prior for RGB-only mode
- SE(3) photometric tracking loop
- local keyframe window optimization
- active uncertainty-guided exploration
- tactile encoder
- material-aware Gaussian embedding
- semantic/material Gaussian fields
- LEGS-style language supervision projected from 2D masks onto Gaussian-level features
