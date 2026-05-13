# Surface QA Dataset Generation Pipeline

Visual QA dataset generation pipeline for **placeable surface understanding** in robotic manipulation contexts.

Two complementary pipelines produce QA pairs that teach vision-language models (VLMs) to identify support surfaces and the objects placed on them:

| Pipeline | Dataset | Scene type | Surface detection method |
|---|---|---|---|
| **GraspNet Tabletop v2** | GraspNet-1Billion | Tabletop close-up | Depth + camera-table transform |
| **ScanNet++ Indoor** | ScanNet++ | Full indoor room | 3D OBB + mesh + ray casting |

---

## Repository Structure

```
scripts/
  graspnet_surface_qa_v2.py       # GraspNet tabletop QA generator (v2)
  scannetpp_indoor_surface_qa.py  # ScanNet++ indoor surface QA generator
annotations/
  graspnet/          # GraspNet test scene QA (scene_0090~)
  graspnet_train1/   # GraspNet train scene QA (scene_0000~)
  scannetpp/         # ScanNet++ indoor QA
```

---

## Pipeline 1 — GraspNet Tabletop v2

### Overview

Generates `table_layout`, `object_layout`, and `freespace` QA for tabletop scenes in the GraspNet-1Billion dataset. Uses the camera-to-table extrinsic matrix and per-pixel object labels for pixel-accurate surface extraction — no mesh or OBB approximation needed.

### Why v2?

v1 problems:
- Estimated table Z from depth only → convex hull extended beyond the actual table edge
- Object subtraction used OBB projection → inaccurate outlines

v2 fixes:
- Uses `cam0_wrt_table.npy` (camera→table frame transform) → table Z=0 plane is exact
- Transforms depth into table frame, keeps only pixels with |z| < 3 cm → pixel-perfect table mask
- Uses `label/{frame}.png` (per-pixel object ID) → exact object footprints, no OBB needed
- Removes convex hull step → correctly handles rounded/partially-visible table edges
- Excludes pixels touching image edges from free space

### Data Requirements

[GraspNet-1Billion](https://graspnet.net/) dataset with the following structure per scene:

```
{scene_id}/
  realsense/
    rgb/{frame:04d}.png
    depth/{frame:04d}.png
    label/{frame:04d}.png
    camera_poses.npy          # (N, 4, 4) camera pose per frame
    cam0_wrt_table.npy        # (4, 4) camera-0 to table transform
    camK.npy                  # (3, 3) intrinsic matrix
```

### Usage

```bash
python scripts/graspnet_surface_qa_v2.py \
    --graspnet_root /path/to/graspnet \
    --scene scene_0093 \
    --out_dir annotations/graspnet/scene_0093 \
    --debug_dir debug/graspnet_qa_v2 \
    [--max_frames 50] \
    [--frame_step 5]
```

### Output QA format

```json
{
  "qa_type": "table_layout",
  "question": "<image>What is the total area of the table surface and what is its polygon shape? Output coordinates in [0, 1000] scale.",
  "answer": "The total table surface occupies 897833 pixels. Its shape is bounded by the polygon: [(75,0), ...]."
}
```

QA types produced:
- `table_layout` — visible table surface polygon and area
- `object_layout` — per-object footprint polygon (with 15 px safety margin)
- `freespace` — free placement region on the table

---

## Pipeline 2 — ScanNet++ Indoor Surface

### Overview

Generates `surface_layout` and `objects_on_surface` QA for indoor scenes in ScanNet++. Detects support surfaces (tables, desks, counters, etc.) using 3D oriented bounding boxes (OBB) from dense reconstruction annotations, refines boundaries with the scene mesh, and verifies visibility with ray casting.

### Key Design Decisions

#### No dependency on depth estimation
The OBB annotations in `segments_anno.json` provide exact 3D extents. The top face of each OBB gives a precise surface plane — no depth map or plane fitting needed.

#### Mesh-based boundary refinement
Instead of projecting the rectangular OBB face, the pipeline extracts actual mesh vertices belonging to the surface object and takes the convex hull of the topmost vertices. This correctly captures round tables, L-shaped counters, and other non-rectangular surfaces.

#### Ray-casting visibility filter
Before generating QA for a surface, the pipeline casts 25 rays from the camera toward sample points on the surface top face. If fewer than 20% of rays reach the surface unobstructed (blocked by walls, furniture, etc.), the surface is skipped. This prevents generating QA for surfaces that are hidden behind walls or completely occluded.

#### Three-stage visibility check
1. **Camera altitude** — camera must be above the surface (no QA for surfaces viewed from below)
2. **Centroid projection** — surface centroid must project inside the image frame
3. **Ray casting** — ≥20% of surface sample points must be directly visible

### Supported Surface Categories

```python
SUPPORT_SURFACE_LABELS = {
    'table', 'desk', 'kitchen counter', 'counter',
    'coffee table', 'dining table', 'end table', 'side table',
    'nightstand', 'tv stand', 'bench',
}
```

### Data Requirements

[ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) dataset with the following structure per scene:

```
{scene_id}/
  scans/
    mesh_aligned_0.05.ply       # dense reconstruction mesh
    segments.json               # vertex → segment ID mapping
    segments_anno.json          # segment groups with OBB + label
  dslr/
    colmap/
      cameras.txt               # OPENCV_FISHEYE camera model
      images.txt                # per-image pose (quaternion + translation)
    resized_images/
      {image_name}.JPG          # DSLR RGB frames
```

### Usage

```bash
# Single scene (with debug images)
python scripts/scannetpp_indoor_surface_qa.py \
    --data_root /path/to/ScanNetPP/data \
    --scenes 09c1414f1b \
    --max_frames 20 \
    --workers 1 \
    --out_json annotations/scannetpp/scannetpp_indoor_qa.json \
    --debug_dir debug/scannetpp_indoor_qa

# All scenes, parallel
python scripts/scannetpp_indoor_surface_qa.py \
    --data_root /path/to/ScanNetPP/data \
    --out_json annotations/scannetpp/scannetpp_indoor_qa.json \
    --debug_dir debug/scannetpp_indoor_qa \
    --max_frames 15 \
    --workers 4

# Fast mode (skip mesh refinement, OBB top face only)
python scripts/scannetpp_indoor_surface_qa.py \
    --data_root /path/to/ScanNetPP/data \
    --out_json annotations/scannetpp/scannetpp_indoor_qa.json \
    --no_mesh_refine
```

### Output QA format

```json
{
  "scene_id": "09c1414f1b",
  "image_name": "DSC05469.JPG",
  "image_path": "/path/to/resized_images/DSC05469.JPG",
  "qa": [
    {
      "qa_type": "surface_layout",
      "question": "<image>What is the visible area and boundary of the coffee table surface? Output coordinates in [0, 1000] scale.",
      "answer": "The visible coffee table surface occupies 219900 pixels. Its boundary polygon is: [(810,1000), (847,781), ...]."
    },
    {
      "qa_type": "objects_on_surface",
      "question": "<image>What objects are on the coffee table?",
      "answer": "The following objects are on the coffee table: remote controller, coaster."
    }
  ]
}
```

QA types produced:
- `surface_layout` — visible surface boundary polygon and pixel area
- `objects_on_surface` — objects physically placed on the surface

---

## Dependencies

```bash
pip install open3d opencv-python numpy shapely scipy tqdm
```

| Package | Usage |
|---|---|
| `open3d` | Mesh loading, ray casting (`o3d.t.geometry.RaycastingScene`) |
| `opencv-python` | Image I/O, fisheye projection (`cv2.fisheye.projectPoints`) |
| `shapely` | 2D polygon operations (intersection, difference, convex hull) |
| `scipy` | 3D convex hull for mesh boundary extraction |
| `tqdm` | Progress bars |

---

## QA Coordinate System

All polygon coordinates are normalised to **[0, 1000]** scale regardless of image resolution:

```
pixel_x_norm = int(pixel_x / image_width  * 1000)
pixel_y_norm = int(pixel_y / image_height * 1000)
```

This makes QA answers resolution-independent and directly usable for fine-tuning VLMs that output normalised coordinates (e.g. Qwen2-VL, InternVL).

---

## Dataset Statistics (ScanNet++ 50 scenes, 15 frames/scene)

| Metric | Value |
|---|---|
| Scenes processed | 50 |
| Scenes with support surfaces | ~37 |
| Frames with visible surfaces | 271 |
| Total QA pairs | 1,004 |
| QA pairs per frame (avg) | ~3.7 |
| Surface types detected | table, desk, kitchen counter, coffee table, dining table, nightstand |
