import os
import json
import numpy as np
import shutil
from PIL import Image
from scipy.spatial.transform import Rotation as R

# ================= [설정] =================
SOURCE_ROOT = "/home/sungbin/isaac/sg/captured_sessions"
TARGET_BASE = "/home/sungbin/extra/data/cg_replica"
# ==========================================

def get_latest_session(root_path):
    all_sessions = [d for d in os.listdir(root_path) if d.startswith("session_")]
    if not all_sessions: return None
    all_sessions.sort()
    return os.path.join(root_path, all_sessions[-1])

def convert():
    latest_session = get_latest_session(SOURCE_ROOT)
    if not latest_session:
        print("❌ 세션을 찾을 수 없습니다.")
        return

    scene_id = os.path.basename(latest_session)
    target_scene_dir = os.path.join(TARGET_BASE, scene_id)
    target_results_dir = os.path.join(target_scene_dir, "results")
    
    if os.path.exists(target_scene_dir):
        shutil.rmtree(target_scene_dir)
    os.makedirs(target_results_dir, exist_ok=True)

    # [수정] record.py가 jpg로 저장하므로 jpg도 찾도록 수정 (버그 수정됨)
    rgb_dir = os.path.join(latest_session, "rgb")
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    all_poses = []

    # ==========================================
    # [블로그 가이드 - 변환 로직 정교화]
    # 문제 1: "카메라가 전진하면 월드도 같이 전진해서 배경이 움직이는 것 같음" (월드 좌표 고정 실패)
    # 해결: Isaac Sim(Z-up) → OpenCV(Y-down, Z-forward) 변환 수식을 정확하게 구성.
    # 일반적인 SLAM이나 Replica는 카메라 프레임(C2W)이 "오른손 좌표계 (Right-Down-Forward, x-y-z)"를 씁니다.
    # Isaac Sim은 기본적으로 로컬 카메라가 (Right-Up-Back, x-y-z) 이며 월드는 (Right-Forward-Up, x-y-z) 즉 Z-up 입니다.
    # 이를 맞추기 위한 완벽한 변환(좌표 축 스왑과 반전) 행렬 T를 앞뒤로 곱해줍니다.
    
    # T_convert: Z-Up 월드를 Y-Down / Z-Forward 월드로 바꾸는 행렬
    # (x_new = x_old, y_new = -z_old, z_new = y_old)
    T_convert = np.array([
        [1,  0,  0, 0],
        [0,  0, -1, 0],
        [0,  1,  0, 0],
        [0,  0,  0, 1],
    ], dtype=float)

    # 카메라 내부 축도 Y를 뒤집고 Z를 뒤집는 반전을 추가해야 할 수 있지만, 
    # 보통 Replica 궤적 변환 시엔 T_convert @ C2W_isaac 이면 해결됩니다.
    # 추가로 Isaac의 `cameraViewTransform` 행렬 자체가 Y-Up/Z-Back 기반일 경우 로컬 축 반전이 필요.
    T_local = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=float)

    print(f"🚀 변환 시작: {scene_id} ({len(rgb_files)} 프레임)")

    for i, filename in enumerate(rgb_files):
        idx_padded = f"{i:06d}"
        
        # 파일명에서 인덱스 추출 (000123.jpg -> 000123)
        base_name = filename.split('.')[0] 
        # 혹시 rgb_000123 형식이라면 _ 뒤를 취함
        if '_' in base_name:
             orig_idx_str = base_name.split('_')[1]
        else:
             orig_idx_str = base_name

        # 1. RGB 저장 (확장자 무관하게 읽어서 무조건 JPG로 통일)
        img_path = os.path.join(rgb_dir, filename)
        img_rgb = Image.open(img_path).convert("RGB")
        img_rgb.save(os.path.join(target_results_dir, f"frame{idx_padded}.jpg"), "JPEG")

        # 2. Depth 저장 (NPY -> 16bit PNG)
        # record.py가 이미 depth/ 폴더에 PNG를 만들었다면 복사 (빠름)
        # 없다면 depth_raw/ 폴더의 NPY를 읽어서 변환 (안전장치)
        npy_path = os.path.join(latest_session, "depth_raw", f"depth_{orig_idx_str}.npy")
        png_path_src = os.path.join(latest_session, "depth", f"{orig_idx_str}.png") # record.py 최신본 대응

        if os.path.exists(png_path_src):
            shutil.copy2(png_path_src, os.path.join(target_results_dir, f"depth{idx_padded}.png"))
        elif os.path.exists(npy_path):
            depth_m = np.load(npy_path)
            # [수정] 무한대(inf)나 NaN 처리 후 변환. 무한대인 부분은 0 (센서 미감지)으로.
            depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 클리핑: uint16 최대 표현값인 65535를 넘지 않게 자릅니다 (약 65.5미터 제한)
            depth_mm = np.clip(depth_m * 1000, 0, 65535).astype(np.uint16)
            Image.fromarray(depth_mm).save(os.path.join(target_results_dir, f"depth{idx_padded}.png"))

        # 3. 포즈 계산
        # 파일명 포맷 대응 (pose_000123.json 또는 000123.json)
        pose_path = os.path.join(latest_session, "poses", f"pose_{orig_idx_str}.json")
        if not os.path.exists(pose_path):
             pose_path = os.path.join(latest_session, "poses", f"{orig_idx_str}.json")

        if os.path.exists(pose_path):
            with open(pose_path, 'r') as f:
                data = json.load(f)
            
            pos = np.array(data["position"])
            quat = np.array(data["quaternion_wxyz"])
            
            # Quaternion [w,x,y,z] -> Rotation Matrix
            # Scipy는 [x,y,z,w] 순서를 받으므로 재배열
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) 
            
            # Isaac C2W (Camera-to-World) 행렬
            c2w_isaac = np.eye(4)
            c2w_isaac[:3, :3] = r.as_matrix()
            c2w_isaac[:3, 3] = pos
            
            # [좌표계 최종 수정] 
            # Isaac Sim의 월드 좌표계(Z-Up)는 그대로 유지하여 바닥이 올바르게 수평(XY 평면)이 되도록 합니다.
            # 카메라의 로컬 축(Isaac: Right-Up-Back)만 OpenCV(Right-Down-Forward)로 변환(T_local)합니다.
            c2w_cv = c2w_isaac @ T_local
            
            all_poses.append(c2w_cv.flatten())

    # 4. traj.txt 저장
    if all_poses:
        np.savetxt(os.path.join(target_scene_dir, "traj.txt"), np.array(all_poses), fmt='%.15e')

    # 5. 씬 객체 목록(scene_objects.json) 복사
    scene_objects_path = os.path.join(latest_session, "scene_objects.json")
    if os.path.exists(scene_objects_path):
        shutil.copy2(scene_objects_path, os.path.join(target_scene_dir, "scene_objects.json"))
        print(f"📦 씬 객체 목록(scene_objects.json) 복사 완료")

    # 6. 카메라 파라미터 (cam_params.json) 생성
    # HOV-SG(create_hybrid_scene_graph.py)가 올바른 3D 포인트 클라우드를 생성하기 위해서는 
    # Isaac Sim의 정확한 Intrinsics 파라미터가 필요합니다.
    cam_info_path = os.path.join(latest_session, "camera_info.json")
    if os.path.exists(cam_info_path):
        with open(cam_info_path, 'r') as f:
            cam_info = json.load(f)
        
        # ZED NARROW / WIDE 등에 맞춰 기록된 값
        # focal_length (mm) / horizontal_aperture (mm) * image_width (px)
        focal_mm = cam_info.get("intrinsics", {}).get("focal", 2.12)
        res_w, res_h = cam_info.get("intrinsics", {}).get("res", [1280, 720])
        
        # Isaac Sim의 센서 가로/세로 어퍼쳐 (NARROW 기준)
        lens_mode = cam_info.get("lens", "NARROW (4.0mm)")
        if "NARROW" in lens_mode:
            h_aperture = 6.02
            v_aperture = 3.39
        else:
            h_aperture = 4.8
            v_aperture = 2.7

        fx = (focal_mm / h_aperture) * res_w
        fy = (focal_mm / v_aperture) * res_h
        cx = res_w / 2.0
        cy = res_h / 2.0

        cam_params = {
            "camera": {
                "w": res_w,
                "h": res_h,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "scale": 1000.0  # 우리는 depth를 1000 곱해서 uint16 PNG로 만들었으므로
            }
        }
        with open(os.path.join(target_scene_dir, "cam_params.json"), "w") as f:
            json.dump(cam_params, f, indent=4)
        print(f"📸 카메라 파라미터 생성 완료: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # ================= [실행 명령어 생성기] =================
    stride = 5  
    exp_suffix = f"r_mapping_stride10"
    exp_dir = os.path.join(TARGET_BASE, scene_id, "exps", exp_suffix)
    
    print(f"\n✅ 데이터 변환 완료!")
    print(f"📂 저장 경로: {target_scene_dir}")
    print(f"\n🚀 [전체 파이프라인 명령어] 아래 내용을 복사해서 순서대로 실행하세요:")
    print("=" * 80)
    
    # 1. Mapping 4mm
    print(f"# 1. ConceptGraphs 4mm 매핑 실행")
    print(f"python slam/rerun_realtime_mapping.py \\\n"
          f"  dataset_root={TARGET_BASE} \\\n"
          f"  scene_id={scene_id} \\\n"
          f"  dataset_config=dataset/dataconfigs/zed4.yaml \\\n"
          f"  image_height=720 image_width=1280 \\\n"
          f"  force_detection=true save_detections=true stride={stride} \\\n"
          f"  scene_objects_file={os.path.join(target_scene_dir, 'scene_objects.json')} \\\n"
          f"  ++exit_early_file=null")
    print("-" * 80)
    print("python conceptgraph/scripts/visualize_cfslam_results.py \
    --result_path latest_pcd_save ")
    # 2. Merge Centers
    print(f"# 2. 객체 중심 정보 병합")
    print(f"python scripts/merge_centers_into_pcd.py \\\n"
          f"  --pcd_path {exp_dir}/pcd_{exp_suffix}.pkl.gz \\\n"
          f"  --obj_json_path {exp_dir}/obj_json_{exp_suffix}.json \\\n"
          f"  --out_pcd_path {exp_dir}/pcd_{exp_suffix}_with_center.pkl.gz")
    print("-" * 80)

    # 3. Build Scene Graph (Gemini)
    print(f"# 3. Gemini를 이용한 엣지 생성 (gemini-2.5-flash)")
    print(f"python scripts/build_scenegraph_with_gemini.py \\\n"
          f"  --pcd_path {exp_dir}/pcd_{exp_suffix}_with_center.pkl.gz \\\n"
          f"  --out_pcd_path {exp_dir}/pcd_{exp_suffix}_gemini_with_center.pkl.gz \\\n"
          f"  --edge_json_path {exp_dir}/edge_json_{exp_suffix}_gemini.json \\\n"
          f"  --node_model gemini-2.5-flash \\\n"
          f"  --edge_model gemini-2.5-flash \\\n"
          f"  --max_dist 3.0 \\\n"
          f"  --max_pairs 200")
    print("-" * 80)

    # 4. Convert Edges
    print(f"# 4. 엣지 포맷 변환 (Gemini -> CFSLAM)")
    print(f"python scripts/convert_gemini_edges_to_cfslam.py \\\n"
          f"  --pcd_path {exp_dir}/pcd_{exp_suffix}_gemini_with_center.pkl.gz \\\n"
          f"  --gemini_edge_path {exp_dir}/edge_json_{exp_suffix}_gemini.json \\\n"
          f"  --out_edge_path {exp_dir}/edge_json_{exp_suffix}_cfslam.json")
    print("-" * 80)

    # 5. 구조물 (벽/바닥/천장) 검출
    print(f"# 5. 구조물 (벽/바닥/천장) 검출")
    print(f"python scripts/detect_structures.py \\\n"
          f"  --scene_dir {target_scene_dir} \\\n"
          f"  --out_json {exp_dir}/structures.json \\\n"
          f"  --obj_json_path {exp_dir}/obj_json_{exp_suffix}.json")
    print("-" * 80)

    # 6. Visualize
    print(f"# 6. 최종 결과 시각화 (U 키를 눌러 구조물 On/Off)")
    print(f"python scripts/visualize_cfslam_results.py \\\n"
          f"  --result_path {exp_dir}/pcd_{exp_suffix}_gemini_with_center.pkl.gz \\\n"
          f"  --edge_file {exp_dir}/edge_json_{exp_suffix}_cfslam.json \\\n"
          f"  --structures_json {exp_dir}/structures.json")
    print("=" * 80)

    # 3-v2. Build Scene Graph (Gemini v2 - MST 기반 개선판)
    print(f"\n# ===== [V2 개선판] MST 기반 + 계층 구조 =====")
    print(f"# 3-v2. Gemini v2 엣지 생성 (MST 기반 페어링 + 개선된 프롬프트)")
    print(f"python scripts/build_scenegraph_with_gemini_v2.py \\\n"
          f"  --pcd_path {exp_dir}/pcd_{exp_suffix}_with_center.pkl.gz \\\n"
          f"  --out_pcd_path {exp_dir}/pcd_{exp_suffix}_gemini_v2_with_center.pkl.gz \\\n"
          f"  --edge_json_path {exp_dir}/edge_json_{exp_suffix}_gemini_v2.json \\\n"
          f"  --obj_json_path {exp_dir}/obj_json_{exp_suffix}.json \\\n"
          f"  --edge_model gemini-2.5-flash \\\n"
          f"  --proximity_threshold 0.3 \\\n"
          f"  --min_overlap 0.005 \\\n"
          f"  --build_hierarchy --single_room \\\n"
          f"  --hierarchy_json_path {exp_dir}/hierarchy_{exp_suffix}_gemini_v2.json")
    print("-" * 80)

    # 4-v2. Convert Edges (v2)
    print(f"# 4-v2. 엣지 포맷 변환 (Gemini v2 -> CFSLAM)")
    print(f"python scripts/convert_gemini_edges_to_cfslam.py \\\n"
          f"  --pcd_path {exp_dir}/pcd_{exp_suffix}_gemini_v2_with_center.pkl.gz \\\n"
          f"  --gemini_edge_path {exp_dir}/edge_json_{exp_suffix}_gemini_v2.json \\\n"
          f"  --out_edge_path {exp_dir}/edge_json_{exp_suffix}_v2_cfslam.json")
    print("-" * 80)

    # 5-v2. Visualize (v2)
    print(f"# 5-v2. 최종 결과 시각화 (v2) (U 키를 눌러 구조물 On/Off)")
    print(f"python scripts/visualize_cfslam_results.py \\\n"
          f"  --result_path {exp_dir}/pcd_{exp_suffix}_gemini_v2_with_center.pkl.gz \\\n"
          f"  --edge_file {exp_dir}/edge_json_{exp_suffix}_v2_cfslam.json \\\n"
          f"  --structures_json {exp_dir}/structures.json")
    print("=" * 80)


if __name__ == "__main__":
    convert()