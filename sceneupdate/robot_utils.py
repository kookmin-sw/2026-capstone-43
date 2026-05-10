import numpy as np
import omni.usd
from pxr import UsdGeom, Gf, Usd, UsdPhysics  # noqa: F401
from omni.isaac.core.utils.prims import get_prim_at_path, create_prim  # noqa: F401
import math

def find_robot_path(stage):
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if "fetch" in name or "jackal" in name or "robot" in name:
            if prim.GetTypeName() == "Xform":
                path = str(prim.GetPath())
                parts = path.split("/")
                if len(parts) <= 11 and "link" not in path.lower() and "camera" not in path.lower() and "lidar" not in path.lower() and "collisions" not in path.lower() and "fr3" not in path.lower() and "panda" not in path.lower():
                    return path
    return None

def get_prim_transform(prim_path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return np.array([0, 0, 0]), np.array([1, 0, 0, 0])
    
    pose = omni.usd.get_world_transform_matrix(prim)
    pos = pose.ExtractTranslation()
    quat = pose.ExtractRotation().GetQuaternion()
    return np.array([pos[0], pos[1], pos[2]]), np.array([quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2]])

def quat_to_euler_z(quat):
    # w, x, y, z
    w, x, y, z = quat
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def attach_fresh_lidar(robot_path, base_link_path, simulation_app):
    """Attach a LiDAR sensor to the robot.

    Prefers to attach at the existing 'laser_link' prim (correct height),
    falls back to base_link if not found.

    Returns (lidar_path, lidar_frame_id) — both needed by ROS2Navigator.
    """
    import omni.kit.commands as _omni_kit_cmds  # import here to avoid UnboundLocalError scoping issue
    stage = omni.usd.get_context().get_stage()

    # Search for laser_link (or similar) under base_link
    laser_parent = base_link_path
    lidar_frame = "base_link"
    base_prim = stage.GetPrimAtPath(base_link_path)
    if base_prim.IsValid():
        for prim in Usd.PrimRange(base_prim):
            name = prim.GetName().lower()
            if name in ("laser_link", "laser", "lidar_link", "lidar"):
                laser_parent = str(prim.GetPath())
                lidar_frame = prim.GetName()
                print(f"[Lidar] Found laser prim: {laser_parent} → frame='{lidar_frame}'")
                break

    lidar_path = f"{laser_parent}/Lidar"
    try:
        _omni_kit_cmds.execute(
            "RangeSensorCreateLidar", path="Lidar", parent=laser_parent,
            min_range=0.35, max_range=8.0, draw_points=False, draw_lines=False,
            horizontal_fov=360.0, horizontal_resolution=1.0,
            rotation_rate=0.0, high_lod=False, yaw_offset=0.0,
            enable_semantics=False
        )
        print(f"[Lidar] Attached lidar at {lidar_path}, ROS frame='{lidar_frame}'")
        return lidar_path, lidar_frame
    except Exception as e:
        print(f"Failed to attach Lidar: {e}")
        return None, "base_link"

def find_articulation_root(robot_path, dc):
    """Try robot_path, then robot_path/base_link to find the articulation handle.
    Returns (art_handle, art_path) or (0, robot_path) on failure."""
    art = dc.get_articulation(robot_path)
    if art and art != 0:
        return art, robot_path
    base_link = robot_path + "/base_link"
    art = dc.get_articulation(base_link)
    if art and art != 0:
        print(f"[RobotUtils] Articulation found at {base_link}")
        return art, base_link
    print(f"[RobotUtils] WARNING: No articulation found at {robot_path} or {base_link}")
    return 0, robot_path

def setup_wheel_drives(robot_path, dc, art):
    left_idxs = []
    right_idxs = []
    n_dofs = dc.get_articulation_dof_count(art)
    for i in range(n_dofs):
        dof_ptr = dc.get_articulation_dof(art, i)
        dof_name = dc.get_dof_name(dof_ptr).lower()
        if ("left" in dof_name and "wheel" in dof_name) or dof_name == "l_wheel_joint":
            left_idxs.append(i)
            print(f"[WheelDrive] LEFT  DOF[{i}]: {dof_name}")
        elif ("right" in dof_name and "wheel" in dof_name) or dof_name == "r_wheel_joint":
            right_idxs.append(i)
            print(f"[WheelDrive] RIGHT DOF[{i}]: {dof_name}")

    if not left_idxs or not right_idxs:
        print(f"[WheelDrive] WARNING: wheels not found by name. Total DOFs={n_dofs}. Listing all:")
        for i in range(n_dofs):
            dof_ptr = dc.get_articulation_dof(art, i)
            print(f"  DOF[{i}]: {dc.get_dof_name(dof_ptr)}")

    # Configure wheel drives via UsdPhysics.DriveAPI (matches old working code)
    _configure_wheel_drives_usd(robot_path)

    return left_idxs, right_idxs


def _configure_wheel_drives_usd(robot_path):
    """Configure wheel joint drives via UsdPhysics.DriveAPI.

    Uses damping=100000 (same as the proven sayplan_nav2_final.py).
    This works much better than DC properties for velocity control.
    """
    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_path)
    configured = 0
    for prim in Usd.PrimRange(robot_prim):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            name = prim.GetName()
            if "wheel" in name:
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if not drive:
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateStiffnessAttr(0.0)
                drive.CreateDampingAttr(100000.0)
                configured += 1
                print(f"[WheelDrive] USD DriveAPI '{name}': stiffness=0, damping=100000")

    print(f"[WheelDrive] Configured {configured} wheel joints via UsdPhysics.DriveAPI")

def find_scene_prim_near(pos, radius=1.0, object_name=None):
    stage = omni.usd.get_context().get_stage()
    target_pos = np.array(pos)
    best_path = None
    min_dist = float('inf')
    
    for prim in stage.Traverse():
        if prim.GetTypeName() in ["Xform", "Mesh"]:
            if object_name and object_name.lower() not in prim.GetName().lower():
                continue
            
            prim_pos, _ = get_prim_transform(str(prim.GetPath()))
            dist = np.linalg.norm(np.array(prim_pos)[:2] - target_pos[:2])
            
            if dist < radius and dist < min_dist:
                min_dist = dist
                best_path = str(prim.GetPath())
                
    return best_path
