import gzip
import pickle
import re
from pathlib import Path

import numpy as np
import open3d as o3d


STRUCTURAL_OBJECT_SUBSTRINGS = [
    "wall",
    "floor",
    "ceiling",
    "railing",
    "roof",
    "void",
    "unlabeled",
    "misc",
    "baseboard",
    "doorpost",
    "door frame",
    "frame",
    "paneling",
]


def normalize_object_name(name: str) -> str:
    if name is None:
        return ""
    normalized = name.lower().replace("_", " ").replace("/", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def should_skip_object_name(name: str) -> bool:
    normalized = normalize_object_name(name)
    return any(token in normalized for token in STRUCTURAL_OBJECT_SUBSTRINGS)


def load_conceptgraph_objects(
    mapping_path: str,
    min_points: int = 0,
    min_detections: int = 1,
    include_background: bool = False,
):
    mapping_path = Path(mapping_path)
    with gzip.open(mapping_path, "rb") as infile:
        results = pickle.load(infile)

    loaded_objects = []
    for obj in results["objects"]:
        if not include_background and obj.get("is_background", False):
            continue
        if int(obj.get("num_detections", 1)) < int(min_detections):
            continue

        points = np.asarray(obj.get("pcd_np", []), dtype=np.float64)
        if points.shape[0] < int(min_points):
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = np.asarray(obj.get("pcd_color_np", []), dtype=np.float64)
        if colors.shape[0] == points.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        embedding = obj.get("clip_ft")
        if embedding is not None:
            embedding = np.asarray(embedding, dtype=np.float32)

        loaded_objects.append(
            {
                "source_id": str(obj.get("curr_obj_num", obj.get("id", ""))),
                "name": obj.get("class_name", "object"),
                "caption": obj.get("consolidated_caption"),
                "embedding": embedding,
                "pcd": pcd,
                "num_detections": int(obj.get("num_detections", 1)),
                "n_points": int(obj.get("n_points", points.shape[0])),
            }
        )
    return loaded_objects

