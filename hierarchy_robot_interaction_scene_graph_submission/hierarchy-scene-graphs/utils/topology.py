import json
import os

import numpy as np
import open3d as o3d


def _write_compatible_point_cloud(path, point_cloud):
    o3d.io.write_point_cloud(path, point_cloud, write_ascii=True)


class Floor:
    def __init__(self, floor_id, name=None):
        self.floor_id = floor_id
        self.name = name
        self.rooms = []
        self.txt_embeddings = []
        self.pcd = None
        self.vertices = []
        self.floor_height = None
        self.floor_zero_level = None

    def add_room(self, room):
        self.rooms.append(room)

    def save(self, path):
        _write_compatible_point_cloud(os.path.join(path, f"{self.floor_id}.ply"), self.pcd)
        metadata = {
            "floor_id": self.floor_id,
            "name": self.name,
            "rooms": [room.room_id for room in self.rooms],
            "vertices": np.asarray(self.vertices).tolist(),
            "floor_height": self.floor_height,
            "floor_zero_level": self.floor_zero_level,
        }
        with open(os.path.join(path, f"{self.floor_id}.json"), "w") as outfile:
            json.dump(metadata, outfile)

    def load(self, path):
        self.pcd = o3d.io.read_point_cloud(os.path.join(path, f"{self.floor_id}.ply"))
        with open(os.path.join(path, f"{self.floor_id}.json")) as infile:
            metadata = json.load(infile)
        self.name = metadata["name"]
        self.rooms = metadata["rooms"]
        self.vertices = np.asarray(metadata["vertices"])
        self.floor_height = metadata["floor_height"]
        self.floor_zero_level = metadata["floor_zero_level"]


class Room:
    def __init__(self, room_id, floor_id, name=None):
        self.room_id = room_id
        self.name = name
        self.category = None
        self.floor_id = floor_id
        self.objects = []
        self.vertices = []
        self.embeddings = []
        self.pcd = None
        self.room_height = None
        self.room_zero_level = None
        self.represent_images = []
        self.object_counter = 0
        self.adjacent_room_ids = []
        self.adjacent_room_names = []
        self.adjacent_room_instance_names = []
        self.room_type_index = None
        self.instance_name = None

    def add_object(self, objectt):
        self.objects.append(objectt)

    def save(self, path):
        _write_compatible_point_cloud(os.path.join(path, f"{self.room_id}.ply"), self.pcd)
        metadata = {
            "room_id": self.room_id,
            "name": self.name,
            "floor_id": self.floor_id,
            "objects": [obj.object_id for obj in self.objects],
            "vertices": np.asarray(self.vertices).tolist(),
            "room_height": self.room_height,
            "room_zero_level": self.room_zero_level,
            "embeddings": [np.asarray(emb).tolist() for emb in self.embeddings],
            "represent_images": self.represent_images,
            "adjacent_room_ids": self.adjacent_room_ids,
            "adjacent_room_names": self.adjacent_room_names,
            "adjacent_room_instance_names": self.adjacent_room_instance_names,
            "room_type_index": self.room_type_index,
            "instance_name": self.instance_name,
        }
        with open(os.path.join(path, f"{self.room_id}.json"), "w") as outfile:
            json.dump(metadata, outfile)

    def load(self, path):
        self.pcd = o3d.io.read_point_cloud(os.path.join(path, f"{self.room_id}.ply"))
        with open(os.path.join(path, f"{self.room_id}.json")) as infile:
            metadata = json.load(infile)
        self.name = metadata["name"]
        self.floor_id = metadata["floor_id"]
        self.vertices = np.asarray(metadata["vertices"])
        self.room_height = metadata["room_height"]
        self.room_zero_level = metadata["room_zero_level"]
        self.embeddings = [np.asarray(emb) for emb in metadata["embeddings"]]
        self.represent_images = metadata["represent_images"]
        self.adjacent_room_ids = metadata.get("adjacent_room_ids", [])
        self.adjacent_room_names = metadata.get("adjacent_room_names", [])
        self.adjacent_room_instance_names = metadata.get(
            "adjacent_room_instance_names", []
        )
        self.room_type_index = metadata.get("room_type_index")
        self.instance_name = metadata.get("instance_name")


class Object:
    def __init__(self, object_id, room_id, name=None):
        self.object_id = object_id
        self.vertices = None
        self.embedding = None
        self.pcd = None
        self.room_id = room_id
        self.name = name
        self.gt_name = None
        self.caption = None
        self.source_id = None

    def save(self, path):
        _write_compatible_point_cloud(os.path.join(path, f"{self.object_id}.ply"), self.pcd)
        metadata = {
            "object_id": self.object_id,
            "vertices": np.asarray(self.vertices).tolist(),
            "room_id": self.room_id,
            "name": self.name,
            "embedding": self.embedding.tolist() if self.embedding is not None else "",
            "caption": self.caption if self.caption is not None else "",
            "source_id": self.source_id if self.source_id is not None else "",
        }
        with open(os.path.join(path, f"{self.object_id}.json"), "w") as outfile:
            json.dump(metadata, outfile)

    def load(self, path):
        self.pcd = o3d.io.read_point_cloud(os.path.join(path, f"{self.object_id}.ply"))
        with open(os.path.join(path, f"{self.object_id}.json")) as infile:
            metadata = json.load(infile)
        self.vertices = np.asarray(metadata["vertices"])
        self.room_id = metadata["room_id"]
        self.name = metadata["name"]
        self.embedding = (
            np.asarray(metadata["embedding"]) if metadata["embedding"] != "" else None
        )
        self.caption = metadata.get("caption", "") or None
        self.source_id = metadata.get("source_id", "") or None

    def __add__(self, other):
        if self.pcd.is_empty():
            return other
        if other.pcd.is_empty():
            return self
        self.pcd += other.pcd
        self.vertices = self.pcd.get_axis_aligned_bounding_box().get_box_points()
        if self.embedding is not None and other.embedding is not None:
            self.embedding = np.mean([self.embedding, other.embedding], axis=0)
        return self
