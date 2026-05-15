from collections import Counter, defaultdict
import os

import cv2
import faiss
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree, distance
from sklearn.cluster import DBSCAN, KMeans
from tqdm import tqdm


def find_intersection_share(map_points, obj_points, radius=0.05):
    obj_tree_points = cKDTree(obj_points)
    _, indices = obj_tree_points.query(
        map_points, k=1, distance_upper_bound=radius, p=2, workers=-1
    )
    indices = indices[indices != obj_points.shape[0]]
    if map_points.shape[0] == 0 or obj_points.shape[0] == 0:
        return 0
    return indices.shape[0] / obj_points.shape[0]


def compute_room_embeddings(
    room_pcds,
    pose_list,
    emb_list,
    pcd_min,
    pcd_max,
    num_views=5,
    save_path=None,
):
    img2room_id = []
    room_id2img_id = defaultdict(list)

    flattened_room_points = []
    plt.figure()
    cmap = cm.get_cmap("tab20")
    for room_idx, room_pcd in enumerate(room_pcds):
        room_2d_points = np.stack(
            [np.asarray(room_pcd.points)[:, 0], np.asarray(room_pcd.points)[:, 2]],
            axis=1,
        )
        plt.scatter(room_2d_points[:, 0], room_2d_points[:, 1], s=0.1, c=cmap(room_idx))
        flattened_room_points.append(room_2d_points)

    pose_cmap = cm.get_cmap("Set1")
    pbar = tqdm(enumerate(pose_list), total=len(pose_list), desc="assign camera to room")
    for i, pose in pbar:
        pos = pose[0, 3], pose[2, 3]
        z = pose[1, 3]
        if z < pcd_min[1] or z > pcd_max[1]:
            img2room_id.append(-1)
            continue
        room_dists = []
        for room_points in flattened_room_points:
            room_dists.append(
                np.min(
                    distance.cdist(
                        np.array([pos]), np.array(room_points), metric="euclidean"
                    )
                )
            )
        closest_room_idx = np.argmin(room_dists)
        plt.scatter(pos[0], pos[1], s=3.0, c=pose_cmap(closest_room_idx))
        img2room_id.append(closest_room_idx)
        room_id2img_id[closest_room_idx].append(i)

    for room_id in range(len(flattened_room_points)):
        if room_id in room_id2img_id:
            continue
        closest_cam_pose = []
        pbar = tqdm(
            enumerate(pose_list),
            total=len(pose_list),
            desc="find closest camera pose to room w/o assigned image",
        )
        for _, pose in pbar:
            pos = pose[0, 3], pose[2, 3]
            z = pose[1, 3]
            if z < pcd_min[1] or z > pcd_max[1]:
                closest_cam_pose.append(
                    np.min(
                        distance.cdist(
                            np.array([pos]),
                            np.array(flattened_room_points[room_id]),
                            metric="euclidean",
                        )
                    )
                )
            else:
                closest_cam_pose.append(np.inf)
        closest_cam_pose_idx = np.argmin(np.array(closest_cam_pose))
        room_id2img_id[room_id].append(closest_cam_pose_idx)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "pcd_camera_pose.png"))
    plt.close()

    repr_img_ids_list = []
    repr_embs_list = []
    for room_id in range(len(flattened_room_points)):
        img_ids = room_id2img_id[room_id]
        if len(img_ids) == 0:
            repr_img_ids_list.append([])
            repr_embs_list.append([])
            continue
        room_clip_embeddings = [emb_list[i] for i in img_ids]
        room_clip_embeddings = np.squeeze(np.array(room_clip_embeddings), axis=1)
        if len(img_ids) < num_views:
            repr_img_ids_list.append(img_ids)
            repr_embs_list.append([emb for emb in room_clip_embeddings])
            continue
        kmeans = KMeans(n_clusters=num_views, max_iter=100, n_init=5, random_state=0).fit(
            room_clip_embeddings
        )
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        repr_img_ids = []
        repr_embs = []
        for unique_label in np.unique(labels):
            ids = np.where(labels == unique_label)[0]
            cluster = room_clip_embeddings[ids]
            mean_feats = centers[unique_label]
            similarity = np.dot(cluster, mean_feats)
            max_idx = np.argmax(similarity)
            repr_img_ids.append(img_ids[ids[max_idx]])
            repr_embs.append(cluster[max_idx])
        repr_img_ids_list.append(repr_img_ids)
        repr_embs_list.append(repr_embs)
    return repr_embs_list, repr_img_ids_list


def map_grid_to_point_cloud(occupancy_grid_map, resolution, point_cloud):
    occupancy_grid_map = (occupancy_grid_map > 0).astype(np.uint8) # 마스크를 0/1로 정리하는 줄 (흰 영역은 1, 나머지는 0)
    y_cells, x_cells = np.where(occupancy_grid_map == 1) # 한 흰색 픽셀 위치 찾기 (행, 열)

    # -10.5는 
    #   -- copyMakeBorder(..., 10, 10, 10, 10)으로 추가한 10 픽셀 border 되돌리는 보정
    #   -- 0.5는 픽셀의 왼쪽 위 모서리가 ㅇ니라 칸 중심 쪽으로 맞추려는 보정
    # - resolution
    #   -- 칸 번호를 미터 단위 거리로 바꿈
    # - np.min(point_cloud[: ,0])
    #   -- 이 grid가 시작하는 실제 월드 x 원점을 더해주기 
    mapped_x_coords = (x_cells - 10.5) * resolution + np.min(point_cloud[:, 0]) 
    mapped_y_coords = (y_cells - 10.5) * resolution + np.min(point_cloud[:, 1])
    return np.column_stack((mapped_x_coords, mapped_y_coords))


def distance_transform(occupancy_map, resolution, tmp_path):
    """
    occupancy_map: 최종 장애물 맵
    resolution: 한 픽셀이 몇 m인지
    tmp_path: 디버그 이미지 저장 경로
    """

    print("occupancy_map shape: ", occupancy_map.shape) # ex: (480, 220)이면 세로 480칸 가로 220칸짜리 2D 지도
    bw = occupancy_map.copy() # distance transform 계산용
    full_map = occupancy_map.copy() # 나중에 watershed용

    # 흑백 반전(반전 후, 흰색: 자유공간, 검정: 장애물) 
    # -> cv2.distanceTransform()은 0이 아닌 픽셀(흰색) 안에서 거리 계산을 하기 때문
    bw = cv2.bitwise_not(bw) 
    bw = np.uint8(bw) # OpenCV가 잘 처리하게 자료형을 uint8로 맞춤

    # 자유공간의 각 픽셀마다 가장 가까운 장애물까지 거리 계산
    #   - DIST_L2라서 유클리드 거리
    #   - 벽 바로 옆은 값이 작고, 방 한가운데는 값이 큼
    #       -- 방 중심은 밝고, 벽 근처는 어두움
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) 
    print("range of dist: ", np.min(dist), np.max(dist))
    cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX) # 거리값을 0~255로 스케일 조정 (stretch 적용)
    plt.figure()
    plt.imshow(dist, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist.png")) # 장애물에서 멀수록 밝게 보이는 지도
    plt.close()

    # ------------------------------------- Otsu 파트 수정 -------------------------------------
    dist = np.uint8(dist) # OpenCV에서 잘 다룰 수 있게 8비트 정수로 변환
    blur = cv2.GaussianBlur(dist, (11, 1), 10) # (11, 1)이라 가로 방향으로만 많이 blur
    plt.figure()
    plt.imshow(blur, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist_blur.png")) # dist.png보다 부드럽고 퍼진 버전
    plt.close()

    # - 여기서부터 기존로직이랑 살짝 다름
    # 1. 계산을 위해 0~255 값을 0.0~1.0 사이의 소수로 만듦
    blur_float = blur.astype(np.float32) / 255.0

    # 2. 루트를 씌워서 작은 값을 확 끌어올리기
    gamma_corrected = np.power(blur_float, 0.5) * 255.0

    # 3. 다시 OpenCV가 좋아하는 8비트 정수로 되돌림
    blur_gamma = np.uint8(gamma_corrected)

    # 4. 점수가 조작된 blur_gamma를 Otsu 알고리즘 돌리기
    _, dist = cv2.threshold(blur_gamma, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # --------------------------------------------------------------------------------------

    plt.figure()
    plt.imshow(dist, cmap="jet", origin="lower")
    # 결과는 binary 이미지
    plt.savefig(os.path.join(tmp_path, "dist_thresh.png")) # dist_blur.png에서 threshold보다 큰 애들만 살아남은 버전
    plt.close()

    # seed blob들의 외곽(contour) 찾기
    # 각 contour 하나가 방 seed 하나
    dist_8u = dist.astype("uint8")
    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("number of seeds, aka rooms: ", len(contours))
    for i, contour in enumerate(contours):
        print(f"area of seed {i}: ", cv2.contourArea(contour))

    # 너무 작은 seed는 잡음으로 제거하려고 최소 면적을 정함
    min_area_m = 0.5
    min_area = (min_area_m / resolution) ** 2 # 0.5m x 0.5m보다 작은 영역은 버리겠다는 의미
    print("min_area: ", min_area)
    contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    print("number of contours after remove small seeds: ", len(contours))

    # watershed 알고리즘을 적용할 label 맵 생성
    #   - 처음엔 전부 0
    markers = np.zeros(dist.shape, dtype=np.int32)
    # 각 seed 영역을 서로 다른 번호로 칠함
    # ex)
    #   - 첫 번째 room seed: 1
    #   - 두 번째: 2
    #   - 세 번째: 3
    for i, contour in enumerate(contours):
        cv2.drawContours(markers, contours, i, (i + 1), -1)
    # 모서리 근처에 배경 marker 하나 추가 (watershed가 바깥쪽도 하나의 기준 영역으로 알게 하려는 용도)
    cv2.circle(markers, (3, 3), 1, len(contours) + 1, -1) 

    full_map = cv2.cvtColor(full_map, cv2.COLOR_GRAY2BGR) # OpenCV watershed는 3채널 이미지를 기대해서 BGR로 변환
    # watershed 알고리즘 적용
    #   - seed들에서부터 영역을 퍼뜨리면서 분할
    #   - 서로 만나는 경계에서 멈춤
    #   - 실행 후 markers 안 값이 바뀜
    #       -- 각 방 영역은 방 번호
    #       -- 경계는 보통 -1
    cv2.watershed(full_map, markers) 

    plt.figure()
    plt.imshow(markers, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "markers.png"))
    plt.close()

    # room별 좌표 묶음을 정리해서 반환
    room_vertices = [np.where(markers == i + 1) for i in range(len(contours))]
    room_vertices = np.array(room_vertices, dtype=object).squeeze()
    print("room_vertices shape: ", room_vertices.shape)
    return room_vertices

def find_overlapping_ratio_faiss(pcd1, pcd2, radius=0.02):
    if isinstance(pcd1, o3d.geometry.PointCloud):
        pcd1 = np.asarray(pcd1.points)
    if isinstance(pcd2, o3d.geometry.PointCloud):
        pcd2 = np.asarray(pcd2.points)
    if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
        return 0
    index1 = faiss.IndexFlatL2(pcd1.shape[1])
    index2 = faiss.IndexFlatL2(pcd2.shape[1])
    index1.add(pcd1.astype(np.float32))
    index2.add(pcd2.astype(np.float32))
    d1, _ = index2.search(pcd1.astype(np.float32), k=1)
    d2, _ = index1.search(pcd2.astype(np.float32), k=1)
    overlap1 = np.sum(d1 < radius**2)
    overlap2 = np.sum(d2 < radius**2)
    return max(overlap1 / pcd1.shape[0], overlap2 / pcd2.shape[0])


def feats_denoise_dbscan(feats, eps=0.02, min_points=2):
    feats = np.array(feats)
    clustering = DBSCAN(eps=eps, min_samples=min_points, metric="cosine").fit(feats)
    labels = clustering.labels_
    counter = Counter(labels)
    if -1 in counter:
        del counter[-1]
    if counter:
        most_common_label, _ = counter.most_common(1)[0]
        largest_mask = labels == most_common_label
        feats = feats[largest_mask]
        if len(feats) > 1:
            feats = np.mean(feats, axis=0)
    else:
        feats = np.mean(feats, axis=0)
    return feats


def pcd_denoise_dbscan(pcd, eps=0.02, min_points=10):
    pcd_clusters = pcd.cluster_dbscan(eps=eps, min_points=min_points)
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)
    counter = Counter(pcd_clusters)
    if -1 in counter:
        del counter[-1]
    if not counter:
        return pcd
    most_common_label, _ = counter.most_common(1)[0]
    largest_mask = pcd_clusters == most_common_label
    largest_cluster_points = obj_points[largest_mask]
    largest_cluster_colors = obj_colors[largest_mask]
    if len(largest_cluster_points) < 5:
        return pcd
    largest_cluster_pcd = o3d.geometry.PointCloud()
    largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
    largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
    return largest_cluster_pcd


def merge_room_objects(room, overlap_threshold=0.01, radius=0.1):
    overlap_scores = np.zeros((len(room.objects), len(room.objects)))
    for i, obj1 in enumerate(room.objects):
        for j, obj2 in enumerate(room.objects):
            if i >= j:
                continue
            if obj1.name != obj2.name:
                continue
            overlap = find_overlapping_ratio_faiss(obj1.pcd, obj2.pcd, radius)
            if overlap > overlap_threshold:
                overlap_scores[i, j] = overlap
                overlap_scores[j, i] = overlap

    new_room_objects = defaultdict(list)
    merging_indices = []
    i_indices, j_indices = np.where(overlap_scores > 0)
    for i, j in zip(i_indices, j_indices):
        merging_indices.extend([i, j])
        if i not in new_room_objects and j not in new_room_objects:
            new_room_objects[i].append(j)
        elif i in new_room_objects:
            new_room_objects[i].append(j)
        else:
            new_room_objects[j].append(i)

    merging_indices = list(set(merging_indices))
    for idx in range(len(room.objects)):
        if idx not in merging_indices:
            new_room_objects[idx].append(idx)

    merged_objects = []
    object_index = 0
    for i, group in new_room_objects.items():
        group = list(set(group))
        if len(group) == 1:
            jj = group[0]
            if i == jj:
                obj = room.objects[i]
                obj.object_id = f"{room.room_id}_{object_index}"
                merged_objects.append(obj)
                object_index += 1
            else:
                obj = room.objects[i]
                obj += room.objects[jj]
                obj.object_id = f"{room.room_id}_{object_index}"
                merged_objects.append(obj)
                object_index += 1
        else:
            obj = room.objects[i]
            for jj in group:
                obj += room.objects[jj]
            obj.object_id = f"{room.room_id}_{object_index}"
            merged_objects.append(obj)
            object_index += 1
    room.objects = merged_objects
