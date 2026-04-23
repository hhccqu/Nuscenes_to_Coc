"""Access wrappers around nuScenes and map expansion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

from .constants import CAMERA_NAMES, DEFAULT_FRONT_DISTANCE_M, DEFAULT_FRONT_LATERAL_RANGE_M
from .geometry import (
    estimate_polyline_curvature,
    quaternion_to_yaw,
    signed_lateral_offset,
    transform_world_to_ego,
)
from .motion import compute_acceleration, compute_speed, compute_yaw_rate


if "seaborn-v0_8-whitegrid" not in plt.style.available and "seaborn-whitegrid" in plt.style.available:
    _original_style_use = plt.style.use

    def _patched_style_use(style):
        if style == "seaborn-v0_8-whitegrid":
            style = "seaborn-whitegrid"
        return _original_style_use(style)

    plt.style.use = _patched_style_use

from nuscenes.map_expansion.map_api import NuScenesMap  # noqa: E402


@dataclass
class LaneInfo:
    lane_token: str
    is_lane_connector: bool
    lane_centerline: List[List[float]]
    lane_width_estimate: Optional[float]
    lane_count_nearby: int
    distance_to_center: float
    signed_offset: float
    heading_error: float
    curvature: float


class NuScenesCOCAccess:
    """Unified data access for the CoC pipeline."""

    def __init__(self, dataroot: str, version: str) -> None:
        self.dataroot = dataroot
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self._map_cache: Dict[str, NuScenesMap] = {}
        self._scene_location_cache: Dict[str, str] = {}
        self._lane_connector_token_cache: Dict[str, set] = {}
        self._sample_pose_cache: Dict[str, Dict] = {}
        self._sample_state_cache: Dict[str, Dict] = {}
        self._sample_lane_cache: Dict[str, Optional[LaneInfo]] = {}
        self._objects_in_front_cache: Dict[str, List[Dict]] = {}
        self._static_control_cache: Dict[str, Dict] = {}

    def get_scene_sample_sequence(self, scene_token: str) -> List[Dict]:
        scene = self.nusc.get("scene", scene_token)
        samples: List[Dict] = []
        token = scene["first_sample_token"]
        while token:
            sample = self.nusc.get("sample", token)
            samples.append(sample)
            if token == scene["last_sample_token"]:
                break
            token = sample["next"]
        return samples

    def get_sample_camera_paths(self, sample_token: str) -> Dict[str, str]:
        sample = self.nusc.get("sample", sample_token)
        output: Dict[str, str] = {}
        for cam_name in CAMERA_NAMES:
            sd = self.nusc.get("sample_data", sample["data"][cam_name])
            output[cam_name] = os.path.join(self.dataroot, sd["filename"])
        return output

    def get_sample_ego_pose(self, sample_token: str) -> Dict:
        if sample_token in self._sample_pose_cache:
            return self._sample_pose_cache[sample_token]
        sample = self.nusc.get("sample", sample_token)
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        pose = {
            "translation": ego_pose["translation"],
            "rotation": ego_pose["rotation"],
            "yaw": quaternion_to_yaw(ego_pose["rotation"]),
            "timestamp": sample["timestamp"],
        }
        self._sample_pose_cache[sample_token] = pose
        return pose

    def get_ego_state(self, sample_token: str) -> Dict:
        if sample_token in self._sample_state_cache:
            return self._sample_state_cache[sample_token]
        sample = self.nusc.get("sample", sample_token)
        pose = self.get_sample_ego_pose(sample_token)

        prev_pose = self.get_sample_ego_pose(sample["prev"]) if sample["prev"] else None
        next_pose = self.get_sample_ego_pose(sample["next"]) if sample["next"] else None

        speed_prev = compute_speed(prev_pose, pose) if prev_pose else None
        speed_next = compute_speed(pose, next_pose) if next_pose else None
        if speed_prev is not None and speed_next is not None:
            speed = (speed_prev + speed_next) / 2.0
        else:
            speed = speed_prev if speed_prev is not None else speed_next

        dt_prev = (pose["timestamp"] - prev_pose["timestamp"]) / 1e6 if prev_pose else None
        dt_next = (next_pose["timestamp"] - pose["timestamp"]) / 1e6 if next_pose else None
        acc = None
        if speed_prev is not None and speed_next is not None and dt_prev is not None and dt_next is not None:
            acc = compute_acceleration(speed_prev, speed_next, (dt_prev + dt_next) / 2.0)
        yaw_rate = None
        if prev_pose and next_pose:
            dt = (next_pose["timestamp"] - prev_pose["timestamp"]) / 1e6
            yaw_rate = compute_yaw_rate(prev_pose["yaw"], next_pose["yaw"], dt)

        state = {
            "speed": speed,
            "acceleration": acc,
            "translation": pose["translation"],
            "rotation": pose["rotation"],
            "yaw": pose["yaw"],
            "yaw_rate": yaw_rate,
            "timestamp": pose["timestamp"],
        }
        self._sample_state_cache[sample_token] = state
        return state

    def get_scene_map(self, scene_token: str) -> NuScenesMap:
        if scene_token not in self._scene_location_cache:
            scene = self.nusc.get("scene", scene_token)
            log = self.nusc.get("log", scene["log_token"])
            self._scene_location_cache[scene_token] = log["location"]
        location = self._scene_location_cache[scene_token]
        if location not in self._map_cache:
            nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=location)
            self._map_cache[location] = nusc_map
            self._lane_connector_token_cache[location] = {record["token"] for record in nusc_map.lane_connector}
        return self._map_cache[location]

    def get_ego_lane_info(self, sample_token: str) -> Optional[LaneInfo]:
        if sample_token in self._sample_lane_cache:
            return self._sample_lane_cache[sample_token]

        sample = self.nusc.get("sample", sample_token)
        pose = self.get_sample_ego_pose(sample_token)
        nusc_map = self.get_scene_map(sample["scene_token"])
        lane_token = nusc_map.get_closest_lane(pose["translation"][0], pose["translation"][1], radius=5)
        if not lane_token:
            self._sample_lane_cache[sample_token] = None
            return None

        centerlines = nusc_map.discretize_lanes([lane_token], 0.5)
        centerline = centerlines.get(lane_token, [])
        if not centerline:
            self._sample_lane_cache[sample_token] = None
            return None

        lateral_offset, center_heading = signed_lateral_offset(centerline, (pose["translation"][0], pose["translation"][1]))
        heading_error = pose["yaw"] - center_heading
        lane_records = nusc_map.get_records_in_radius(
            pose["translation"][0],
            pose["translation"][1],
            10.0,
            ["lane", "lane_connector"],
        )
        nearby_count = len(lane_records["lane"]) + len(lane_records["lane_connector"])
        curvature = estimate_polyline_curvature(centerline)
        lane_info = LaneInfo(
            lane_token=lane_token,
            is_lane_connector=lane_token in self._lane_connector_token_cache[nusc_map.map_name],
            lane_centerline=[[float(v) for v in item] for item in centerline],
            lane_width_estimate=None,
            lane_count_nearby=nearby_count,
            distance_to_center=abs(lateral_offset),
            signed_offset=float(lateral_offset),
            heading_error=float(heading_error),
            curvature=float(curvature),
        )
        self._sample_lane_cache[sample_token] = lane_info
        return lane_info

    def get_ego_lateral_offset(self, sample_token: str) -> Optional[float]:
        lane_info = self.get_ego_lane_info(sample_token)
        if lane_info is None:
            return None
        pose = self.get_sample_ego_pose(sample_token)
        lateral_offset, _ = signed_lateral_offset(
            lane_info.lane_centerline,
            (pose["translation"][0], pose["translation"][1]),
        )
        return lateral_offset

    def get_objects_in_front(
        self,
        sample_token: str,
        distance: float = DEFAULT_FRONT_DISTANCE_M,
        lateral_range: float = DEFAULT_FRONT_LATERAL_RANGE_M,
    ) -> List[Dict]:
        if sample_token in self._objects_in_front_cache:
            return self._objects_in_front_cache[sample_token]
        sample = self.nusc.get("sample", sample_token)
        pose = self.get_sample_ego_pose(sample_token)
        ego_lane = self.get_ego_lane_info(sample_token)
        nusc_map = self.get_scene_map(sample["scene_token"])
        output: List[Dict] = []

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            rel_xyz = transform_world_to_ego(pose["translation"], pose["rotation"], ann["translation"])
            x, y, z = float(rel_xyz[0]), float(rel_xyz[1]), float(rel_xyz[2])
            if x <= 0 or x > distance or abs(y) > lateral_range:
                continue
            obj_lane_token = None
            same_lane = False
            if ego_lane is not None:
                try:
                    obj_lane_token = nusc_map.get_closest_lane(
                        ann["translation"][0],
                        ann["translation"][1],
                        radius=5,
                    )
                except Exception:
                    obj_lane_token = None
                same_lane = obj_lane_token == ego_lane.lane_token if obj_lane_token else False
            velocity = self.nusc.box_velocity(ann_token)
            output.append(
                {
                    "annotation_token": ann_token,
                    "instance_token": ann["instance_token"],
                    "category_name": ann["category_name"],
                    "relative_pose": {"x": x, "y": y, "z": z},
                    "distance": float((x * x + y * y + z * z) ** 0.5),
                    "box_size": ann["size"],
                    "velocity": None if velocity is None else [float(v) for v in velocity[:2]],
                    "same_lane": same_lane,
                    "lane_token": obj_lane_token,
                }
            )
        output.sort(key=lambda item: item["distance"])
        self._objects_in_front_cache[sample_token] = output
        return output

    def get_static_control_nearby(self, sample_token: str, distance: float = 35.0) -> Dict:
        if sample_token in self._static_control_cache:
            return self._static_control_cache[sample_token]
        sample = self.nusc.get("sample", sample_token)
        pose = self.get_sample_ego_pose(sample_token)
        nusc_map = self.get_scene_map(sample["scene_token"])
        records = nusc_map.get_records_in_radius(
            pose["translation"][0],
            pose["translation"][1],
            distance,
            ["stop_line", "traffic_light"],
        )
        stop_lines = records.get("stop_line", [])
        traffic_lights = records.get("traffic_light", [])
        min_dist = None
        stop_line_type = None
        for token in stop_lines:
            record = nusc_map.get("stop_line", token)
            bounds = nusc_map.get_bounds("stop_line", token)
            cx = (bounds[0] + bounds[2]) / 2.0
            cy = (bounds[1] + bounds[3]) / 2.0
            dist = ((cx - pose["translation"][0]) ** 2 + (cy - pose["translation"][1]) ** 2) ** 0.5
            if min_dist is None or dist < min_dist:
                min_dist = float(dist)
                stop_line_type = record["stop_line_type"]
        result = {
            "has_stop_line": bool(stop_lines),
            "stop_line_type": stop_line_type,
            "distance_to_stop_line": min_dist,
            "has_traffic_light_geometry": bool(traffic_lights),
            "traffic_light_status": "unknown",
        }
        self._static_control_cache[sample_token] = result
        return result
