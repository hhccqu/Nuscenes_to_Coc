"""Geometry helpers."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from pyquaternion import Quaternion


def quaternion_to_yaw(rotation: Sequence[float]) -> float:
    """Return yaw in radians from a wxyz quaternion."""
    rot = Quaternion(rotation).rotation_matrix
    return math.atan2(rot[1, 0], rot[0, 0])


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def transform_world_to_ego(
    ego_translation: Sequence[float],
    ego_rotation: Sequence[float],
    point_world: Sequence[float],
) -> np.ndarray:
    """Transform a world point into the ego frame."""
    ego_rot = Quaternion(ego_rotation).rotation_matrix
    rel = np.asarray(point_world, dtype=np.float64) - np.asarray(ego_translation, dtype=np.float64)
    return ego_rot.T @ rel


def nearest_point_index(points: Iterable[Sequence[float]], x: float, y: float) -> int:
    """Return index of the nearest xy point."""
    pts = np.asarray([[p[0], p[1]] for p in points], dtype=np.float64)
    dists = np.linalg.norm(pts - np.asarray([x, y], dtype=np.float64), axis=1)
    return int(np.argmin(dists))


def polyline_length(points: List[Sequence[float]]) -> float:
    """Compute cumulative length of a polyline."""
    if len(points) < 2:
        return 0.0
    pts = np.asarray([[p[0], p[1]] for p in points], dtype=np.float64)
    diffs = pts[1:] - pts[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())


def estimate_polyline_curvature(points: List[Sequence[float]]) -> float:
    """Estimate maximum curvature from a discretized centerline."""
    if len(points) < 3:
        return 0.0
    pts = np.asarray([[p[0], p[1]] for p in points], dtype=np.float64)
    diffs = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(diffs, axis=1)
    valid = seg_len > 1e-6
    if valid.sum() < 2:
        return 0.0
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    values: List[float] = []
    for idx in range(1, len(headings)):
        ds = float(seg_len[idx - 1] + seg_len[idx]) / 2.0 if idx < len(seg_len) else float(seg_len[idx - 1])
        if ds <= 1e-6:
            continue
        values.append(abs(wrap_angle(float(headings[idx] - headings[idx - 1]))) / ds)
    return max(values) if values else 0.0


def signed_lateral_offset(
    centerline: List[Sequence[float]],
    point_xy: Tuple[float, float],
) -> Tuple[float, float]:
    """Return signed lateral offset and heading at nearest centerline point."""
    idx = nearest_point_index(centerline, point_xy[0], point_xy[1])
    ref = np.asarray(centerline[idx][:2], dtype=np.float64)
    if idx < len(centerline) - 1:
        nxt = np.asarray(centerline[idx + 1][:2], dtype=np.float64)
    elif idx > 0:
        nxt = np.asarray(centerline[idx - 1][:2], dtype=np.float64)
    else:
        nxt = ref + np.asarray([1.0, 0.0], dtype=np.float64)
    tangent = nxt - ref
    heading = math.atan2(tangent[1], tangent[0])
    normal_left = np.asarray([-math.sin(heading), math.cos(heading)], dtype=np.float64)
    delta = np.asarray(point_xy, dtype=np.float64) - ref
    return float(np.dot(delta, normal_left)), float(heading)
