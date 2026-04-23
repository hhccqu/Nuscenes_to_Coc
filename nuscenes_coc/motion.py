"""Motion estimation helpers."""

from __future__ import annotations

from typing import Dict, Optional

from .geometry import wrap_angle


def compute_speed(prev_pose: Dict, next_pose: Dict) -> Optional[float]:
    """Estimate speed from two poses."""
    dt = (next_pose["timestamp"] - prev_pose["timestamp"]) / 1e6
    if dt <= 1e-6:
        return None
    dx = next_pose["translation"][0] - prev_pose["translation"][0]
    dy = next_pose["translation"][1] - prev_pose["translation"][1]
    dz = next_pose["translation"][2] - prev_pose["translation"][2]
    return float((dx * dx + dy * dy + dz * dz) ** 0.5 / dt)


def compute_acceleration(prev_speed: Optional[float], next_speed: Optional[float], dt: float) -> Optional[float]:
    """Estimate acceleration from two speed values."""
    if prev_speed is None or next_speed is None or dt <= 1e-6:
        return None
    return float((next_speed - prev_speed) / dt)


def compute_yaw_rate(prev_yaw: float, next_yaw: float, dt: float) -> Optional[float]:
    """Estimate yaw rate from two yaws."""
    if dt <= 1e-6:
        return None
    return float(wrap_angle(next_yaw - prev_yaw) / dt)
