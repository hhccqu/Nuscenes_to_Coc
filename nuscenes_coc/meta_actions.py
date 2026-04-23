"""Rule-based atomic meta action labeling."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List


def infer_longitudinal_meta_action(ego_state: Dict) -> str:
    """Infer a frame-level longitudinal meta action."""
    speed = ego_state.get("speed")
    acceleration = ego_state.get("acceleration")

    if speed is None:
        return "maintain_speed"
    if speed < -0.2:
        return "reverse"
    if abs(speed) < 0.3:
        return "stop"
    if acceleration is None:
        return "maintain_speed"
    if acceleration > 1.5:
        return "strong_accelerate"
    if acceleration > 0.3:
        return "gentle_accelerate"
    if acceleration < -1.5:
        return "strong_decelerate"
    if acceleration < -0.3:
        return "gentle_decelerate"
    return "maintain_speed"


def infer_lateral_meta_action(ego_state: Dict) -> str:
    """Infer a frame-level lateral meta action from yaw rate."""
    yaw_rate = ego_state.get("yaw_rate")
    speed = ego_state.get("speed")
    if yaw_rate is None:
        return "go_straight"
    if speed is not None and speed < -0.2:
        if yaw_rate > 0.15:
            return "reverse_left"
        if yaw_rate < -0.15:
            return "reverse_right"
        return "go_straight"
    if yaw_rate > 0.25:
        return "sharp_steer_left"
    if yaw_rate > 0.05:
        return "steer_left"
    if yaw_rate < -0.25:
        return "sharp_steer_right"
    if yaw_rate < -0.05:
        return "steer_right"
    return "go_straight"


def infer_meta_actions(ego_state: Dict) -> Dict:
    """Infer both longitudinal and lateral atomic actions."""
    return {
        "longitudinal": infer_longitudinal_meta_action(ego_state),
        "lateral": infer_lateral_meta_action(ego_state),
    }


def summarize_meta_actions(states: List[Dict]) -> Dict:
    """Summarize atomic actions over a time window."""
    actions = [infer_meta_actions(state) for state in states]
    longitudinal_counter = Counter(action["longitudinal"] for action in actions)
    lateral_counter = Counter(action["lateral"] for action in actions)
    return {
        "per_frame": actions,
        "longitudinal_counts": dict(longitudinal_counter),
        "lateral_counts": dict(lateral_counter),
        "dominant_longitudinal": longitudinal_counter.most_common(1)[0][0] if longitudinal_counter else None,
        "dominant_lateral": lateral_counter.most_common(1)[0][0] if lateral_counter else None,
    }
