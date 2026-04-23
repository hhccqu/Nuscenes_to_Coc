"""Candidate segment filtering."""

from __future__ import annotations

from typing import Dict, List


def detect_action_indices(sample_infos: List[Dict]) -> List[Dict]:
    """Detect behavior-change points from precomputed per-sample info."""
    events: List[Dict] = []
    for idx, info in enumerate(sample_infos):
        state = info["ego_state"]
        lat_offset = info.get("lateral_offset")

        longitudinal = False
        reasons = []
        acc = state.get("acceleration")
        speed = state.get("speed")
        if acc is not None and abs(acc) > 1.5:
            longitudinal = True
            reasons.append(f"abs_acc>{abs(acc):.2f}")
        if idx >= 1:
            prev_speed = sample_infos[idx - 1]["ego_state"].get("speed")
            if prev_speed is not None and speed is not None and prev_speed > 2.0 and speed < 0.5:
                longitudinal = True
                reasons.append("moving_to_stop")

        lateral = False
        if idx >= 2:
            prev_prev_offset = sample_infos[idx - 2].get("lateral_offset")
            if prev_prev_offset is not None and lat_offset is not None and abs(lat_offset - prev_prev_offset) > 0.3:
                lateral = True
                reasons.append("lateral_offset_change")
        yaw_rate = state.get("yaw_rate")
        if yaw_rate is not None and abs(yaw_rate) > 0.35:
            lateral = True
            reasons.append("yaw_rate_change")

        if longitudinal or lateral:
            events.append(
                {
                    "action_idx": idx,
                    "reasons": reasons,
                    "longitudinal": longitudinal,
                    "lateral": lateral,
                }
            )
    return merge_close_events(events)


def merge_close_events(events: List[Dict], min_gap: int = 3) -> List[Dict]:
    """Merge close action events."""
    if not events:
        return []
    merged = [events[0]]
    for event in events[1:]:
        if event["action_idx"] - merged[-1]["action_idx"] < min_gap:
            if len(event["reasons"]) >= len(merged[-1]["reasons"]):
                merged[-1] = event
        else:
            merged.append(event)
    return merged


def build_candidate_segments(sample_infos: List[Dict], history: int = 4, future: int = 5) -> List[Dict]:
    """Build candidate segments centered around action points."""
    events = detect_action_indices(sample_infos)
    segments: List[Dict] = []
    for event in events:
        action_idx = event["action_idx"]
        if action_idx == 0:
            continue
        start = max(0, action_idx - history)
        end = min(len(sample_infos), action_idx + future + 1)
        indices = list(range(start, end))
        keyframe_idx = action_idx - 1
        if keyframe_idx not in indices:
            continue
        segments.append(
            {
                "sample_indices": indices,
                "action_idx": action_idx,
                "keyframe_idx": keyframe_idx,
                "reasons": event["reasons"],
            }
        )
    return segments
