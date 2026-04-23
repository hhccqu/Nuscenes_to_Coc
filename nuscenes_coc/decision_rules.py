"""Rule-based driving decision generation."""

from __future__ import annotations

from typing import Dict, List

from .constants import DYNAMIC_VRU_CATEGORIES, VEHICLE_CATEGORIES


def decide_longitudinal(key_info: Dict, future_infos: List[Dict]) -> Dict:
    """Return a longitudinal decision and evidence."""
    evidence: List[str] = []
    key_state = key_info["ego_state"]
    key_objects = key_info["objects_in_front"]
    key_controls = key_info["static_controls"]
    key_lane = key_info.get("lane_info")

    future_speeds = [info["ego_state"]["speed"] for info in future_infos if info["ego_state"]["speed"] is not None]
    future_min_speed = min(future_speeds) if future_speeds else None
    future_avg_speed = sum(future_speeds) / len(future_speeds) if future_speeds else None

    if key_controls["has_stop_line"] and future_min_speed is not None:
        key_speed = key_state["speed"]
        strong_stop = future_min_speed < 0.5
        notable_slowdown = (
            key_speed is not None
            and future_avg_speed is not None
            and key_speed > 1.5
            and (key_speed - future_avg_speed) > 1.5
        )
        if strong_stop or notable_slowdown:
            if strong_stop:
                evidence.append("future_speed_below_0.5")
            if notable_slowdown:
                evidence.append("future_speed_drop")
            evidence.append(f"stop_line:{key_controls['stop_line_type']}")
            confidence = "high" if strong_stop else "medium"
            return {"decision": "stop_static_constraint", "confidence": confidence, "evidence": evidence}

    vru_candidates = [obj for obj in key_objects if obj["category_name"] in DYNAMIC_VRU_CATEGORIES and obj["distance"] < 15.0]
    if vru_candidates and key_state["acceleration"] is not None and key_state["acceleration"] < -1.0:
        evidence.append("near_vru")
        evidence.append("negative_acceleration")
        return {"decision": "yield_agent_right_of_way", "confidence": "medium", "evidence": evidence}

    lead_vehicles = [
        obj
        for obj in key_objects
        if obj["category_name"] in VEHICLE_CATEGORIES
        and obj["same_lane"]
        and abs(obj["relative_pose"]["y"]) < 5.5
        and obj["distance"] < 35.0
    ]
    if lead_vehicles:
        lead = lead_vehicles[0]
        evidence.append(f"lead_vehicle:{lead['category_name']}")
        evidence.append(f"lead_distance:{lead['distance']:.2f}")
        evidence.append(f"lead_lateral:{lead['relative_pose']['y']:.2f}")
        return {"decision": "lead_obstacle_following", "confidence": "high", "evidence": evidence}

    if key_lane is not None and key_lane.curvature > 0.08 and key_state["acceleration"] is not None and key_state["acceleration"] < -1.0:
        evidence.append(f"curve_curvature:{key_lane.curvature:.3f}")
        evidence.append("negative_acceleration")
        return {"decision": "speed_adaptation_road", "confidence": "medium", "evidence": evidence}

    if key_state["acceleration"] is not None and abs(key_state["acceleration"]) < 1.0:
        evidence.append("stable_acceleration")
        return {"decision": "set_speed_tracking", "confidence": "high", "evidence": evidence}

    return {"decision": "none", "confidence": "low", "evidence": ["no_longitudinal_rule"]}


def decide_lateral(key_info: Dict, future_infos: List[Dict]) -> Dict:
    """Return a lateral decision and evidence."""
    evidence: List[str] = []
    lane_info = key_info.get("lane_info")
    offset = key_info.get("lateral_offset")
    future_lane_tokens = [info["lane_info"].lane_token for info in future_infos if info.get("lane_info") is not None]
    future_yaws = [info["ego_state"]["yaw"] for info in future_infos]
    key_yaw = key_info["ego_state"]["yaw"]

    if lane_info is not None:
        changed_future_infos = [
            info for info in future_infos if info.get("lane_info") is not None and info["lane_info"].lane_token != lane_info.lane_token
        ]
        if changed_future_infos:
            target_lane = changed_future_infos[-1]["lane_info"]
            offset_values = [info.get("lateral_offset") for info in future_infos if info.get("lateral_offset") is not None]
            offset_delta = None
            if offset is not None and offset_values:
                offset_delta = offset_values[-1] - offset

            if (
                not lane_info.is_lane_connector
                and not target_lane.is_lane_connector
                and offset_delta is not None
                and abs(offset_delta) > 0.6
            ):
                evidence.append(f"lane_change:{lane_info.lane_token}->{target_lane.lane_token}")
                evidence.append(f"offset_delta:{offset_delta:.2f}")
                if offset_delta > 0:
                    return {"decision": "lane_change_left", "confidence": "medium", "evidence": evidence}
                return {"decision": "lane_change_right", "confidence": "medium", "evidence": evidence}

        if lane_info.is_lane_connector and future_yaws:
            yaw_delta = abs(future_yaws[-1] - key_yaw)
            if yaw_delta > 0.35:
                evidence.append("lane_connector")
                evidence.append(f"yaw_delta:{yaw_delta:.2f}")
                return {"decision": "turn_intersection", "confidence": "medium", "evidence": evidence}

    if offset is not None:
        if 0.3 < offset < 1.0:
            evidence.append(f"offset:{offset:.2f}")
            return {"decision": "in_lane_nudge_left", "confidence": "medium", "evidence": evidence}
        if -1.0 < offset < -0.3:
            evidence.append(f"offset:{offset:.2f}")
            return {"decision": "in_lane_nudge_right", "confidence": "medium", "evidence": evidence}
        if abs(offset) < 0.3:
            evidence.append(f"offset:{offset:.2f}")
            return {"decision": "lane_keeping_centering", "confidence": "high", "evidence": evidence}

    return {"decision": "none", "confidence": "low", "evidence": ["no_lateral_rule"]}
