"""Rule-based driving decision generation."""

from __future__ import annotations

from typing import Dict, List

from .constants import DYNAMIC_VRU_CATEGORIES, VEHICLE_CATEGORIES


def decide_longitudinal(key_info: Dict, future_infos: List[Dict]) -> Dict:
    """Return a longitudinal decision aligned with Alpamayo-R1 Table 1 (7 classes)."""
    evidence: List[str] = []
    key_state = key_info["ego_state"]
    key_objects = key_info["objects_in_front"]
    key_controls = key_info["static_controls"]
    key_lane = key_info.get("lane_info")

    future_speeds = [info["ego_state"]["speed"] for info in future_infos if info["ego_state"]["speed"] is not None]
    future_min_speed = min(future_speeds) if future_speeds else None
    future_avg_speed = sum(future_speeds) / len(future_speeds) if future_speeds else None
    key_speed = key_state.get("speed") or 0.0
    key_accel = key_state.get("acceleration")

    # ── 1. Stop for static constraints ──────────────────────────────────────
    # Decelerate to—and hold at—control points (stop/yield lines, red light).
    if key_controls["has_stop_line"] and future_min_speed is not None:
        strong_stop = future_min_speed < 0.5
        notable_slowdown = (
            key_speed > 1.5
            and future_avg_speed is not None
            and (key_speed - future_avg_speed) > 1.5
        )
        if strong_stop or notable_slowdown:
            evidence.append(f"stop_line:{key_controls['stop_line_type']}")
            if strong_stop:
                evidence.append("future_speed_below_0.5")
            if notable_slowdown:
                evidence.append("future_speed_drop")
            return {"decision": "stop_static_constraint",
                    "confidence": "high" if strong_stop else "medium",
                    "evidence": evidence}

    # ── 2. Yield agent right-of-way ─────────────────────────────────────────
    # Slow/stop to concede priority to pedestrians, cross-traffic, cut-ins.
    vru_close = [obj for obj in key_objects
                 if obj["category_name"] in DYNAMIC_VRU_CATEGORIES and obj["distance"] < 15.0]
    if vru_close and key_accel is not None and key_accel < -0.8:
        evidence.append(f"near_vru:{vru_close[0]['category_name']}")
        evidence.append(f"distance:{vru_close[0]['distance']:.1f}m")
        evidence.append("decelerating")
        return {"decision": "yield_agent_right_of_way", "confidence": "medium", "evidence": evidence}

    # ── 3. Lead obstacle following ──────────────────────────────────────────
    # Maintain safe time gap to the closest in-path entity in the same flow.
    # Excludes geometry-based slowing, gap-matching, or yielding to non-leads.
    lead_vehicles = [
        obj for obj in key_objects
        if obj["category_name"] in VEHICLE_CATEGORIES
        and obj["same_lane"]
        and abs(obj["relative_pose"]["y"]) < 5.5
        and obj["distance"] < 35.0
    ]
    if lead_vehicles:
        lead = lead_vehicles[0]
        evidence.append(f"lead:{lead['category_name']}@{lead['distance']:.1f}m")
        return {"decision": "lead_obstacle_following", "confidence": "high", "evidence": evidence}

    # ── 4. Speed adaptation (road events) ──────────────────────────────────
    # Adjust speed for roadway features: curves, grades, bumps, ramps, turns.
    # Independent of a lead vehicle.
    if key_lane is not None and key_lane.curvature > 0.08 and key_accel is not None and key_accel < -0.8:
        evidence.append(f"curve_curvature:{key_lane.curvature:.3f}")
        evidence.append("decelerating_for_geometry")
        return {"decision": "speed_adaptation_road", "confidence": "medium", "evidence": evidence}

    # ── 5. Gap-searching (for LC/merge/zipper) ──────────────────────────────
    # Adjust speed to match target stream or create a usable gap for a planned
    # lateral maneuver. Detected when ego is accelerating or decelerating
    # noticeably while a lateral change is pending (future lane token differs).
    future_lane_tokens = [
        info["lane_info"].lane_token for info in future_infos
        if info.get("lane_info") is not None
    ]
    current_token = key_lane.lane_token if key_lane is not None else None
    lane_change_pending = (
        current_token is not None
        and future_lane_tokens
        and future_lane_tokens[-1] != current_token
        and not (key_lane.is_lane_connector if key_lane else False)
    )
    if lane_change_pending and key_accel is not None and abs(key_accel) > 0.8:
        evidence.append("lane_change_pending")
        evidence.append(f"accel:{key_accel:.2f}")
        return {"decision": "gap_searching", "confidence": "medium", "evidence": evidence}

    # ── 6. Acceleration for passing/overtaking ──────────────────────────────
    # Increase speed to pass a slower lead with an associated lateral plan.
    # Detected when there is a lead vehicle ahead AND ego is accelerating.
    def _obj_speed(obj):
        v = obj.get("velocity")
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return (v[0] ** 2 + v[1] ** 2) ** 0.5
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    slow_lead = [
        obj for obj in key_objects
        if obj["category_name"] in VEHICLE_CATEGORIES
        and obj["same_lane"]
        and obj["distance"] < 40.0
        and _obj_speed(obj) < key_speed * 0.7  # lead is noticeably slower
    ]
    if slow_lead and key_accel is not None and key_accel > 1.0:
        evidence.append(f"slow_lead:{slow_lead[0]['category_name']}@{slow_lead[0]['distance']:.1f}m")
        evidence.append(f"ego_accel:{key_accel:.2f}")
        return {"decision": "acceleration_passing", "confidence": "medium", "evidence": evidence}

    # ── 7. Set speed tracking (default) ────────────────────────────────────
    # Maintain or reach a target speed when unconstrained; excludes
    # follow/yield/stop logic.
    evidence.append("unconstrained")
    if key_accel is not None:
        evidence.append(f"accel:{key_accel:.2f}")
    return {"decision": "set_speed_tracking", "confidence": "high", "evidence": evidence}


def decide_lateral(key_info: Dict, future_infos: List[Dict]) -> Dict:
    """Return a lateral decision aligned with Alpamayo-R1 Table 1 (8 classes)."""
    evidence: List[str] = []
    lane_info = key_info.get("lane_info")
    offset = key_info.get("lateral_offset")  # signed: positive = left of center
    key_yaw = key_info["ego_state"]["yaw"]
    future_yaws = [info["ego_state"]["yaw"] for info in future_infos if info["ego_state"].get("yaw") is not None]

    if lane_info is not None:
        # ── 1. Turn (intersection/roundabout/U-turn) ─────────────────────
        # Planned path onto a different road segment with significant heading change.
        if lane_info.is_lane_connector and future_yaws:
            yaw_delta = future_yaws[-1] - key_yaw
            # Normalize to [-π, π]
            import math
            yaw_delta = (yaw_delta + math.pi) % (2 * math.pi) - math.pi
            if abs(yaw_delta) > 0.35:
                evidence.append("lane_connector")
                evidence.append(f"yaw_delta:{yaw_delta:.2f}")
                if yaw_delta > 0:
                    return {"decision": "turn_left", "confidence": "medium", "evidence": evidence}
                return {"decision": "turn_right", "confidence": "medium", "evidence": evidence}

        # ── 2. Lane change / Merge-split ─────────────────────────────────
        changed_infos = [
            info for info in future_infos
            if info.get("lane_info") is not None
            and info["lane_info"].lane_token != lane_info.lane_token
        ]
        if changed_infos:
            target_lane = changed_infos[-1]["lane_info"]
            offset_values = [info.get("lateral_offset") for info in future_infos
                             if info.get("lateral_offset") is not None]
            offset_delta = (offset_values[-1] - offset) if (offset is not None and offset_values) else None

            # Merge/split: facility change (connector → non-connector or vice-versa)
            if lane_info.is_lane_connector != target_lane.is_lane_connector:
                evidence.append(f"facility_change:{lane_info.lane_token}->{target_lane.lane_token}")
                return {"decision": "merge_split", "confidence": "medium", "evidence": evidence}

            # Full lane change: both non-connector, meaningful offset shift
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

    # ── 3. In-lane nudge / Out-of-lane nudge / Lane keeping ──────────────
    # Decide based on signed lateral offset from lane center.
    # |offset| > 1.2 m: likely crossed lane line → out-of-lane nudge
    # 0.3 < |offset| <= 1.2 m: within-lane temporary offset → in-lane nudge
    # |offset| <= 0.3 m: centered → lane keeping
    if offset is not None:
        if offset > 1.2:
            evidence.append(f"offset:{offset:.2f}m (crossed left line)")
            return {"decision": "out_of_lane_nudge_left", "confidence": "medium", "evidence": evidence}
        if offset < -1.2:
            evidence.append(f"offset:{offset:.2f}m (crossed right line)")
            return {"decision": "out_of_lane_nudge_right", "confidence": "medium", "evidence": evidence}
        if 0.3 < offset <= 1.2:
            evidence.append(f"offset:{offset:.2f}m")
            return {"decision": "in_lane_nudge_left", "confidence": "medium", "evidence": evidence}
        if -1.2 <= offset < -0.3:
            evidence.append(f"offset:{offset:.2f}m")
            return {"decision": "in_lane_nudge_right", "confidence": "medium", "evidence": evidence}
        evidence.append(f"offset:{offset:.2f}m")
        return {"decision": "lane_keeping_centering", "confidence": "high", "evidence": evidence}

    return {"decision": "lane_keeping_centering", "confidence": "low", "evidence": ["no_offset_data"]}
