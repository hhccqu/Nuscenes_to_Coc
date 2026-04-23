"""Extract critical causal components."""

from __future__ import annotations

from typing import Dict, List

from .constants import DYNAMIC_VRU_CATEGORIES, VEHICLE_CATEGORIES


def extract_components(key_info: Dict, driving_decision: Dict) -> List[Dict]:
    """Extract critical components tied to the chosen decisions."""
    components: List[Dict] = []
    longitudinal = driving_decision["longitudinal"]
    lateral = driving_decision["lateral"]

    if longitudinal == "lead_obstacle_following":
        lead = next((obj for obj in key_info["objects_in_front"] if obj["category_name"] in VEHICLE_CATEGORIES and obj["same_lane"]), None)
        if lead:
            components.append(
                {
                    "category": "critical_objects",
                    "source": {"sample_token": key_info["sample_token"], "annotation_token": lead["annotation_token"], "map_token": lead["lane_token"]},
                    "attributes": {
                        "type": lead["category_name"],
                        "distance": lead["distance"],
                        "relative_pose": lead["relative_pose"],
                        "same_lane": lead["same_lane"],
                    },
                    "confidence": "high",
                }
            )

    if longitudinal == "yield_agent_right_of_way":
        for obj in key_info["objects_in_front"]:
            if obj["category_name"] in DYNAMIC_VRU_CATEGORIES and obj["distance"] < 15.0:
                components.append(
                    {
                        "category": "critical_objects",
                        "source": {"sample_token": key_info["sample_token"], "annotation_token": obj["annotation_token"], "map_token": obj["lane_token"]},
                        "attributes": {
                            "type": obj["category_name"],
                            "distance": obj["distance"],
                            "relative_pose": obj["relative_pose"],
                        },
                        "confidence": "medium",
                    }
                )
                break

    if longitudinal == "stop_static_constraint":
        controls = key_info["static_controls"]
        components.append(
            {
                "category": "traffic_controls",
                "source": {"sample_token": key_info["sample_token"], "annotation_token": None, "map_token": None},
                "attributes": controls,
                "confidence": "high" if controls["has_stop_line"] else "medium",
            }
        )

    if longitudinal == "speed_adaptation_road" and key_info.get("lane_info") is not None:
        lane_info = key_info["lane_info"]
        components.append(
            {
                "category": "road_events",
                "source": {"sample_token": key_info["sample_token"], "annotation_token": None, "map_token": lane_info.lane_token},
                "attributes": {"curvature": lane_info.curvature, "lane_token": lane_info.lane_token},
                "confidence": "medium",
            }
        )

    if lateral in {"lane_change_left", "lane_change_right", "lane_keeping_centering", "in_lane_nudge_left", "in_lane_nudge_right"} and key_info.get("lane_info") is not None:
        lane_info = key_info["lane_info"]
        components.append(
            {
                "category": "lane_info",
                "source": {"sample_token": key_info["sample_token"], "annotation_token": None, "map_token": lane_info.lane_token},
                "attributes": {
                    "lane_token": lane_info.lane_token,
                    "lane_count_nearby": lane_info.lane_count_nearby,
                    "distance_to_center": lane_info.distance_to_center,
                    "signed_offset": lane_info.signed_offset,
                },
                "confidence": "high",
            }
        )

    if longitudinal == "set_speed_tracking" and key_info.get("lane_info") is not None:
        lane_info = key_info["lane_info"]
        components.append(
            {
                "category": "ego_motion",
                "source": {"sample_token": key_info["sample_token"], "annotation_token": None, "map_token": lane_info.lane_token},
                "attributes": {
                    "speed": key_info["ego_state"]["speed"],
                    "acceleration": key_info["ego_state"]["acceleration"],
                },
                "confidence": "high",
            }
        )

    return components[:4]
