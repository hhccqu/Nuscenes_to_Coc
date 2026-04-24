"""Template-based CoC text generation."""

from __future__ import annotations

from typing import Dict, List

from .constants import LATERAL_TEXT_MAP, LONGITUDINAL_TEXT_MAP


def _component_to_text(component: Dict) -> str:
    category = component["category"]
    attrs = component["attributes"]
    if category == "critical_objects":
        if "distance" in attrs:
            return f"前方 {attrs['distance']:.1f}m 处存在{attrs['type']}"
        return f"前方存在{attrs.get('type', '关键目标')}"
    if category == "traffic_controls":
        stop_type = attrs.get("stop_line_type") or "停止线"
        dist = attrs.get("distance_to_stop_line")
        if dist is not None:
            return f"前方 {dist:.1f}m 处存在 {stop_type} 类型停止线"
        return f"前方存在 {stop_type} 类型静态交通控制"
    if category == "road_events":
        return f"前方车道曲率增大，当前曲率约为 {attrs.get('curvature', 0.0):.3f}"
    if category == "lane_info":
        offset = attrs.get("signed_offset")
        if offset is None:
            return f"当前位于车道 {attrs.get('lane_token')}"
        if abs(offset) < 0.3:
            return f"当前位于车道 {attrs.get('lane_token')}，且相对车道中心偏移较小"
        direction = "左侧" if offset > 0 else "右侧"
        return f"当前位于车道 {attrs.get('lane_token')}，且相对车道中心向{direction}偏移约 {abs(offset):.2f}m"
    if category == "ego_motion":
        acc = attrs.get("acceleration")
        if acc is not None:
            return f"自车当前加速度较小，约为 {acc:.2f}m/s^2"
        return "自车当前速度变化较小"
    return "当前场景中存在直接影响决策的关键因素"


def generate_coc_reasoning(driving_decision: Dict, components: List[Dict]) -> str:
    """Generate a rule-based preliminary reasoning string (English, verb-phrase style)."""
    lon = driving_decision.get("longitudinal", "none")
    lat = driving_decision.get("lateral", "lane_keeping_centering")
    longitudinal_text = LONGITUDINAL_TEXT_MAP.get(lon, lon)
    lateral_text = LATERAL_TEXT_MAP.get(lat, lat)

    if not components:
        return f"{longitudinal_text.capitalize()} while {lateral_text} due to stable road conditions."

    # Build a short cause phrase from the first critical component
    c = components[0]
    cat = c.get("category", "")
    attrs = c.get("attributes", {})
    if cat == "critical_objects":
        dist = attrs.get("distance")
        obj_type = attrs.get("type", "object")
        cause = f"the {obj_type} at {dist:.1f}m" if dist is not None else f"the {obj_type} ahead"
    elif cat == "traffic_controls":
        dist = attrs.get("distance_to_stop_line")
        cause = f"the stop line at {dist:.1f}m" if dist is not None else "the upcoming stop line"
    elif cat == "road_events":
        cause = attrs.get("description", "road conditions ahead")
    elif cat == "lane_info":
        offset = attrs.get("signed_offset")
        cause = f"lateral offset of {offset:.2f}m from lane center" if offset is not None else "lane position"
    elif cat == "ego_motion":
        cause = "current ego motion state"
    else:
        cause = "current scene conditions"

    lat_lower = lateral_text[:1].lower() + lateral_text[1:]
    if lat in ("lane_keeping_centering",):
        return f"{longitudinal_text.capitalize()} due to {cause}."
    return f"{longitudinal_text.capitalize()} while {lat_lower} due to {cause}."
