"""Teacher VLM prompt packaging for paper-style intermediate samples."""

from __future__ import annotations

from typing import Dict, List

from .constants import CAMERA_NAMES, LATERAL_DECISIONS, LONGITUDINAL_DECISIONS


PROMPT_VERSION = "alpamayo_intermediate_v1"


def _history_image_assets(history_frames: List[Dict]) -> List[Dict]:
    assets: List[Dict] = []
    for frame in history_frames:
        for camera_name in CAMERA_NAMES:
            camera_path = frame["camera_paths"].get(camera_name)
            if not camera_path:
                continue
            assets.append(
                {
                    "relative_index": frame["relative_index"],
                    "timestamp": frame["timestamp"],
                    "camera_name": camera_name,
                    "path": camera_path,
                }
            )
    return assets


def _teacher_system_prompt() -> str:
    return (
        "You are an autonomous driving CoC (Chain-of-Causality) data annotator following the Alpamayo paper methodology. "
        "You will receive multi-camera history images from the 2s observation window before the keyframe, "
        "plus structured ego-state, lane, object, and 6s future trajectory context. "
        "Your task is to output: (1) the driving decision (longitudinal + lateral), "
        "(2) 1-3 critical causal components observed from the history window only, "
        "(3) a concise English reasoning trace in the style: '[Ego action] [to/because] [causal factor(s)]'. "
        "Three strict rules: (a) Decision grounding — decision must match future trajectory evidence. "
        "(b) Causal locality — components must come from history observation only, never from future frames. "
        "(c) Annotation economy — include only components directly causing the decision. "
        "Output JSON only. No explanations, no Markdown."
    )


def _teacher_user_prompt() -> str:
    longitudinal = ", ".join(sorted(LONGITUDINAL_DECISIONS))
    lateral = ", ".join(sorted(LATERAL_DECISIONS))
    return (
        "Annotate the driving scene following these steps: "
        "Step 1 — Determine the longitudinal and lateral driving decisions using the history images and future trajectory as confirmation. "
        f"Longitudinal must be one of: {longitudinal}. "
        f"Lateral must be one of: {lateral}. "
        "Step 2 — Identify 1-3 critical causal components from the history observation window that directly caused the decision. "
        "Component categories: critical_objects, traffic_controls, road_events, lane_info, ego_motion. "
        "For each component, describe its type, relative position, and crucially HOW it affects ego behavior "
        "(e.g. 'causing partial lane obstruction, ego must nudge left to bypass'). "
        "Step 3 — Write a single English reasoning trace sentence. "
        "Format: '[Ego action verb] [to/because/due to] [causal factor(s)]'. "
        "Examples: 'Decelerate to maintain safe distance from the pedestrian crossing the road at 12m ahead.' "
        "'Nudge right within lane to bypass the stationary construction vehicle partially blocking the lane.' "
        "'Slow down and follow the lead vehicle while navigating through the construction zone.' "
        "Do NOT use 'Because...therefore' or Chinese text. Output must strictly match the provided JSON schema."
    )


def _teacher_output_schema() -> Dict:
    return {
        "type": "object",
        "required": ["driving_decision", "critical_components", "coc_reasoning"],
        "properties": {
            "driving_decision": {
                "type": "object",
                "required": ["longitudinal", "lateral"],
                "properties": {
                    "longitudinal": {
                        "type": "string",
                        "enum": sorted(LONGITUDINAL_DECISIONS),
                    },
                    "lateral": {
                        "type": "string",
                        "enum": sorted(LATERAL_DECISIONS),
                    },
                },
            },
            "critical_components": {
                "type": "array",
                "description": "只保留与当前决策直接相关的 0-3 个关键因果组件。",
                "items": {
                    "type": "object",
                    "required": ["category", "attributes"],
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "critical_objects",
                                "traffic_controls",
                                "road_events",
                                "lane_info",
                                "ego_motion",
                            ],
                        },
                        "attributes": {
                            "type": "object",
                            "description": "Fill the most essential fields for the category (e.g. distance, type, offset, curvature, speed). Include an 'ego_impact' string describing how this component affects ego driving behavior.",
                        },
                    },
                },
            },
            "coc_reasoning": {
                "type": "string",
                "description": "A single English sentence. Format: '[Ego action verb] [to/because/due to] [causal factor(s)]'. Must be consistent with driving_decision and critical_components. No Chinese text.",
            },
        },
    }


def build_teacher_vlm_package(sample_payload: Dict) -> Dict:
    """Build a teacher-labeling package from one intermediate sample."""
    history_frames = sample_payload["history_frames"]
    return {
        "teacher_vlm_input": {
            "prompt_version": PROMPT_VERSION,
            "system_prompt": _teacher_system_prompt(),
            "user_prompt": _teacher_user_prompt(),
            "annotation_principles": [
                "decision_grounding",
                "causal_locality",
                "annotation_economy",
            ],
            "image_assets": _history_image_assets(history_frames),
            "structured_context": {
                "sample_id": sample_payload["sample_id"],
                "nuscenes_sample_token": sample_payload["nuscenes_sample_token"],
                "nuscenes_scene_token": sample_payload["nuscenes_scene_token"],
                "keyframe_timestamp": sample_payload["keyframe_timestamp"],
                "keyframe_index_in_segment": sample_payload["keyframe_index_in_segment"],
                "action_index_in_segment": sample_payload["action_index_in_segment"],
                "history_frames": history_frames,
                "future_trajectory": sample_payload["future_trajectory"],
                "ego_state_summary": sample_payload["ego_state_summary"],
                "object_summary": sample_payload["object_summary"],
                "lane_summary": sample_payload["lane_summary"],
                "static_controls": sample_payload["static_controls"],
            },
            "format_constraints": {
                "json_only": True,
                "reasoning_template": "[Ego action verb] [to/because/due to] [causal factor(s)].",
                "max_critical_components": 3,
            },
        },
        "teacher_vlm_output_schema": _teacher_output_schema(),
    }
