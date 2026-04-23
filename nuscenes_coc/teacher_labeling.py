"""Utilities for teacher labeling requests and final CoC assembly."""

from __future__ import annotations

from typing import Dict, List, Optional


def build_teacher_request(sample: Dict) -> Dict:
    """Convert one intermediate sample into a provider-neutral VLM request."""
    teacher_input = sample["teacher_vlm_input"]
    return {
        "request_id": sample["sample_id"],
        "sample_id": sample["sample_id"],
        "nuscenes_sample_token": sample["nuscenes_sample_token"],
        "nuscenes_scene_token": sample["nuscenes_scene_token"],
        "image_assets": teacher_input["image_assets"],
        "messages": [
            {
                "role": "system",
                "content": teacher_input["system_prompt"],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": teacher_input["user_prompt"],
                    },
                    {
                        "type": "structured_context",
                        "data": teacher_input["structured_context"],
                    },
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": sample["teacher_vlm_output_schema"],
        },
        "weak_label_hint": sample["preliminary_coc"],
        "metadata": {
            "prompt_version": teacher_input["prompt_version"],
            "format_constraints": teacher_input["format_constraints"],
            "annotation_principles": teacher_input["annotation_principles"],
        },
    }


def build_teacher_requests(samples: List[Dict]) -> List[Dict]:
    """Build provider-neutral VLM requests for all intermediate samples."""
    return [build_teacher_request(sample) for sample in samples]


def build_rule_fallback_teacher_response(sample: Dict) -> Dict:
    """Use the preliminary rule label as a local teacher response."""
    preliminary = sample["preliminary_coc"]
    return {
        "request_id": sample["sample_id"],
        "sample_id": sample["sample_id"],
        "teacher_source": "rule_fallback",
        "teacher_label": {
            "driving_decision": preliminary["driving_decision"],
            "critical_components": preliminary["critical_components"],
            "coc_reasoning": preliminary["coc_reasoning"],
        },
        "confidence": preliminary["confidence"],
        "evidence": preliminary["evidence"],
    }


def build_rule_fallback_teacher_responses(samples: List[Dict]) -> List[Dict]:
    """Build fallback teacher responses for all intermediate samples."""
    return [build_rule_fallback_teacher_response(sample) for sample in samples]


def _response_by_sample_id(teacher_responses: Optional[List[Dict]]) -> Dict[str, Dict]:
    if not teacher_responses:
        return {}
    indexed: Dict[str, Dict] = {}
    for response in teacher_responses:
        sample_id = response.get("sample_id") or response.get("request_id")
        if sample_id:
            indexed[sample_id] = response
    return indexed


def _label_from_response(response: Optional[Dict], sample: Dict) -> Dict:
    if response and "teacher_label" in response:
        return response["teacher_label"]
    return sample["preliminary_coc"]


def assemble_final_sample(sample: Dict, teacher_response: Optional[Dict] = None) -> Dict:
    """Assemble an Alpamayo-style final CoC sample from teacher output."""
    teacher_label = _label_from_response(teacher_response, sample)
    return {
        "sample_id": sample["sample_id"],
        "nuscenes_sample_token": sample["nuscenes_sample_token"],
        "nuscenes_scene_token": sample["nuscenes_scene_token"],
        "keyframe_timestamp": sample["keyframe_timestamp"],
        "camera_paths": sample["camera_paths"],
        "history_frames": sample["history_frames"],
        "future_trajectory": sample["future_trajectory"],
        "ego_state_summary": sample["ego_state_summary"],
        "object_summary": sample["object_summary"],
        "lane_summary": sample["lane_summary"],
        "driving_decision": teacher_label["driving_decision"],
        "critical_components": teacher_label["critical_components"],
        "coc_reasoning": teacher_label["coc_reasoning"],
        "teacher_source": teacher_response.get("teacher_source", "preliminary_coc") if teacher_response else "preliminary_coc",
        "teacher_confidence": teacher_response.get("confidence") if teacher_response else sample["confidence"],
        "teacher_evidence": teacher_response.get("evidence") if teacher_response else sample["evidence"],
        "preliminary_coc": sample["preliminary_coc"],
    }


def assemble_final_dataset(samples: List[Dict], teacher_responses: Optional[List[Dict]] = None) -> List[Dict]:
    """Assemble all final CoC samples, optionally using external teacher responses."""
    responses = _response_by_sample_id(teacher_responses)
    return [assemble_final_sample(sample, responses.get(sample["sample_id"])) for sample in samples]
