"""CLI entrypoints for nuScenes CoC generation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List

from .component_extractor import extract_components
from .decision_rules import decide_lateral, decide_longitudinal
from .exporter import write_json
from .meta_actions import infer_meta_actions, summarize_meta_actions
from .nusc_access import NuScenesCOCAccess
from .quality import validate_sample
from .segment_filter import build_candidate_segments
from .teacher_prompt import build_teacher_vlm_package
from .text_templates import generate_coc_reasoning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Demo nuScenes CoC dataset.")
    parser.add_argument("--data-root", required=True, help="nuScenes dataroot")
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes version")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--stats-output", default=None, help="Optional stats JSON path")
    parser.add_argument("--max-scenes", type=int, default=-1, help="Max number of scenes to process")
    parser.add_argument("--export-low-confidence", action="store_true", help="Keep low-confidence samples")
    return parser.parse_args()


def _scene_sample_infos(access: NuScenesCOCAccess, scene_token: str) -> List[Dict]:
    infos: List[Dict] = []
    for sample in access.get_scene_sample_sequence(scene_token):
        lane_info = access.get_ego_lane_info(sample["token"])
        infos.append(
            {
                "sample_token": sample["token"],
                "scene_token": scene_token,
                "sample": sample,
                "ego_state": access.get_ego_state(sample["token"]),
                "lane_info": lane_info,
                "lateral_offset": access.get_ego_lateral_offset(sample["token"]),
                "objects_in_front": access.get_objects_in_front(sample["token"]),
                "static_controls": access.get_static_control_nearby(sample["token"]),
                "camera_paths": access.get_sample_camera_paths(sample["token"]),
            }
        )
    return infos


def _history_frames(sample_infos: List[Dict], keyframe_idx: int, history_window: int = 4) -> List[Dict]:
    """Build history-frame records for the 2s observation window."""
    start = max(0, keyframe_idx - history_window)
    frames: List[Dict] = []
    for idx in range(start, keyframe_idx + 1):
        info = sample_infos[idx]
        frames.append(
            {
                "relative_index": idx - keyframe_idx,
                "sample_token": info["sample_token"],
                "timestamp": info["ego_state"]["timestamp"],
                "camera_paths": info["camera_paths"],
                "ego_state": {
                    "speed": info["ego_state"]["speed"],
                    "acceleration": info["ego_state"]["acceleration"],
                    "yaw_rate": info["ego_state"]["yaw_rate"],
                },
                "meta_actions": infer_meta_actions(info["ego_state"]),
            }
        )
    return frames


def _future_trajectory(sample_infos: List[Dict], keyframe_idx: int, future_window: int = 12) -> List[Dict]:
    """Build a 6s relative future trajectory sequence from the keyframe."""
    key_pose = sample_infos[keyframe_idx]["ego_state"]["translation"]
    trajectory: List[Dict] = []
    end_idx = min(len(sample_infos), keyframe_idx + future_window + 1)
    for idx in range(keyframe_idx + 1, end_idx):
        info = sample_infos[idx]
        translation = info["ego_state"]["translation"]
        trajectory.append(
            {
                "relative_index": idx - keyframe_idx,
                "sample_token": info["sample_token"],
                "timestamp": info["ego_state"]["timestamp"],
                "relative_translation": {
                    "x": translation[0] - key_pose[0],
                    "y": translation[1] - key_pose[1],
                    "z": translation[2] - key_pose[2],
                },
                "speed": info["ego_state"]["speed"],
                "acceleration": info["ego_state"]["acceleration"],
                "yaw": info["ego_state"]["yaw"],
                "yaw_rate": info["ego_state"]["yaw_rate"],
                "meta_actions": infer_meta_actions(info["ego_state"]),
            }
        )
    return trajectory


def _ego_state_summary(sample_infos: List[Dict], segment: Dict, keyframe_idx: int) -> Dict:
    """Summarize ego kinematics over history and future windows."""
    history_states = [sample_infos[idx]["ego_state"] for idx in segment["sample_indices"] if idx <= keyframe_idx]
    future_states = [sample_infos[idx]["ego_state"] for idx in segment["sample_indices"] if idx > keyframe_idx]
    history_speeds = [state["speed"] for state in history_states if state["speed"] is not None]
    future_speeds = [state["speed"] for state in future_states if state["speed"] is not None]
    meta_actions = {
        "history": summarize_meta_actions(history_states),
        "future": summarize_meta_actions(future_states),
    }
    return {
        "keyframe": {
            "speed": sample_infos[keyframe_idx]["ego_state"]["speed"],
            "acceleration": sample_infos[keyframe_idx]["ego_state"]["acceleration"],
            "yaw_rate": sample_infos[keyframe_idx]["ego_state"]["yaw_rate"],
            "meta_actions": infer_meta_actions(sample_infos[keyframe_idx]["ego_state"]),
        },
        "history": {
            "num_frames": len(history_states),
            "mean_speed": mean(history_speeds) if history_speeds else None,
            "min_speed": min(history_speeds) if history_speeds else None,
            "max_speed": max(history_speeds) if history_speeds else None,
        },
        "future": {
            "num_frames": len(future_states),
            "mean_speed": mean(future_speeds) if future_speeds else None,
            "min_speed": min(future_speeds) if future_speeds else None,
            "max_speed": max(future_speeds) if future_speeds else None,
        },
        "meta_action_summary": meta_actions,
    }


def _object_summary(key_info: Dict) -> Dict:
    """Summarize decision-relevant front objects for downstream teacher VLM use."""
    objects = key_info["objects_in_front"]
    nearest = objects[0] if objects else None
    same_lane = [obj for obj in objects if obj["same_lane"]]
    summary = {
        "num_front_objects": len(objects),
        "nearest_object": None,
        "nearest_same_lane_vehicle": None,
        "objects_topk": objects[:5],
    }
    if nearest is not None:
        summary["nearest_object"] = {
            "category_name": nearest["category_name"],
            "distance": nearest["distance"],
            "relative_pose": nearest["relative_pose"],
            "same_lane": nearest["same_lane"],
        }
    if same_lane:
        lead = same_lane[0]
        summary["nearest_same_lane_vehicle"] = {
            "category_name": lead["category_name"],
            "distance": lead["distance"],
            "relative_pose": lead["relative_pose"],
            "velocity": lead["velocity"],
        }
    return summary


def _lane_summary(key_info: Dict) -> Dict:
    """Summarize lane/topology information."""
    lane_info = key_info.get("lane_info")
    if lane_info is None:
        return {
            "available": False,
            "lane_token": None,
            "is_lane_connector": None,
            "lane_count_nearby": None,
            "distance_to_center": None,
            "signed_offset": None,
            "heading_error": None,
            "curvature": None,
        }
    return {
        "available": True,
        "lane_token": lane_info.lane_token,
        "is_lane_connector": lane_info.is_lane_connector,
        "lane_count_nearby": lane_info.lane_count_nearby,
        "distance_to_center": lane_info.distance_to_center,
        "signed_offset": lane_info.signed_offset,
        "heading_error": lane_info.heading_error,
        "curvature": lane_info.curvature,
    }


def _preliminary_coc(
    key_info: Dict,
    driving_decision: Dict,
    components: List[Dict],
    coc_reasoning: str,
    confidence: str,
    evidence: List[str],
) -> Dict:
    """Package the current rule-generated CoC as a preliminary teacher signal."""
    return {
        "driving_decision": driving_decision,
        "critical_components": components,
        "coc_reasoning": coc_reasoning,
        "confidence": confidence,
        "evidence": evidence,
        "keyframe_sample_token": key_info["sample_token"],
    }


def build_dataset(args: argparse.Namespace) -> Dict:
    access = NuScenesCOCAccess(args.data_root, args.version)
    scene_tokens = [scene["token"] for scene in access.nusc.scene]
    if args.max_scenes >= 0:
        scene_tokens = scene_tokens[: args.max_scenes]

    exported_samples: List[Dict] = []
    decision_counter: Counter = Counter()
    confidence_counter: Counter = Counter()
    drop_reasons: Counter = Counter()
    candidate_segments_total = 0
    samples_seen = 0

    for scene_token in scene_tokens:
        sample_infos = _scene_sample_infos(access, scene_token)
        samples_seen += len(sample_infos)
        segments = build_candidate_segments(sample_infos)
        candidate_segments_total += len(segments)

        for segment in segments:
            key_info = sample_infos[segment["keyframe_idx"]]
            future_infos = [sample_infos[idx] for idx in segment["sample_indices"] if idx > segment["keyframe_idx"]]

            longitudinal = decide_longitudinal(key_info, future_infos)
            lateral = decide_lateral(key_info, future_infos)
            driving_decision = {
                "longitudinal": longitudinal["decision"],
                "lateral": lateral["decision"],
            }
            confidence = "high"
            if "low" in {longitudinal["confidence"], lateral["confidence"]}:
                confidence = "low"
            elif "medium" in {longitudinal["confidence"], lateral["confidence"]}:
                confidence = "medium"

            # P0 filter: reject low-signal segments with no explicit causal trigger.
            # Pure set_speed_tracking + lane_keeping_centering is the "low-signal clip"
            # class the paper explicitly excludes (Fig 3, step 1). Only keep it when
            # there is a clear lead vehicle close ahead or active traffic control —
            # i.e. the ego is explicitly *following* something or *reacting* to something.
            if (
                driving_decision["longitudinal"] == "set_speed_tracking"
                and driving_decision["lateral"] == "lane_keeping_centering"
            ):
                close_lead = [
                    obj for obj in key_info["objects_in_front"]
                    if obj["same_lane"] and obj["distance"] < 18.0
                ]
                very_close_any = [
                    obj for obj in key_info["objects_in_front"]
                    if obj["distance"] < 6.0
                ]
                has_controls = key_info["static_controls"].get("has_stop_line", False)
                if not close_lead and not very_close_any and not has_controls:
                    drop_reasons["low_signal_no_causal_trigger"] += 1
                    continue

            evidence = longitudinal["evidence"] + lateral["evidence"] + segment["reasons"]
            components = extract_components(key_info, driving_decision)
            coc_reasoning = generate_coc_reasoning(driving_decision, components)
            history_frames = _history_frames(sample_infos, segment["keyframe_idx"])
            future_trajectory = _future_trajectory(sample_infos, segment["keyframe_idx"])
            ego_state_summary = _ego_state_summary(sample_infos, segment, segment["keyframe_idx"])
            object_summary = _object_summary(key_info)
            lane_summary = _lane_summary(key_info)
            preliminary_coc = _preliminary_coc(
                key_info,
                driving_decision,
                components,
                coc_reasoning,
                confidence,
                evidence,
            )

            sample_payload = {
                "sample_id": f"nuscenes_coc_{key_info['sample_token']}",
                "nuscenes_sample_token": key_info["sample_token"],
                "nuscenes_scene_token": key_info["scene_token"],
                "keyframe_timestamp": key_info["ego_state"]["timestamp"],
                "keyframe_index_in_segment": segment["keyframe_idx"] - segment["sample_indices"][0],
                "action_index_in_segment": segment["action_idx"] - segment["sample_indices"][0],
                "segment_sample_tokens": [sample_infos[idx]["sample_token"] for idx in segment["sample_indices"]],
                "camera_paths": key_info["camera_paths"],
                "ego_state": {
                    "speed": key_info["ego_state"]["speed"],
                    "acceleration": key_info["ego_state"]["acceleration"],
                    "yaw_rate": key_info["ego_state"]["yaw_rate"],
                },
                "history_frames": history_frames,
                "future_trajectory": future_trajectory,
                "ego_state_summary": ego_state_summary,
                "object_summary": object_summary,
                "lane_summary": lane_summary,
                "static_controls": key_info["static_controls"],
                "preliminary_coc": preliminary_coc,
                "driving_decision": driving_decision,
                "critical_components": components,
                "coc_reasoning": coc_reasoning,
                "confidence": confidence,
                "evidence": evidence,
                "quality_flags": [],
            }
            sample_payload.update(build_teacher_vlm_package(sample_payload))

            keep, reasons = validate_sample(sample_payload, export_low_confidence=args.export_low_confidence)
            if keep:
                exported_samples.append(sample_payload)
                decision_counter[(driving_decision["longitudinal"], driving_decision["lateral"])] += 1
                confidence_counter[confidence] += 1
            else:
                for reason in reasons:
                    drop_reasons[reason] += 1

    stats = {
        "version": args.version,
        "num_scenes": len(scene_tokens),
        "num_samples_seen": samples_seen,
        "num_candidate_segments": candidate_segments_total,
        "num_exported_samples": len(exported_samples),
        "decision_distribution": {f"{k[0]}__{k[1]}": v for k, v in decision_counter.items()},
        "confidence_distribution": dict(confidence_counter),
        "drop_reasons": dict(drop_reasons),
    }
    return {"samples": exported_samples, "stats": stats}


def main() -> None:
    args = parse_args()
    result = build_dataset(args)
    output_path = Path(args.output)
    stats_path = Path(args.stats_output) if args.stats_output else output_path.with_name(output_path.stem + "_stats.json")
    write_json(output_path, result["samples"])
    write_json(stats_path, result["stats"])
    print(f"Exported {len(result['samples'])} samples to {output_path}")
    print(f"Wrote stats to {stats_path}")


if __name__ == "__main__":
    main()
