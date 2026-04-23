#!/usr/bin/env python3
"""
Export nuScenes CoC dataset to NVIDIA PhysicalAI-AV official ood_reasoning.parquet format.

Official schema (verified from actual ood_reasoning.parquet sample):
  Index (clip_uuid) : str   - UUID string, used as DataFrame index
  feature           : str   - always "camera_front_wide_120fov"
  event_cluster     : str   - OOD event category (UPPER_CASE)
  events            : list  - [{event_start_frame, event_start_timestamp, cot}]
                              cot: short English sentence describing the action
  split             : str   - "train" / "val"

Usage:
  python scripts/export_to_official_format.py \\
      --input out/nuscenes_coc_final_rule_fallback_full.json \\
      --output-dir out/official \\
      --auto-split

  # With teacher LLM responses:
  python scripts/export_to_official_format.py \\
      --input out/nuscenes_coc_final_teacher.json \\
      --output-dir out/official \\
      --auto-split
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# OOD event_cluster mapping
# Maps our driving_decision -> official OOD taxonomy (UPPER_CASE)
# ---------------------------------------------------------------------------
# Lateral decisions that map to specific OOD clusters
LATERAL_TO_CLUSTER = {
    "turn_intersection":    "COMPLEX_INTERSECTION_INTERACTION",
    "lane_change_left":     "SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR",
    "lane_change_right":    "SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR",
    "in_lane_nudge_left":   "SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR",
    "in_lane_nudge_right":  "SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR",
}

# Longitudinal decisions that map to specific OOD clusters
LONGITUDINAL_TO_CLUSTER = {
    "yield_agent_right_of_way": "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY",
    "stop_static_constraint":   "WORK_ZONES_TEMP_TRAFFIC_CONTROL",
    "speed_adaptation_road":    "WORK_ZONES_TEMP_TRAFFIC_CONTROL",
    "lead_obstacle_following":  "SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR",
    "set_speed_tracking":       "SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR",
}


def derive_event_cluster(longitudinal: str, lateral: str, components: list) -> str:
    """Map driving decision to official OOD event_cluster category."""
    # Check if pedestrian in components → pedestrian cluster
    for comp in components:
        attrs = comp.get("attributes", {})
        cat = attrs.get("type", "") or attrs.get("category", "")
        if "pedestrian" in cat.lower():
            return "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY"

    # Lateral overrides first
    if lateral in LATERAL_TO_CLUSTER:
        return LATERAL_TO_CLUSTER[lateral]
    # Longitudinal next
    if longitudinal in LONGITUDINAL_TO_CLUSTER:
        return LONGITUDINAL_TO_CLUSTER[longitudinal]
    return "SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR"


def coc_reasoning_to_cot(reasoning: str, longitudinal: str, lateral: str) -> str:
    """
    Convert Chinese "因为...，所以..." reasoning to a short English cot sentence.
    Mirrors the official dataset cot style (concise, action-focused English).
    """
    # Template-based English cot from decision type
    lon_templates = {
        "lead_obstacle_following":  "Maintain a safe following distance from the lead vehicle ahead.",
        "yield_agent_right_of_way": "Decelerate to yield the right-of-way to the agent.",
        "stop_static_constraint":   "Decelerate to a stop due to a static constraint ahead.",
        "speed_adaptation_road":    "Adapt speed to road conditions ahead.",
        "set_speed_tracking":       "Maintain current speed following the traffic flow.",
        "none":                     "Maintain current driving behavior.",
    }
    lat_templates = {
        "turn_intersection":        "Navigate through the intersection with a turning maneuver.",
        "lane_change_left":         "Perform a lane change to the left.",
        "lane_change_right":        "Perform a lane change to the right.",
        "in_lane_nudge_left":       "Nudge slightly left within the current lane.",
        "in_lane_nudge_right":      "Nudge slightly right within the current lane.",
        "lane_keeping_centering":   "",
        "none":                     "",
    }

    lon_text = lon_templates.get(longitudinal, "Maintain current driving behavior.")
    lat_text = lat_templates.get(lateral, "")

    # Build cot
    if lat_text and lateral not in ("lane_keeping_centering", "none"):
        if lon_text and longitudinal not in ("set_speed_tracking", "none"):
            cot = lon_text.rstrip(".") + " while " + lat_text[:1].lower() + lat_text[1:]
        else:
            cot = lat_text
    else:
        cot = lon_text

    return cot


def convert_sample(sample: dict, split: str) -> tuple:
    """
    Convert one CoC sample to official format.
    Returns (clip_uuid, row_dict).
    """
    dd = sample.get("driving_decision", {})
    longitudinal = dd.get("longitudinal", "none")
    lateral = dd.get("lateral", "none")
    components = sample.get("critical_components", [])
    reasoning = sample.get("coc_reasoning", "")

    event_cluster = derive_event_cluster(longitudinal, lateral, components)

    # Use VLM coc_reasoning directly if it's already English (v2+),
    # otherwise fall back to template-based generation for legacy Chinese text.
    if reasoning and not any(c > '\u007f' for c in reasoning):
        cot = reasoning.strip()
    else:
        cot = coc_reasoning_to_cot(reasoning, longitudinal, lateral)

    # Use keyframe timestamp as event_start_timestamp
    keyframe_ts = sample.get("keyframe_timestamp", 0)

    # Estimate event_start_frame: nuScenes samples at ~2 Hz, timestamps in microseconds
    # Use relative offset from first history frame
    history = sample.get("history_frames", [])
    if history:
        t0 = history[0]["timestamp"]
        elapsed_us = keyframe_ts - t0
        # nuScenes ~2Hz → 0.5s per frame; but official dataset is 30fps video
        # Map: report approximate frame assuming 30fps from segment start
        event_start_frame = max(0, round(elapsed_us / 1e6 * 30))
    else:
        event_start_frame = 0

    events = [{
        "event_start_frame": event_start_frame,
        "event_start_timestamp": keyframe_ts,
        "cot": cot,
        # Extended fields (not in official but kept for traceability)
        "longitudinal": longitudinal,
        "lateral": lateral,
        "coc_reasoning_zh": reasoning,
    }]

    clip_uuid = sample.get("sample_id", sample.get("nuscenes_sample_token", ""))

    row = {
        "feature": "camera_front_wide_120fov",
        "event_cluster": event_cluster,
        "events": events,
        "split": split,
    }
    return clip_uuid, row


def main():
    parser = argparse.ArgumentParser(
        description="Export nuScenes CoC JSON to official ood_reasoning.parquet format"
    )
    parser.add_argument(
        "--input", "-i",
        default="out/nuscenes_coc_final_rule_fallback_full.json",
        help="Path to the final CoC JSON dataset"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="out/official",
        help="Output directory (will contain reasoning/ood_reasoning.parquet)"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="Dataset split label (used when --auto-split is not set)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of scenes assigned to val (default: 0.2)"
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        help="Automatically assign train/val splits (80/20 by scene)"
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas required. Run: pip install pandas pyarrow")
        sys.exit(1)

    # Load input
    print(f"[INFO] Loading {args.input} ...")
    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"[INFO] Loaded {len(samples)} samples")

    # Auto-split by scene token
    if args.auto_split:
        scene_tokens = sorted({s.get("nuscenes_scene_token", "") for s in samples})
        n_val = max(1, int(len(scene_tokens) * args.val_ratio))
        val_scenes = set(scene_tokens[-n_val:])
        print(f"[INFO] Auto-split: {len(scene_tokens)-n_val} train scenes, {n_val} val scenes")
    else:
        val_scenes = set()

    # Convert
    index_list = []
    rows = []
    for s in samples:
        scene_tok = s.get("nuscenes_scene_token", "")
        split = "val" if (args.auto_split and scene_tok in val_scenes) else args.split
        clip_uuid, row = convert_sample(s, split)
        index_list.append(clip_uuid)
        rows.append(row)

    df = pd.DataFrame(rows, index=index_list)
    df.index.name = "clip_uuid"

    # Output
    reasoning_dir = os.path.join(args.output_dir, "reasoning")
    os.makedirs(reasoning_dir, exist_ok=True)
    out_path = os.path.join(reasoning_dir, "ood_reasoning.parquet")
    df.to_parquet(out_path, engine="pyarrow")
    print(f"[INFO] Saved {len(df)} rows -> {out_path}")

    # Schema summary
    print("\n[INFO] Schema (matches official format):")
    print(f"  {'(index) clip_uuid':<30} object     e.g. {index_list[0][:50]}")
    for col, dtype in df.dtypes.items():
        val = df[col].iloc[0]
        example = str(val)[:80] if not isinstance(val, list) else json.dumps(val[0])[:80]
        print(f"  {col:<30} {str(dtype):<10} e.g. {example}")

    print("\n[INFO] Split distribution:")
    print(df["split"].value_counts().to_string())

    print("\n[INFO] event_cluster distribution:")
    print(df["event_cluster"].value_counts().to_string())

    print("\n[INFO] Sample cot texts:")
    for i, row in df.iterrows():
        cot = row["events"][0]["cot"] if row["events"] else ""
        print(f"  [{row['event_cluster']}] {cot}")

    # Also write JSONL for human inspection
    jsonl_path = os.path.join(reasoning_dir, "ood_reasoning.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for clip_uuid, row in zip(index_list, rows):
            record = {"clip_uuid": clip_uuid}
            record.update(row)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n[INFO] Human-readable JSONL -> {jsonl_path}")


if __name__ == "__main__":
    main()
