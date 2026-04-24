"""Quality checks for exported samples."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from .constants import CONFIDENCE_LEVELS, LATERAL_DECISIONS, LONGITUDINAL_DECISIONS, MAX_REASONABLE_SPEED_MPS


def validate_sample(sample: Dict, export_low_confidence: bool = False) -> Tuple[bool, List[str]]:
    """Validate an exported sample and return keep flag plus reasons."""
    reasons: List[str] = []
    longitudinal = sample["driving_decision"]["longitudinal"]
    lateral = sample["driving_decision"]["lateral"]
    confidence = sample["confidence"]

    if longitudinal not in LONGITUDINAL_DECISIONS:
        reasons.append("invalid_longitudinal_decision")
    if lateral not in LATERAL_DECISIONS:
        reasons.append("invalid_lateral_decision")
    if confidence not in CONFIDENCE_LEVELS:
        reasons.append("invalid_confidence")
    if confidence == "low" and not export_low_confidence:
        reasons.append("low_confidence_filtered")
    if len(sample["coc_reasoning"].strip()) < 10:
        reasons.append("invalid_reasoning_template")
    if longitudinal == "none" and lateral == "none":
        reasons.append("both_decisions_none")

    speed = sample["ego_state"]["speed"]
    if speed is None or speed < 0 or speed > MAX_REASONABLE_SPEED_MPS:
        reasons.append("invalid_speed")

    if not sample["critical_components"] and not (
        longitudinal == "set_speed_tracking" and lateral == "lane_keeping_centering"
    ):
        reasons.append("missing_components")

    for _, path in sample["camera_paths"].items():
        if not os.path.exists(path):
            reasons.append("missing_camera_path")
            break

    return (len(reasons) == 0, reasons)
