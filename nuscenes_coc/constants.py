"""Constants shared by the nuScenes CoC pipeline."""

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Longitudinal decisions aligned with Alpamayo-R1 Table 1 (7 classes)
LONGITUDINAL_DECISIONS = {
    "set_speed_tracking",           # Maintain/reach target speed when unconstrained
    "lead_obstacle_following",      # Maintain safe time gap to lead entity
    "speed_adaptation_road",        # Adjust speed for road features (curves, bumps, turns)
    "gap_searching",                # Adjust speed to create usable gap for LC/merge
    "acceleration_passing",         # Increase speed to pass a slower lead (with lateral plan)
    "yield_agent_right_of_way",     # Slow/stop to concede priority (pedestrians, cut-ins, etc.)
    "stop_static_constraint",       # Decelerate to and hold at control points (stop line, red light)
    "none",
}

# Lateral decisions aligned with Alpamayo-R1 Table 1 (8 classes)
LATERAL_DECISIONS = {
    "lane_keeping_centering",       # Maintain position within lane; minor in-lane offsets allowed
    "merge_split",                  # Transition between facilities (on-ramp ↔ mainline)
    "out_of_lane_nudge_left",       # Brief intentional line-crossing left for clearance
    "out_of_lane_nudge_right",      # Brief intentional line-crossing right for clearance
    "in_lane_nudge_left",           # Temporary offset left within lane (no line crossing)
    "in_lane_nudge_right",          # Temporary offset right within lane (no line crossing)
    "lane_change_left",             # Full adjacent-lane transition left
    "lane_change_right",            # Full adjacent-lane transition right
    "pull_over",                    # Move toward edge/shoulder or designated stop area
    "turn_left",                    # Planned path onto different road segment, heading change left
    "turn_right",                   # Planned path onto different road segment, heading change right
    "lateral_maneuver_abort",       # Cancel ongoing lateral maneuver and re-center
    "none",
}

CONFIDENCE_LEVELS = {"high", "medium", "low"}

LONGITUDINAL_TEXT_MAP = {
    "set_speed_tracking":        "maintain or reach target speed",
    "lead_obstacle_following":   "follow the lead vehicle at a safe distance",
    "speed_adaptation_road":     "adapt speed to road conditions",
    "gap_searching":             "adjust speed to create a usable gap for the planned maneuver",
    "acceleration_passing":      "accelerate to pass the slower lead vehicle",
    "yield_agent_right_of_way":  "yield right-of-way",
    "stop_static_constraint":    "decelerate to stop at the control point",
    "none":                      "make no significant longitudinal adjustment",
}

LATERAL_TEXT_MAP = {
    "lane_keeping_centering":    "keep centered in the current lane",
    "merge_split":               "merge into or split from the adjacent facility",
    "out_of_lane_nudge_left":    "briefly cross the left lane line to increase clearance",
    "out_of_lane_nudge_right":   "briefly cross the right lane line to increase clearance",
    "in_lane_nudge_left":        "nudge left within the lane",
    "in_lane_nudge_right":       "nudge right within the lane",
    "lane_change_left":          "change to the left adjacent lane",
    "lane_change_right":         "change to the right adjacent lane",
    "pull_over":                 "pull over toward the edge or shoulder",
    "turn_left":                 "steer left through the intersection",
    "turn_right":                "steer right through the intersection",
    "lateral_maneuver_abort":    "abort the lateral maneuver and re-center",
    "none":                      "make no significant lateral adjustment",
}

DYNAMIC_VRU_CATEGORIES = {
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "vehicle.bicycle",
    "vehicle.motorcycle",
}

VEHICLE_CATEGORIES = {
    "vehicle.car",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.trailer",
    "vehicle.truck",
}

DEFAULT_FRONT_DISTANCE_M = 50.0
DEFAULT_FRONT_LATERAL_RANGE_M = 8.0
MAX_REASONABLE_SPEED_MPS = 30.0

LONGITUDINAL_META_ACTIONS = {
    "strong_accelerate",
    "gentle_accelerate",
    "maintain_speed",
    "gentle_decelerate",
    "strong_decelerate",
    "stop",
    "reverse",
}

LATERAL_META_ACTIONS = {
    "sharp_steer_left",
    "steer_left",
    "go_straight",
    "steer_right",
    "sharp_steer_right",
    "reverse_left",
    "reverse_right",
}
