"""Constants shared by the nuScenes CoC pipeline."""

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

LONGITUDINAL_DECISIONS = {
    "stop_static_constraint",
    "yield_agent_right_of_way",
    "lead_obstacle_following",
    "speed_adaptation_road",
    "set_speed_tracking",
    "none",
}

LATERAL_DECISIONS = {
    "turn_intersection",
    "lane_change_left",
    "lane_change_right",
    "in_lane_nudge_left",
    "in_lane_nudge_right",
    "lane_keeping_centering",
    "none",
}

CONFIDENCE_LEVELS = {"high", "medium", "low"}

LONGITUDINAL_TEXT_MAP = {
    "stop_static_constraint": "减速停车等待",
    "yield_agent_right_of_way": "减速让行",
    "lead_obstacle_following": "跟车行驶",
    "speed_adaptation_road": "减速通过",
    "set_speed_tracking": "保持当前车速",
    "none": "不做明显纵向调整",
}

LATERAL_TEXT_MAP = {
    "turn_intersection": "完成路口转向",
    "lane_change_left": "向左变道",
    "lane_change_right": "向右变道",
    "in_lane_nudge_left": "向左小幅偏移",
    "in_lane_nudge_right": "向右小幅偏移",
    "lane_keeping_centering": "保持在车道中心",
    "none": "不做明显横向调整",
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
