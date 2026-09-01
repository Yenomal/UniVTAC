"""Schema for the first GR00T N1.7 OXE-DROID integration test."""

EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"

VIDEO_KEYS = {
    "head": "exterior_image_1_left",
    "wrist": "wrist_image_left",
}
VIDEO_DELTA_INDICES = (-15, 0)

STATE_KEYS = ("eef_9d", "gripper_position", "joint_position")
STATE_DIMS = {
    "eef_9d": 9,
    "gripper_position": 1,
    "joint_position": 7,
}
LANGUAGE_KEY = "annotation.language.language_instruction"

ACTION_KEYS = ("eef_9d", "gripper_position", "joint_position")
ACTION_HORIZON = 40

UNIVTAC_ARM_DOF = 7
UNIVTAC_QPOS_DOF = 8

