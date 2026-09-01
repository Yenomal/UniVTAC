"""Temporary OXE-DROID <-> UniVTAC state/action conversion.

The DROID checkpoint needs ``eef_9d`` as an input even when the first test
only executes its joint and gripper outputs. This file keeps that model-
specific requirement isolated from the reusable GR00T client.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .schema import STATE_KEYS, VIDEO_KEYS


def _quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to GR00T's first-two-rows rot6d format."""
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        raise ValueError("EEF quaternion has near-zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rotation[:2].reshape(6)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_uint8(value: Any) -> np.ndarray:
    image = _to_numpy(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _gripper_percent(task: Any, joint: np.ndarray) -> float:
    manager = getattr(task, "_robot_manager", None)
    if manager is not None and hasattr(manager, "get_gripper_percentage"):
        return float(manager.get_gripper_percentage())
    max_qpos = float(getattr(getattr(task.cfg, "robot", None), "gripper_max_qpos", 0.039))
    return float(joint[7] / max_qpos)


def encode_observation(task: Any, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return current RGB frames and the flat DROID state (9+1+7)."""
    cameras = observation["observation"]
    video = {
        server_key: _image_uint8(cameras[local_key]["rgb"])
        for local_key, server_key in VIDEO_KEYS.items()
    }

    raw_joint = _to_numpy(observation["embodiment"]["joint"]).reshape(-1).astype(np.float32)
    if raw_joint.shape[0] < 8:
        raise ValueError(f"UniVTAC joint observation must contain at least 8 values, got {raw_joint.shape}")
    joint = raw_joint[:7]
    gripper = np.array([_gripper_percent(task, raw_joint)], dtype=np.float32)

    ee = _to_numpy(observation["embodiment"]["ee"]).reshape(-1).astype(np.float32)
    if ee.shape[0] != 7:
        raise ValueError(f"UniVTAC ee observation must be [x,y,z,w,x,y,z], got {ee.shape}")
    eef_9d = np.concatenate([ee[:3], _quat_to_rot6d(ee[3:])]).astype(np.float32)
    state = np.concatenate([eef_9d, gripper, joint]).astype(np.float32)
    assert STATE_KEYS == ("eef_9d", "gripper_position", "joint_position")
    return video, state


def decode_qpos(task: Any, action: dict[str, np.ndarray], index: int = 0) -> torch.Tensor:
    """Use DROID's decoded joint/gripper outputs as UniVTAC qpos."""
    joint = np.asarray(action["joint_position"], dtype=np.float32)[index].reshape(-1)
    gripper = float(np.asarray(action["gripper_position"], dtype=np.float32)[index].reshape(-1)[0])
    if joint.shape[0] != 7:
        raise ValueError(f"Expected 7 decoded joint actions, got {joint.shape}")
    manager = getattr(task, "_robot_manager", None)
    if manager is not None and hasattr(manager, "gripper_percent2qpos"):
        gripper = float(manager.gripper_percent2qpos(gripper))
    else:
        gripper = gripper * float(getattr(getattr(task.cfg, "robot", None), "gripper_max_qpos", 0.039))
    return torch.from_numpy(np.concatenate([joint, [gripper]]).astype(np.float32))

