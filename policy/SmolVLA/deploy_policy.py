"""UniVTAC policy adapter for the standalone LeRobot Runtime server."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from .._base_policy import BasePolicy
from .runtime_client import RuntimeClient


def _as_array(value: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=dtype)
    return np.ascontiguousarray(array)


def _compress_qpos(action: np.ndarray) -> np.ndarray:
    if action.shape[-1] == 9:
        return action[:8]
    if action.shape[-1] == 8:
        return action
    raise ValueError(f"SmolVLA Runtime action must have 8 or 9 dimensions, got {action.shape[-1]}")


class Policy(BasePolicy):
    """Upload one UniVTAC observation and execute one returned action."""

    def __init__(self, args: Mapping[str, Any]):
        super().__init__(args)
        self._client = RuntimeClient(
            args.get("host", "127.0.0.1"),
            int(args.get("port", 8000)),
            timeout_s=float(args.get("timeout_s", 10.0)),
        )
        self._side_camera = str(args.get("side_camera", "head"))
        self._wrist_camera = args.get("wrist_camera", "wrist")
        self._state_dim = int(args.get("state_dim", 9))
        self._state_feature = str(args.get("state_feature", "observation.state"))
        self._side_feature = str(args.get("side_feature", "observation.images.image"))
        self._wrist_feature = args.get("wrist_feature", "observation.images.image2")
        self._action_indices = args.get("action_indices")
        self._use_tactile = bool(args.get("use_tactile", False))
        self._left_tactile = str(args.get("left_tactile", "left_tactile"))
        self._right_tactile = str(args.get("right_tactile", "right_tactile"))
        self._sequence = 0
        self._started = False

    def _features(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        try:
            source_images = observation["observation"]
            features = {
                self._side_feature: _as_array(source_images[self._side_camera]["rgb"], np.uint8),
                self._state_feature: _as_array(observation["embodiment"]["joint"], np.float32).reshape(-1)[: self._state_dim],
            }
        except KeyError as exc:
            raise ValueError(f"UniVTAC observation is missing {exc.args[0]!r}") from exc
        state = features[self._state_feature]
        if state.shape != (self._state_dim,):
            raise ValueError(f"Expected state shape ({self._state_dim},), got {state.shape}")
        if self._wrist_feature is not None and self._wrist_camera is not None:
            try:
                features[str(self._wrist_feature)] = _as_array(
                    source_images[str(self._wrist_camera)]["rgb"], np.uint8
                )
            except KeyError as exc:
                raise ValueError(f"UniVTAC observation is missing {exc.args[0]!r}") from exc
        if self._use_tactile:
            try:
                tactile = observation["tactile"]
                features["observation.tactile.left_marker"] = _as_array(
                    tactile[self._left_tactile]["marker"], np.float32
                )
                features["observation.tactile.right_marker"] = _as_array(
                    tactile[self._right_tactile]["marker"], np.float32
                )
            except KeyError as exc:
                raise ValueError(f"UniVTAC observation is missing {exc.args[0]!r}") from exc
        return features

    def reset(self):
        self._client.reset()
        self._sequence = 0
        self._started = True

    def needs_observation(self, task) -> bool:
        return True

    def eval(self, task, observation):
        if observation is None:
            raise ValueError("SmolVLA Runtime requires an observation every control step")
        if not self._started:
            self.reset()
        response = self._client.observe(self._features(observation), task=task.instruction, sequence=self._sequence)
        
        # print(
        #     f"runtime action_id={response['action_id']} "
        #     f"queue_size={response.get('queue_size')} "
        #     f"replayed={response.get('replayed', False)}"
        # )
        
        action = response["action"]
        if self._action_indices is not None:
            indices = tuple(int(index) for index in self._action_indices)
            action = action[list(indices)]
        action = _compress_qpos(action)
        if not np.isfinite(action).all():
            raise ValueError("Runtime action contains non-finite values")
        execution = task.take_action(torch.as_tensor(action, device=task.device), action_type="qpos")
        if isinstance(execution, tuple) and execution and not bool(execution[0]):
            raise RuntimeError(f"Runtime action {response['action_id']} was not executed successfully")
        self._client.acknowledge(response["action_id"])
        self._sequence += 1

    def close(self):
        self._client.close()
