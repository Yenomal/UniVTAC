"""Synchronous single-server OpenPI policy for UniVTAC."""

from __future__ import annotations

from typing import Any

from policy._base_policy import BasePolicy
from policy._openpi import build_observation
from policy._openpi import close_client
from policy._openpi import make_client
from policy._openpi import select_actions
import torch


class Policy(BasePolicy):
    def __init__(self, args: dict[str, Any]):
        super().__init__(args)
        self._client = make_client(args.get("host", "127.0.0.1"), int(args.get("port", 8000)))
        self._side_camera = args.get("side_camera", "head")
        self._wrist_camera = args.get("wrist_camera", "wrist")
        self._state_dim = int(args.get("state_dim", 8))
        self._action_indices = args.get("action_indices")

    def eval(self, task, observation):
        request = build_observation(
            observation,
            task.instruction,
            side_camera=self._side_camera,
            wrist_camera=self._wrist_camera,
            state_dim=self._state_dim,
        )
        response = self._client.infer(request)
        action_dim = len(request["state"])
        action = select_actions(
            response,
            action_dim=action_dim,
            action_indices=self._action_indices,
        )[0]
        task.take_action(torch.as_tensor(action, device=task.device), action_type="qpos")

    def close(self):
        close_client(self._client)
