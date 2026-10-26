"""Synchronous observation/action client for the managed OpenPI Runtime."""

from __future__ import annotations

import contextlib
from typing import Any

import torch

from policy._base_policy import BasePolicy
from policy._openpi import build_observation
from policy._openpi import close_client
from policy._openpi import compress_qpos
from policy._openpi import is_connection_error
from policy._openpi import make_client
from policy._openpi import select_actions


def build(args: dict[str, Any]) -> "Policy":
    return Policy(args)


class Policy(BasePolicy):
    """Upload one observation and synchronously receive one action per step."""

    def __init__(self, args: dict[str, Any]):
        super().__init__(args)
        self._host = args.get("host", "127.0.0.1")
        self._port = int(args.get("port", 8000))
        self._reconnect_attempts = int(args.get("reconnect_attempts", 1))
        if self._reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be non-negative")
        self._client = make_client(self._host, self._port)
        self._side_camera = args.get("side_camera", "head")
        self._wrist_camera = args.get("wrist_camera", "wrist")
        self._state_dim = int(args.get("state_dim", 9))
        self._action_indices = args.get("action_indices")
        self._use_tactile = bool(args.get("use_tactile", False))
        self._left_tactile = args.get("left_tactile", "left_tactile")
        self._right_tactile = args.get("right_tactile", "right_tactile")
        marker_count = args.get("marker_count")
        self._marker_count = int(marker_count) if marker_count is not None else None
        self._session_started = False

    def _reconnect(self) -> None:
        with contextlib.suppress(Exception):
            close_client(self._client)
        self._client = make_client(self._host, self._port)
        self._session_started = False

    def _infer(self, request: dict[str, Any], *, reset_after_reconnect: bool = False) -> dict[str, Any]:
        for attempt in range(self._reconnect_attempts + 1):
            try:
                return self._client.infer(request)
            except Exception as exc:
                if attempt >= self._reconnect_attempts or not is_connection_error(exc):
                    raise
                self._reconnect()
                if reset_after_reconnect:
                    self.reset()
        raise AssertionError("unreachable")

    def reset(self):
        response = self._infer({
            "op": "reset",
            "clear_prefix_cache": True,
        })
        if response.get("status") != "reset":
            raise RuntimeError("OpenPI Runtime did not reset the streaming session")
        self._session_started = True

    def needs_observation(self, task) -> bool:
        # Runtime owns the action queue and decides whether this frame is used
        # for inference, prefix refresh, or only retained as the newest input.
        return True

    def eval(self, task, observation):
        if observation is None:
            raise ValueError("Runtime streaming requires an observation every control step")
        if not self._session_started:
            self.reset()
        payload = build_observation(
            observation,
            task.instruction,
            side_camera=self._side_camera,
            wrist_camera=self._wrist_camera,
            state_dim=self._state_dim,
            include_tactile=self._use_tactile,
            left_tactile=self._left_tactile,
            right_tactile=self._right_tactile,
            marker_count=self._marker_count,
        )
        response = self._infer(payload, reset_after_reconnect=True)
        actions = select_actions(
            response,
            action_dim=self._state_dim,
            action_indices=self._action_indices,
        )
        if len(actions) != 1:
            raise ValueError(f"Runtime must return one action per observation, got {len(actions)}")
        action = compress_qpos(actions)[0]
        task.take_action(torch.as_tensor(action, device=task.device), action_type="qpos")

    def close(self):
        close_client(self._client)
