"""Synchronous observation/action client for the managed OpenPI Runtime."""

from __future__ import annotations

import contextlib
from typing import Any

import torch

from policy._base_policy import BasePolicy
from policy._openpi import build_observation, close_client, compress_qpos
from policy._openpi import is_connection_error, make_client, select_actions


def build(args: dict[str, Any]) -> "Policy":
    return Policy(args)


class Policy(BasePolicy):
    """Upload one observation and synchronously receive one action."""

    _reset_op = "reset"

    def __init__(self, args: dict[str, Any]):
        super().__init__(args)
        self._host = args.get("host", "127.0.0.1")
        self._port = int(args.get("port", 8000))
        self._reconnect_attempts = int(args.get("reconnect_attempts", 1))
        if self._reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be non-negative")
        self._client = self._make_client()
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
        self._next_action_id = 0
        self._last_executed_action_id: int | None = None

    def _reconnect(self) -> None:
        with contextlib.suppress(Exception):
            close_client(self._client)
        self._client = self._make_client()

    def _make_client(self):
        return make_client(self._host, self._port)

    def _infer(self, request: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self._reconnect_attempts + 1):
            try:
                return self._client.infer(request)
            except Exception as exc:
                if attempt >= self._reconnect_attempts or not is_connection_error(exc):
                    raise
                self._reconnect()
        raise AssertionError("unreachable")

    def _reset_request(self) -> None:
        response = self._infer({"op": self._reset_op, "clear_prefix_cache": True})
        if response.get("status") != "reset":
            raise RuntimeError("OpenPI Runtime did not reset the Runtime session")
        self._session_started = True

    def reset(self):
        self._reset_request()
        self._next_action_id = 0
        self._last_executed_action_id = None

    def needs_observation(self, task) -> bool:
        return True

    def eval(self, task, observation):
        if observation is None:
            raise ValueError("OpenPI Runtime requires an observation every control step")
        if not self._session_started:
            self.reset()
        payload = build_observation(
            observation, task.instruction,
            side_camera=self._side_camera, wrist_camera=self._wrist_camera,
            state_dim=self._state_dim, include_tactile=self._use_tactile,
            left_tactile=self._left_tactile, right_tactile=self._right_tactile,
            marker_count=self._marker_count,
        )
        action_id = self._next_action_id
        payload["runtime_action_id"] = action_id
        if self._last_executed_action_id is not None:
            payload["runtime_ack_action_id"] = self._last_executed_action_id
        response = self._infer(payload)
        if response.get("action_id") != action_id:
            raise RuntimeError(
                f"OpenPI Runtime returned action_id {response.get('action_id')!r}; "
                f"expected {action_id}"
            )
        actions = select_actions(response, action_dim=self._state_dim, action_indices=self._action_indices)
        if len(actions) != 1:
            raise ValueError(f"Runtime must return one action per observation, got {len(actions)}")
        execution = task.take_action(
            torch.as_tensor(compress_qpos(actions)[0], device=task.device), action_type="qpos"
        )
        if isinstance(execution, tuple) and execution and not bool(execution[0]):
            raise RuntimeError(f"Runtime action {action_id} was not executed successfully")
        self._last_executed_action_id = action_id
        self._next_action_id = action_id + 1

    def close(self):
        close_client(self._client)
