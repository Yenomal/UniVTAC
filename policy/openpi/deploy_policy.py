"""Synchronous single-server OpenPI policy for UniVTAC."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

from policy._base_policy import BasePolicy
from policy._openpi import build_observation
from policy._openpi import close_client
from policy._openpi import compress_qpos
from policy._openpi import is_connection_error
from policy._openpi import make_client
from policy._openpi import select_actions
import torch

logger = logging.getLogger(__name__)


class Policy(BasePolicy):
    def __init__(self, args: dict[str, Any]):
        super().__init__(args)
        self._host = args.get("host", "127.0.0.1")
        self._port = int(args.get("port", 8000))
        self._reconnect_attempts = int(args.get("reconnect_attempts", 1))
        self._client = make_client(self._host, self._port)
        self._side_camera = args.get("side_camera", "head")
        self._wrist_camera = args.get("wrist_camera", "wrist")
        self._state_dim = int(args.get("state_dim", 8))
        self._action_indices = args.get("action_indices")
        self._watermark = int(args.get("watermark", 1))
        if self._watermark <= 0:
            raise ValueError("watermark must be positive")
        if self._reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be non-negative")
        self._action_buffer = None
        self._timing = {"count": 0, "last_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}

    def reset(self):
        self._action_buffer = None
        self._timing = {"count": 0, "last_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}

    def needs_observation(self, task) -> bool:
        """Only sample a new frame when the synchronous chunk is exhausted."""
        return self._action_buffer is None or len(self._action_buffer) == 0

    def eval(self, task, observation):
        if self.needs_observation(task):
            if observation is None:
                raise ValueError("OpenPI requires an observation when requesting a new action chunk")
            request = build_observation(
                observation,
                task.instruction,
                side_camera=self._side_camera,
                wrist_camera=self._wrist_camera,
                state_dim=self._state_dim,
            )
            response = self._infer(request)
            actions = select_actions(
                response,
                action_dim=len(request["state"]),
                action_indices=self._action_indices,
            )
            if len(actions) < self._watermark:
                raise ValueError(
                    f"OpenPI returned {len(actions)} actions, but watermark={self._watermark}"
                )
            self._action_buffer = compress_qpos(actions[:self._watermark])
        action = self._action_buffer[0]
        self._action_buffer = self._action_buffer[1:]
        task.take_action(torch.as_tensor(action, device=task.device), action_type="qpos")
        task.metadata["policy_timing"] = {"infer": self._timing.copy()}

    def _infer(self, request):
        started_at = time.perf_counter()
        try:
            for attempt in range(self._reconnect_attempts + 1):
                try:
                    return self._client.infer(request)
                except Exception as exc:
                    if attempt >= self._reconnect_attempts or not is_connection_error(exc):
                        raise
                    logger.warning("OpenPI connection failed; reconnecting", exc_info=True)
                    with contextlib.suppress(Exception):
                        close_client(self._client)
                    self._client = make_client(self._host, self._port)
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            count = self._timing["count"] + 1
            self._timing["count"] = count
            self._timing["last_ms"] = elapsed_ms
            self._timing["mean_ms"] += (elapsed_ms - self._timing["mean_ms"]) / count
            self._timing["max_ms"] = max(self._timing["max_ms"], elapsed_ms)

    def close(self):
        close_client(self._client)
