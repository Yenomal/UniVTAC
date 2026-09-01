"""UniVTAC adapter for OpenPI's VLM/FM streaming server pair."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from typing import Any
import uuid

from policy._base_policy import BasePolicy
from policy._openpi import build_observation
from policy._openpi import close_client
from policy._openpi import make_client
from policy._openpi import select_actions
import torch

logger = logging.getLogger(__name__)


class Policy(BasePolicy):
    def __init__(self, args: dict[str, Any]):
        super().__init__(args)
        host = args.get("host", "127.0.0.1")
        self._fm = make_client(host, int(args.get("fm_port", 8000)))
        self._fm_refresh = make_client(host, int(args.get("fm_port", 8000)))
        self._vlm = make_client(host, int(args.get("vlm_port", 8001)))
        self._side_camera = args.get("side_camera", "head")
        self._wrist_camera = args.get("wrist_camera", "wrist")
        self._state_dim = int(args.get("state_dim", 8))
        self._action_indices = args.get("action_indices")
        self._session_prefix = args.get("session_prefix", "univtac")
        self._num_steps = int(args.get("num_steps", 10))
        self._vlm_refresh_interval = int(args.get("vlm_refresh_interval_actions", 10))
        if self._num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self._vlm_refresh_interval <= 0:
            raise ValueError("vlm_refresh_interval_actions must be positive")

        self._cache_version: str | None = None
        self._cache_lock = threading.Lock()
        self._generation = 0
        self._refresh_queue: queue.Queue[tuple[int, dict[str, Any] | None]] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()
        self._session_id = ""
        self._last_refresh_execution_id = -self._vlm_refresh_interval

    def reset(self):
        with self._cache_lock:
            self._generation += 1
            self._cache_version = None
        self._clear_refresh_queue()
        self._session_id = f"{self._session_prefix}-{uuid.uuid4()}"
        self._last_refresh_execution_id = -self._vlm_refresh_interval
        self._fm.infer({"op": "reset_stream", "session_id": self._session_id})

    def eval(self, task, observation):
        request = build_observation(
            observation,
            task.instruction,
            side_camera=self._side_camera,
            wrist_camera=self._wrist_camera,
            state_dim=self._state_dim,
        )
        if not self._session_id:
            self.reset()
        with self._cache_lock:
            cache_version = self._cache_version
        if cache_version is None:
            self._refresh_cache(request, self._generation)
            self._last_refresh_execution_id = task.take_action_cnt
        elif task.take_action_cnt - self._last_refresh_execution_id >= self._vlm_refresh_interval:
            self._publish_refresh(request)
            self._last_refresh_execution_id = task.take_action_cnt

        response = self._stream_infer(request, task.take_action_cnt)
        action = select_actions(
            response,
            action_dim=len(request["state"]),
            action_indices=self._action_indices,
        )[0]
        task.take_action(torch.as_tensor(action, device=task.device), action_type="qpos")

    def close(self):
        self._stop_event.set()
        self._publish_refresh(None)
        self._refresh_thread.join(timeout=1.0)
        for client in (self._fm, self._fm_refresh, self._vlm):
            close_client(client)

    def _stream_infer(self, observation: dict[str, Any], execution_id: int) -> dict[str, Any]:
        for _ in range(2):
            with self._cache_lock:
                cache_version = self._cache_version
            if cache_version is None:
                raise RuntimeError("No active VLM cache")
            try:
                return self._fm.infer({
                    "op": "stream_infer",
                    "observation": observation,
                    "expected_cache_version": cache_version,
                    "session_id": self._session_id,
                    "executed_action_id": int(execution_id),
                    "num_steps": self._num_steps,
                })
            except RuntimeError as exc:
                if "Cache version mismatch" not in str(exc):
                    raise
        raise RuntimeError("VLM cache changed while submitting streaming inference")

    def _refresh_loop(self):
        while not self._stop_event.is_set():
            try:
                generation, observation = self._refresh_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if observation is not None:
                try:
                    self._refresh_cache(observation, generation)
                except Exception:
                    logger.exception("OpenPI VLM cache refresh failed; retaining the active cache")

    def _refresh_cache(self, observation: dict[str, Any], generation: int) -> None:
        encoded = self._vlm.infer({"op": "encode_prefix", "observation": observation})
        with self._cache_lock:
            if generation != self._generation:
                return
        refreshed = self._fm_refresh.infer({
            "op": "refresh_prefix",
            "cache_id": encoded["cache_id"],
            "cache_version": encoded["cache_version"],
        })
        with self._cache_lock:
            if generation == self._generation:
                self._cache_version = refreshed["cache_version"]

    def _publish_refresh(self, observation: dict[str, Any] | None) -> None:
        with self._cache_lock:
            generation = self._generation
        try:
            self._refresh_queue.put_nowait((generation, observation))
            return
        except queue.Full:
            pass
        with contextlib.suppress(queue.Empty):
            self._refresh_queue.get_nowait()
        self._refresh_queue.put_nowait((generation, observation))

    def _clear_refresh_queue(self) -> None:
        while True:
            try:
                self._refresh_queue.get_nowait()
            except queue.Empty:
                return
