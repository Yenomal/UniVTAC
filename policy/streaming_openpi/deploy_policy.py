"""UniVTAC adapter for OpenPI's VLM/FM streaming server pair."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from typing import Any
import uuid

import numpy as np
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
        self._ports = {
            "fm": int(args.get("fm_port", 8000)),
            "fm_refresh": int(args.get("fm_port", 8000)),
            "vlm": int(args.get("vlm_port", 8001)),
        }
        self._fm = make_client(self._host, self._ports["fm"])
        self._fm_refresh = make_client(self._host, self._ports["fm_refresh"])
        self._vlm = make_client(self._host, self._ports["vlm"])
        self._side_camera = args.get("side_camera", "head")
        self._wrist_camera = args.get("wrist_camera", "wrist")
        self._state_dim = int(args.get("state_dim", 8))
        self._action_indices = args.get("action_indices")
        self._use_tactile = bool(args.get("use_tactile", False))
        self._left_tactile = args.get("left_tactile", "left_tactile")
        self._right_tactile = args.get("right_tactile", "right_tactile")
        marker_count = args.get("marker_count")
        self._marker_count = int(marker_count) if marker_count is not None else None
        self._session_prefix = args.get("session_prefix", "univtac")
        self._num_steps = int(args.get("num_steps", 10))
        self._horizon = int(args.get("horizon", 50))
        self._watermark = int(args.get("watermark", 30))
        self._chunk_size = int(args.get("streaming_chunk_size", 5))
        self._reconnect_attempts = int(args.get("reconnect_attempts", 1))
        if self._num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self._horizon <= 0:
            raise ValueError("horizon must be positive")
        if self._watermark < 0 or self._watermark >= self._horizon:
            raise ValueError("watermark must be in [0, horizon)")
        if self._chunk_size <= 0 or self._horizon % self._chunk_size:
            raise ValueError("streaming_chunk_size must be positive and divide horizon")
        if self._reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be non-negative")
        if self._marker_count is not None and self._marker_count <= 0:
            raise ValueError("marker_count must be positive")
        self._refresh_after = self._horizon - self._watermark

        self._cache_version: str | None = None
        self._cache_lock = threading.Lock()
        self._refresh_rpc_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._timing = {}
        self._generation = 0
        self._refresh_queue: queue.Queue[tuple[int, dict[str, Any] | None]] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()
        self._session_id = ""
        self._last_refresh_execution_id = None
        self._action_buffer = np.empty((0, 8), dtype=np.float32)

    def reset(self):
        with self._cache_lock:
            self._generation += 1
            self._cache_version = None
        with self._metrics_lock:
            self._timing.clear()
        self._clear_refresh_queue()
        self._session_id = f"{self._session_prefix}-{uuid.uuid4()}"
        self._last_refresh_execution_id = None
        self._action_buffer = np.empty((0, 8), dtype=np.float32)
        with self._refresh_rpc_lock:
            response = self._timed_infer("fm", {
                "op": "reset_stream",
                "session_id": self._session_id,
                "clear_prefix_cache": True,
            })
        if response.get("prefix_cache_cleared") is not True:
            raise RuntimeError("FM did not clear its prefix and streaming caches")

    def needs_observation(self, task) -> bool:
        with self._cache_lock:
            cache_missing = self._cache_version is None
        if cache_missing or self._action_buffer is None or len(self._action_buffer) == 0:
            return True
        if self._last_refresh_execution_id is None:
            return True
        return task.take_action_cnt - self._last_refresh_execution_id >= self._refresh_after

    def eval(self, task, observation):
        if not self._session_id:
            self.reset()
        request = None
        if observation is not None:
            request = build_observation(
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
        with self._cache_lock:
            cache_version = self._cache_version
            generation = self._generation
        if cache_version is None:
            if request is None:
                raise ValueError("Streaming OpenPI requires an observation for initialisation")
            self._refresh_cache(request, generation, initial=True)
            self._last_refresh_execution_id = task.take_action_cnt
        elif (
            self._last_refresh_execution_id is None
            or task.take_action_cnt - self._last_refresh_execution_id >= self._refresh_after
        ):
            if request is None:
                raise ValueError("Streaming OpenPI requires an observation for VLM refresh")
            self._publish_refresh(request)
            self._last_refresh_execution_id = task.take_action_cnt

        if self._action_buffer is None or len(self._action_buffer) == 0:
            if request is None:
                raise ValueError("Streaming OpenPI requires an observation for FM inference")
            try:
                response = self._stream_infer(request, task.take_action_cnt)
            except RuntimeError as exc:
                if not self._is_stream_state_error(exc):
                    raise
                logger.warning("OpenPI FM stream state was lost; rebuilding the stream", exc_info=True)
                self.reset()
                with self._cache_lock:
                    generation = self._generation
                self._refresh_cache(request, generation, initial=True)
                self._last_refresh_execution_id = task.take_action_cnt
                response = self._stream_infer(request, task.take_action_cnt)
            self._action_buffer = self._parse_stream_actions(response, task.take_action_cnt)
        action = self._action_buffer[0]
        self._action_buffer = self._action_buffer[1:]
        task.take_action(torch.as_tensor(action, device=task.device), action_type="qpos")
        task.metadata["policy_timing"] = self.timing_snapshot()

    def close(self):
        self._stop_event.set()
        self._publish_refresh(None)
        self._refresh_thread.join(timeout=1.0)
        for client in (self._fm, self._fm_refresh, self._vlm):
            close_client(client)

    def timing_snapshot(self):
        with self._metrics_lock:
            return {name: values.copy() for name, values in self._timing.items()}

    def _stream_infer(self, observation: dict[str, Any], execution_id: int) -> dict[str, Any]:
        with self._cache_lock:
            cache_version = self._cache_version
        if cache_version is None:
            raise RuntimeError("No active VLM cache")
        request = {
            "op": "stream_infer",
            "observation": observation,
            "expected_cache_version": cache_version,
            "session_id": self._session_id,
            "executed_action_id": int(execution_id),
            "num_steps": self._num_steps,
        }
        try:
            return self._timed_infer("fm", request)
        except RuntimeError as exc:
            if "Cache version mismatch" not in str(exc):
                raise
            with self._cache_lock:
                current_version = self._cache_version
            if current_version is None or current_version == cache_version:
                raise
            request["expected_cache_version"] = current_version
            return self._timed_infer("fm", request)

    def _parse_stream_actions(self, response: dict[str, Any], execution_id: int) -> np.ndarray:
        streaming = response.get("streaming")
        if not isinstance(streaming, dict):
            raise ValueError("FM streaming response is missing metadata")
        if streaming.get("session_id") != self._session_id:
            raise ValueError("FM streaming response has an unexpected session_id")
        if streaming.get("execution_id") != execution_id:
            raise ValueError("FM streaming response has an unexpected execution_id")
        if streaming.get("action_count") != self._chunk_size:
            raise ValueError(
                f"FM returned chunk size {streaming.get('action_count')}, expected {self._chunk_size}"
            )
        cache_version = streaming.get("cache_version")
        if not isinstance(cache_version, str) or not cache_version:
            raise ValueError("FM streaming response is missing cache_version")
        actions = select_actions(
            response,
            action_dim=self._state_dim,
            action_indices=self._action_indices,
        )
        if len(actions) != self._chunk_size:
            raise ValueError("FM streaming action count does not match metadata")
        with self._cache_lock:
            self._cache_version = cache_version
        return compress_qpos(actions)

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

    def _refresh_cache(self, observation: dict[str, Any], generation: int, initial: bool = False) -> None:
        with self._refresh_rpc_lock:
            with self._cache_lock:
                if generation != self._generation:
                    return
            encoded = self._timed_infer("vlm", {"op": "encode_prefix", "observation": observation})
            with self._cache_lock:
                if generation != self._generation:
                    return
            refreshed = self._timed_infer("fm_refresh", {
                "op": "refresh_prefix",
                "cache_id": encoded["cache_id"],
                "cache_version": encoded["cache_version"],
            })
            accepted = {
                refreshed.get("active_cache_version"),
                refreshed.get("pending_cache_version"),
                refreshed.get("cache_version"),
            }
            if encoded.get("cache_version") not in accepted:
                raise RuntimeError("FM rejected the refreshed VLM prefix cache")
            active_version = refreshed.get("active_cache_version", refreshed.get("cache_version"))
            if initial and active_version != encoded.get("cache_version"):
                raise RuntimeError("FM did not activate the initial VLM prefix cache")
            with self._cache_lock:
                if generation == self._generation:
                    self._cache_version = active_version

    def _timed_infer(self, role: str, request: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        try:
            response = self._infer_with_reconnect(role, request)
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            operation = request.get("op", "infer")
            self._record_timing(f"{role}_{operation}", elapsed_ms)
        if not isinstance(response, dict):
            raise TypeError(f"OpenPI {role} returned {type(response).__name__}, expected dict")
        for timing_group in ("server_timing", "policy_timing"):
            timing = response.get(timing_group)
            if isinstance(timing, dict):
                for name, value in timing.items():
                    if isinstance(value, (int, float)):
                        self._record_timing(f"{role}_{timing_group}_{name}", float(value))
        return response

    def _record_timing(self, name: str, elapsed_ms: float) -> None:
        with self._metrics_lock:
            metric = self._timing.setdefault(
                name,
                {"count": 0, "last_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0},
            )
            count = metric["count"] + 1
            metric["count"] = count
            metric["last_ms"] = elapsed_ms
            metric["mean_ms"] += (elapsed_ms - metric["mean_ms"]) / count
            metric["max_ms"] = max(metric["max_ms"], elapsed_ms)

    def _infer_with_reconnect(self, role: str, request: dict[str, Any]) -> dict[str, Any]:
        attribute = f"_{role}"
        for attempt in range(self._reconnect_attempts + 1):
            client = getattr(self, attribute)
            try:
                return client.infer(request)
            except Exception as exc:
                if attempt >= self._reconnect_attempts or not is_connection_error(exc):
                    raise
                logger.warning("OpenPI %s connection failed; reconnecting", role, exc_info=True)
                with contextlib.suppress(Exception):
                    close_client(client)
                setattr(self, attribute, make_client(self._host, self._ports[role]))
        raise AssertionError("unreachable")

    @staticmethod
    def _is_stream_state_error(exc: RuntimeError) -> bool:
        message = str(exc)
        return any(
            marker in message
            for marker in (
                "No active prefix cache",
                "No active prefix",
                "A different stream session is active",
            )
        )

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
