"""Temporary pi-r2 compatible client used by the UniVTAC integration test.

The model and processor stay in the GR00T server process.  This module only
implements the wire protocol and the decoupled VLM/Action-Head endpoints.
It intentionally avoids importing the ``gr00t`` Python package because that
would pull the server's torch/transformers environment into IsaacSim.
"""

from __future__ import annotations

import io
import threading
from typing import Any

import msgpack
import numpy as np
import zmq


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    if "__ModalityConfig_class__" in obj:
        # Keep the client independent of gr00t.data.types.
        return obj.get("as_json", {})
    return obj


def _pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode)


def _unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode)


class Gr00tTestClient:
    """Small client for the pi-r2 decoupled server contract.

    ``video_history`` is a mapping from server video key to ``(T,H,W,3)``
    uint8 arrays. ``state_history`` is a flat ``(T,D)`` float32 array whose
    columns follow ``state_keys`` and ``state_dims``.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        *,
        task: str = "",
        timeout_ms: int = 60000,
        state_dims: dict[str, int] | None = None,
    ):
        self.task = task
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")
        self._lock = threading.Lock()

        self.modality_config = self._call("get_modality_config", requires_input=False)
        video_cfg = self.modality_config.get("video", {})
        state_cfg = self.modality_config.get("state", {})
        language_cfg = self.modality_config.get("language", {})
        self.video_delta_indices = list(video_cfg.get("delta_indices", [0]))
        self.state_delta_indices = list(state_cfg.get("delta_indices", [0]))
        self.video_keys = list(video_cfg.get("modality_keys", []))
        self.state_keys = list(state_cfg.get("modality_keys", []))
        self.language_keys = list(language_cfg.get("modality_keys", ["task"]))
        self.state_dims = state_dims or {}

    @property
    def video_history_length(self) -> int:
        return len(self.video_delta_indices)

    @property
    def state_history_length(self) -> int:
        return len(self.state_delta_indices)

    @property
    def expected_state_dim(self) -> int:
        missing = [key for key in self.state_keys if key not in self.state_dims]
        if missing:
            raise KeyError(f"Missing state dimensions for server keys: {missing}")
        return sum(self.state_dims[key] for key in self.state_keys)

    def _build_state_dict(self, state_history: np.ndarray) -> dict[str, np.ndarray]:
        state_history = np.asarray(state_history, dtype=np.float32)
        if state_history.ndim != 2:
            raise ValueError(f"state_history must be (T,D), got {state_history.shape}")
        if state_history.shape[-1] != self.expected_state_dim:
            raise ValueError(
                f"state dimension {state_history.shape[-1]} != {self.expected_state_dim}"
            )
        result = {}
        offset = 0
        for key in self.state_keys:
            dim = self.state_dims[key]
            result[key] = state_history[:, offset : offset + dim][None]
            offset += dim
        return result

    def _build_observation(
        self,
        video_history: dict[str, np.ndarray],
        state_history: np.ndarray,
        language: str | None,
    ) -> dict[str, Any]:
        if set(video_history) != set(self.video_keys):
            raise ValueError(
                f"video keys {sorted(video_history)} != server keys {sorted(self.video_keys)}"
            )
        video = {}
        for key in self.video_keys:
            frames = np.asarray(video_history[key])
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(f"video[{key}] must be (T,H,W,3), got {frames.shape}")
            video[key] = frames.astype(np.uint8, copy=False)[None]
        text = self.task if language is None else language
        return {
            "video": video,
            "state": self._build_state_dict(state_history),
            "language": {key: [[text]] for key in self.language_keys},
        }

    def _call(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        *,
        requires_input: bool = True,
    ) -> Any:
        request = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        payload = _pack(request)
        with self._lock:
            self.socket.send(payload)
            response = _unpack(self.socket.recv())
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    @staticmethod
    def _parse_response(response: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if isinstance(response, (list, tuple)):
            action = response[0]
            info = response[1] if len(response) > 1 else {}
        else:
            action, info = response, {}
        action = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in action.items()
        }
        # Server returns (B,T,D); expose (T,D) to the UniVTAC controller.
        for key, value in list(action.items()):
            if value.ndim == 3 and value.shape[0] == 1:
                action[key] = value[0]
        return action, info if isinstance(info, dict) else {}

    def ping(self) -> bool:
        try:
            self._call("ping", requires_input=False)
            return True
        except Exception:
            return False

    def update_vlm_cache(
        self,
        video_history: dict[str, np.ndarray],
        state_history: np.ndarray,
        language: str | None = None,
        *,
        t_image_capture: float | None = None,
    ) -> dict[str, Any]:
        observation = self._build_observation(video_history, state_history, language)
        data: dict[str, Any] = {"observation": observation}
        if t_image_capture is not None:
            data["t_image_capture"] = float(t_image_capture)
        return self._call("update_vlm_cache", data=data)

    def seed_streaming_from_obs(
        self,
        video_history: dict[str, np.ndarray],
        state_history: np.ndarray,
        language: str | None = None,
        *,
        slide_steps: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observation = self._build_observation(video_history, state_history, language)
        data: dict[str, Any] = {"observation": observation}
        if slide_steps is not None:
            data["slide_steps"] = int(slide_steps)
        response = self._call("seed_streaming_from_obs", data=data)
        return self._parse_response(response)

    def get_action_chunk_cached(
        self,
        state_history: np.ndarray,
        *,
        slide_steps: int | None = None,
        inpaint_actions: np.ndarray | None = None,
        force_nonstreaming: bool = False,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        options: dict[str, Any] = {}
        if slide_steps is not None:
            options["slide_steps"] = int(slide_steps)
        if force_nonstreaming:
            options["force_nonstreaming"] = True
        if inpaint_actions is not None:
            inpaint_actions = np.asarray(inpaint_actions, dtype=np.float32)
            if inpaint_actions.ndim != 2:
                raise ValueError("inpaint_actions must have shape (N,D)")
            options["inpaint"] = inpaint_actions
            options["action_horizon"] = int(inpaint_actions.shape[0])
        response = self._call(
            "get_action_chunk_cached",
            data={
                "observation": {"state": self._build_state_dict(state_history)},
                "options": options or None,
            },
        )
        return self._parse_response(response)

    def reset_streaming_buffer(self) -> dict[str, Any]:
        return self._call("reset_streaming_buffer", requires_input=False)

    def reset(self) -> dict[str, Any]:
        return self._call("reset", data={"options": None})

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()

