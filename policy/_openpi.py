"""Shared OpenPI wire adapter for UniVTAC policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import numpy as np


def make_client(host: str, port: int) -> Any:
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError as exc:
        client_source = Path(__file__).resolve().parents[1] / "third_party" / "openpi_dexhand" / "packages" / "openpi-client" / "src"
        if not client_source.is_dir():
            raise ImportError(
                "OpenPI client is unavailable. Install the openpi-client package in the UniVTAC environment."
            ) from exc
        sys.path.insert(0, str(client_source))
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    return WebsocketClientPolicy(host, port)


def build_observation(
    observation: Mapping[str, Any],
    prompt: str,
    *,
    side_camera: str,
    wrist_camera: str,
    state_dim: int,
    include_tactile: bool = False,
    left_tactile: str = "left_tactile",
    right_tactile: str = "right_tactile",
    marker_count: int | None = None,
) -> dict[str, Any]:
    if state_dim <= 0:
        raise ValueError("state_dim must be positive")
    try:
        images = {
            "cam_side": _as_numpy(observation["observation"][side_camera]["rgb"], np.uint8),
            "cam_wrist": _as_numpy(observation["observation"][wrist_camera]["rgb"], np.uint8),
        }
        state = _as_numpy(observation["embodiment"]["joint"], np.float32).reshape(-1)[:state_dim]
    except KeyError as exc:
        raise ValueError(f"UniVTAC observation is missing {exc.args[0]!r}") from exc
    if state.shape != (state_dim,):
        raise ValueError(f"UniVTAC state must have at least {state_dim} dimensions, got {state.shape}")
    request = {"images": images, "state": state, "prompt": prompt}
    if include_tactile:
        try:
            left_marker = _as_numpy(observation["tactile"][left_tactile]["marker"], np.float32)
            right_marker = _as_numpy(observation["tactile"][right_tactile]["marker"], np.float32)
        except KeyError as exc:
            raise ValueError(f"UniVTAC tactile observation is missing {exc.args[0]!r}") from exc
        _validate_marker(left_marker, "left_marker", marker_count)
        _validate_marker(right_marker, "right_marker", marker_count)
        if left_marker.shape != right_marker.shape:
            raise ValueError(
                "UniVTAC left and right tactile markers must have the same shape, "
                f"got {left_marker.shape} and {right_marker.shape}"
            )
        request["left_marker"] = left_marker
        request["right_marker"] = right_marker
    return request


def select_actions(
    response: Mapping[str, Any],
    *,
    action_dim: int,
    action_indices: Sequence[int] | None,
) -> np.ndarray:
    try:
        actions = _as_numpy(response["actions"], np.float32)
    except KeyError as exc:
        raise ValueError("OpenPI response is missing 'actions'") from exc
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(f"OpenPI actions must have shape (T, D), got {actions.shape}")
    if action_indices is not None:
        if len(action_indices) != action_dim:
            raise ValueError("action_indices must contain exactly action_dim entries")
        actions = actions[:, list(action_indices)]
    if actions.shape[1] != action_dim:
        raise ValueError(
            f"OpenPI returned {actions.shape[1]} action dimensions, but UniVTAC requires {action_dim}. "
            "Set action_indices only when the checkpoint has a documented Panda mapping."
        )
    if not np.isfinite(actions).all():
        raise ValueError("OpenPI actions contain non-finite values")
    return actions


def compress_qpos(actions: np.ndarray) -> np.ndarray:
    """Convert UniVTAC's 9D arm+two-finger qpos to its 8D control qpos."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 9:
        raise ValueError(f"UniVTAC model actions must have shape (T, 9), got {actions.shape}")
    return actions[:, :8]


def stack_tactile_history(
    history: Sequence[Any],
    *,
    history_size: int,
    marker_count: int | None = None,
) -> np.ndarray:
    """Stack tactile marker frames oldest-to-newest with stable length."""
    if history_size <= 0:
        raise ValueError("history_size must be positive")
    if not history:
        raise ValueError("tactile history must contain at least one frame")
    frames = []
    for frame in history:
        array = _as_numpy(frame, np.float32)
        _validate_marker(array, "tactile_marker", marker_count)
        frames.append(array)
    frames = frames[-history_size:]
    if len(frames) < history_size:
        frames = [frames[0]] * (history_size - len(frames)) + frames
    return np.stack(frames, axis=0)


def close_client(client: Any) -> None:
    websocket = getattr(client, "_ws", None)
    if websocket is not None:
        websocket.close()


def is_connection_error(exc: BaseException) -> bool:
    """Return whether an inference failure indicates a broken transport."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return type(exc).__name__ in {
        "ConnectionClosed",
        "ConnectionClosedError",
        "ConnectionClosedOK",
    }


def _as_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=dtype)
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, dtype=dtype, copy=True, order="C")
    return array


def _validate_marker(marker: np.ndarray, name: str, marker_count: int | None) -> None:
    if marker.ndim != 3 or marker.shape[0] != 2 or marker.shape[-1] != 2:
        raise ValueError(f"UniVTAC {name} must have shape (2, N, 2), got {marker.shape}")
    if marker_count is not None and marker.shape[1] != marker_count:
        raise ValueError(
            f"UniVTAC {name} must contain {marker_count} markers for this checkpoint, "
            f"got {marker.shape[1]}"
        )
    if not np.isfinite(marker).all():
        raise ValueError(f"UniVTAC {name} contains non-finite values")
