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
        client_source = Path(__file__).resolve().parents[3] / "packages" / "openpi-client" / "src"
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
    return {"images": images, "state": state, "prompt": prompt}


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


def close_client(client: Any) -> None:
    websocket = getattr(client, "_ws", None)
    if websocket is not None:
        websocket.close()


def _as_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))
