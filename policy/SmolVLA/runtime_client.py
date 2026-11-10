"""Synchronous client for LeRobot's standalone observation/action Runtime.

This module deliberately contains only transport and wire validation. It does not
load a checkpoint or know how the server schedules its action queue.
"""

from __future__ import annotations

import math
import socket
import struct
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

PROTOCOL_VERSION = 1
ENVELOPE_VERSION = 1
MAX_FRAME_BYTES = 64 * 1024 * 1024
_MARKER = "__lerobot_type__"


def _require_msgpack():
    try:
        import msgpack
    except ImportError as exc:
        raise ImportError(
            "The Runtime client needs msgpack. Install the remote extra or add msgpack to the UniVTAC environment."
        ) from exc
    return msgpack


def _array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, copy=True, order="C")
    if array.dtype.name not in {"bool", "uint8", "int32", "int64", "float32", "float64"}:
        raise ValueError(f"Unsupported Runtime array dtype: {array.dtype}")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("Runtime arrays must contain finite values")
    return array


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = _array(value)
        return {
            _MARKER: "tensor",
            "dtype": array.dtype.name,
            "shape": list(array.shape),
            "endianness": "little",
            "data": array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C"),
        }
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            raise ValueError("Runtime scalar must be finite")
        return float(value)
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get(_MARKER) == "tensor":
            dtype = np.dtype(value["dtype"]).newbyteorder("<")
            shape = tuple(int(dim) for dim in value["shape"])
            expected = math.prod(shape) * dtype.itemsize
            data = value["data"]
            if len(data) != expected:
                raise ValueError("Runtime tensor payload size does not match its shape")
            return np.frombuffer(data, dtype=dtype).reshape(shape).astype(dtype.newbyteorder("="), copy=True)
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _message(message_type: str, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": ENVELOPE_VERSION,
        "message_type": message_type,
        "instance_id": "runtime",
        "session_id": session_id,
        "generation": 0,
        "request_id": uuid4().hex,
        "body": {"protocol_version": PROTOCOL_VERSION, **body},
    }


def _pack(message: dict[str, Any]) -> bytes:
    msgpack = _require_msgpack()
    payload = msgpack.packb(_encode(message), use_bin_type=True)
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("Runtime message is too large")
    return struct.pack(">I", len(payload)) + payload


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Runtime server closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv(sock: socket.socket) -> dict[str, Any]:
    msgpack = _require_msgpack()
    size = struct.unpack(">I", _read_exact(sock, 4))[0]
    if size <= 0 or size > MAX_FRAME_BYTES:
        raise ValueError("Invalid Runtime frame size")
    decoded = msgpack.unpackb(_read_exact(sock, size), raw=False, strict_map_key=True)
    return _decode(decoded)


class RuntimeClient:
    """Blocking observation/action client with explicit action acknowledgements."""

    def __init__(self, host: str, port: int, *, timeout_s: float = 10.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.session_id = f"client-{uuid4().hex}"
        self._socket: socket.socket | None = None

    def connect(self) -> None:
        if self._socket is None:
            self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            self._socket.settimeout(self.timeout_s)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        self.connect()
        assert self._socket is not None
        self._socket.sendall(_pack(message))
        response = _recv(self._socket)
        if (
            response.get("version") != ENVELOPE_VERSION
            or response.get("instance_id") != "runtime"
            or response.get("session_id") != message["session_id"]
            or response.get("request_id") != message["request_id"]
            or response.get("generation") != message["generation"]
        ):
            raise ValueError("Runtime response correlation does not match its request")
        body = response.get("body", {})
        if body.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported Runtime protocol version in response")
        if response.get("message_type") == "error":
            raise RuntimeError(body.get("message", "Runtime server error"))
        return body

    def reset(self) -> None:
        body = self._request(_message("control", self.session_id, {"op": "reset"}))
        if body.get("op") != "reset":
            raise RuntimeError(f"Unexpected Runtime reset reply: {body!r}")

    def observe(self, features: Mapping[str, Any], *, task: str, sequence: int) -> dict[str, Any]:
        if type(sequence) is not int or sequence < 0:
            raise ValueError("Runtime observation sequence must be a non-negative integer")
        body = self._request(
            _message(
                "observation",
                self.session_id,
                {"op": "observe", "sequence": sequence, "features": dict(features), "task": task},
            )
        )
        if body.get("op") != "action":
            raise RuntimeError(f"Unexpected Runtime observation reply: {body!r}")
        action = _array(body.get("action"))
        if action.ndim != 1 or action.dtype.name != "float32":
            raise ValueError(f"Runtime action must be one-dimensional float32, got {action.shape}, {action.dtype}")
        if type(body.get("action_id")) is not int or body["action_id"] < 0:
            raise ValueError("Runtime action_id is missing or invalid")
        body["action"] = action
        return body

    def acknowledge(self, action_id: int) -> None:
        body = self._request(_message("control", self.session_id, {"op": "ack", "action_id": int(action_id)}))
        if body.get("op") != "ack":
            raise RuntimeError(f"Unexpected Runtime acknowledgement reply: {body!r}")
