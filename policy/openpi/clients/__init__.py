from __future__ import annotations

from typing import Any

from policy.openpi.clients.base import OpenPIBackend


def build_backend(client_type: str, args: dict[str, Any]) -> OpenPIBackend:
    if client_type == "single":
        from policy.openpi.clients.single import build
    elif client_type == "streaming":
        from policy.openpi.clients.streaming import build
    elif client_type == "runtime":
        from policy.openpi.clients.runtime import build
    elif client_type == "runtime_streaming":
        # Compatibility with existing deployment files.
        from policy.openpi.clients.runtime_streaming import build
    else:
        raise ValueError(
            f"Unsupported OpenPI client_type {client_type!r}; expected 'single', 'streaming', or 'runtime'"
        )
    return build(args)


__all__ = ["OpenPIBackend", "build_backend"]
