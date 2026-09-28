"""Unified entry point for single-server and streaming OpenPI clients."""

from __future__ import annotations

from typing import Any

from policy._base_policy import BasePolicy
from policy.openpi.clients import build_backend


class Policy(BasePolicy):
    def __init__(self, args: dict[str, Any]):
        super().__init__(args)
        client_type = args.get("client_type", "single")
        self._client_type = str(client_type).lower()
        self._backend = build_backend(self._client_type, args)

    def reset(self):
        self._backend.reset()

    def needs_observation(self, task) -> bool:
        return self._backend.needs_observation(task)

    def eval(self, task, observation):
        self._backend.eval(task, observation)

    def close(self):
        self._backend.close()
