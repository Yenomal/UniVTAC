from __future__ import annotations

from typing import Any

from policy.streaming_openpi.deploy_policy import Policy


def build(args: dict[str, Any]) -> Policy:
    return Policy(args)
