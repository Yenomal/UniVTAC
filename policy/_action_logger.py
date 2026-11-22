"""Record raw policy responses as one Parquet file per evaluation seed."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


class ActionLogger:
    def __init__(self, directory: str | Path):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Policy action logging requires pyarrow. Install it with: python -m pip install pyarrow"
            ) from exc

        self._pa = pa
        self._pq = pq
        self.directory = Path(directory)
        self._schema = pa.schema([
            ("seed", pa.int64()),
            ("request_index", pa.int64()),
            ("chunk_index", pa.int64()),
            ("step", pa.int64()),
            ("timestamp", pa.float64()),
            ("action_id", pa.int64()),
            ("action", pa.list_(pa.float64())),
        ])
        self._writer = None
        self._rows: list[dict[str, Any]] = []
        self._seed = 0
        self._request_index = 0

    def start_episode(self, seed: int) -> None:
        self.close()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._seed = int(seed)
        self._request_index = 0
        self._writer = self._pq.ParquetWriter(
            self.directory / f"{self._seed}.parquet", self._schema, compression="zstd"
        )

    def record(self, actions: Any, *, step: int, action_id: int | None = None) -> None:
        """Keep all returned dimensions, including every row of an action chunk."""
        if self._writer is None:
            raise RuntimeError("Start an action-log episode before recording actions")
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise ValueError(f"Policy actions must have shape (D,) or (T, D), got {actions.shape}")

        timestamp = time.time()
        for chunk_index, action in enumerate(actions):
            self._rows.append({
                "seed": self._seed,
                "request_index": self._request_index,
                "chunk_index": chunk_index,
                "step": int(step),
                "timestamp": timestamp,
                "action_id": action_id,
                "action": action.tolist(),
            })
            if len(self._rows) >= 256:
                self._flush()
        self._request_index += 1

    def _flush(self) -> None:
        if self._rows:
            table = self._pa.Table.from_pylist(self._rows, schema=self._schema)
            self._writer.write_table(table)
            self._rows.clear()

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._flush()
            finally:
                self._writer.close()
                self._writer = None


def make_action_logger(policy_name: str, save_root: str | Path) -> ActionLogger | None:
    if policy_name in {"openpi", "SmolVLA"}:
        return ActionLogger(Path(save_root) / "actions")
    return None
