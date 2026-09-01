"""Async VLM refresh and pi-r2 Action-Head scheduling for the test adapter."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np

from .groot_test_client import Gr00tTestClient
from .schema import VIDEO_DELTA_INDICES


class _VLMWorker:
    def __init__(self, client: Gr00tTestClient):
        self.client = client
        self._condition = threading.Condition()
        self._pending: tuple[dict[str, np.ndarray], np.ndarray, str] | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="gr00t-vlm-refresh", daemon=True)
        self._thread.start()

    def submit(self, video: dict[str, np.ndarray], state: np.ndarray, language: str) -> None:
        with self._condition:
            self._pending = (video, state, language)
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                video, state, language = self._pending
                self._pending = None
            try:
                self.client.update_vlm_cache(video, state, language, t_image_capture=time.time())
            except Exception as exc:  # Keep the control loop alive; next submit retries.
                print(f"[GR00T] async VLM refresh failed: {exc}")

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify()
        self._thread.join(timeout=2.0)


class StreamingController:
    def __init__(self, client: Gr00tTestClient, *, slide_steps: int = 1):
        self.client = client
        self.slide_steps = slide_steps
        self._video_history: dict[str, deque[np.ndarray]] = {}
        self._state_history: deque[np.ndarray] = deque(maxlen=1)
        self._last_action: np.ndarray | None = None
        self._seeded = False
        self._vlm_worker = _VLMWorker(client)

    def reset(self) -> None:
        self._video_history.clear()
        self._state_history.clear()
        self._last_action = None
        self._seeded = False
        self.client.reset_streaming_buffer()

    def _append_history(self, video: dict[str, np.ndarray], state: np.ndarray) -> None:
        for key, frame in video.items():
            history = self._video_history.setdefault(
                key, deque(maxlen=max(abs(i) for i in VIDEO_DELTA_INDICES) + 1)
            )
            history.append(frame)
        self._state_history.append(state)

    def _sample_video(self) -> dict[str, np.ndarray]:
        sampled = {}
        for key, history in self._video_history.items():
            frames = list(history)
            if not frames:
                raise RuntimeError("RGB history is empty")
            sampled[key] = np.stack(
                [frames[max(0, len(frames) + index - 1)] for index in VIDEO_DELTA_INDICES],
                axis=0,
            )
        return sampled

    def step(self, video: dict[str, np.ndarray], state: np.ndarray, language: str) -> dict[str, np.ndarray]:
        self._append_history(video, state)
        sampled_video = self._sample_video()
        state_history = np.stack(list(self._state_history), axis=0)

        if not self._seeded:
            action, _ = self.client.seed_streaming_from_obs(
                sampled_video,
                state_history,
                language,
                slide_steps=self.slide_steps,
            )
            self._seeded = True
            return action

        # The seed call already primes the first VLM cache. Refresh it in the
        # background from this point onward while the control loop uses the
        # latest completed cache for Action Head calls.
        self._vlm_worker.submit(sampled_video, state_history, language)
        action, _ = self.client.get_action_chunk_cached(
            state_history,
            slide_steps=self.slide_steps,
            inpaint_actions=None,
        )
        return action

    def close(self) -> None:
        self._vlm_worker.close()
        self.client.close()
