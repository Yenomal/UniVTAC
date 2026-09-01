"""UniVTAC deployment adapter for the GR00T N1.7 OXE-DROID test schema."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from .._base_policy import BasePolicy
from .groot_test_client import Gr00tTestClient
from .state_action_adapter import decode_qpos, encode_observation
from .streaming_controller import StreamingController


class Policy(BasePolicy):
    def __init__(self, args):
        super().__init__(args)
        self.task_name = args.get("task_name", "")
        self.client = Gr00tTestClient(
            host=args.get("host", "127.0.0.1"),
            port=int(args.get("port", 5555)),
            task=args.get("language", ""),
            timeout_ms=int(args.get("timeout_ms", 60000)),
            state_dims={
                "eef_9d": 9,
                "gripper_position": 1,
                "joint_position": 7,
            },
        )
        if not self.client.ping():
            raise RuntimeError("Cannot connect to the GR00T decoupled server")
        self.controller = StreamingController(
            self.client,
            slide_steps=int(args.get("slide_steps", 1)),
        )

    def eval(self, task, observation):
        video, state = encode_observation(task, observation)
        action = self.controller.step(video, state, task.instruction)
        qpos = decode_qpos(task, action, index=0).to(task.device)
        task.take_action(qpos, action_type="qpos")

    def reset(self):
        self.controller.reset()

    def close(self):
        self.controller.close()

