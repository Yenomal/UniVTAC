# OpenPI Adapters

`openpi` uses the ordinary one-server `infer` endpoint once per UniVTAC control tick.

`streaming_openpi` uses the VLM/FM server pair. It refreshes VLM cache in a background thread every `vlm_refresh_interval_actions` executed actions and sends the UniVTAC action counter as `executed_action_id` to FM's `stream_infer` endpoint.

Both adapters use UniVTAC's 8-D Panda qpos convention: seven arm joints and one gripper value. A compatible OpenPI checkpoint must return the same action space; otherwise declare the documented `action_indices` mapping in `deploy.yml`.

Start `openpi` against the ordinary OpenPI server. Start `streaming_openpi` against `scripts/serve_policy.py --multi-process`, with FM on `fm_port` and VLM on `vlm_port`.

```bash
bash eval_policy.sh lift_bottle demo openpi/deploy 0
bash eval_policy.sh lift_bottle demo streaming_openpi/deploy 0
```
