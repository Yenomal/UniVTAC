# SmolVLA Runtime client

This UniVTAC policy adapter sends one current observation to the standalone LeRobot Runtime server and waits for one action. The server owns checkpoint loading and action-queue playback; this client only maps UniVTAC observations, checks the returned action, executes it, and acknowledges it.

Configure the server checkpoint's observation feature names in `deploy_runtime.yml`. The defaults are `observation.state`, `observation.images.image`, and `observation.images.image2`; set `wrist_feature: null` for a single-camera checkpoint. Camera values are sent as RGB HWC `uint8` arrays, and state is a flat `float32` vector.

Start the server on the model host:

```bash
uv run --extra remote lerobot-runtime-server \
  --checkpoint=/data/checkpoints/my_smolvla \
  --device=cuda:0 \
  --host=0.0.0.0 \
  --port=8000 \
  --runtime-queue-size=64
```

Run a UniVTAC evaluation on the robot/simulation host:

```bash
bash eval_policy.sh pull_out_key openpi SmolVLA/deploy_runtime 0
```

The final argument is UniVTAC's simulation GPU selection. It is independent from the Runtime server's `--device` setting. The client environment needs NumPy, PyTorch, PyYAML, `msgpack`, and `pyarrow`.

The current Runtime server creates an independent session for each client's
`session_id`. Parallel evaluation workers each create their own client, and can
connect to the same Runtime host and port:

```bash
bash parallel_eval.sh lift_bottle yenomal SmolVLA/deploy_runtime 0 4 50 --run_name my_run
```

Parallel simulation workers run headless with camera rendering enabled and
livestreaming disabled, avoiding conflicts on Isaac Sim's HTTP and streaming ports.

Headless workers explicitly enable PhysX scene queries, which TacEx needs to
attach the tactile gelpads to the moving gripper. Camera rendering alone does
not enable these queries.

Evaluation saves every raw policy action to `actions/<seed>.parquet` inside the
evaluation result directory, before dimension mapping or gripper compression.
See [Policy action logs](../../docs/Deploy.md#policy-action-logs) for the schema.
