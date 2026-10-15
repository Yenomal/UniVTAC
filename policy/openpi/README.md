# OpenPI

统一策略入口由 `client_type` 选择后端。`single` 使用单服务端 `infer` 请求和 watermark 动作缓冲；`streaming` 使用 VLM/FM 服务端、prefix cache 和 FM chunk。两种后端共用 UniVTAC 的 OpenPI 观测与动作适配工具。

streaming 配置使用一个 `history_size` 同时控制 state 历史窗口和启用 tactile 历史时的左右 marker 窗口；旧的 `tactile_history_size` 仍可作为兼容回退项。

```bash
bash eval_policy.sh lift_bottle demo openpi/deploy_single 0
bash eval_policy.sh lift_bottle demo openpi/deploy_streaming 0
```

single 配置使用 `port`。streaming 配置使用 `fm_port`、`vlm_port`、`horizon` 和 `streaming_chunk_size`。两种模式均可设置 `use_tactile`、`marker_count` 和 `tactile_sensor_type`。
