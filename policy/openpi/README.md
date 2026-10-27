# OpenPI

统一策略入口由 `client_type` 选择后端。`single` 使用单服务端 `infer` 请求和 watermark 动作缓冲；`streaming` 使用 VLM/FM 服务端、prefix cache 和 FM chunk。两种后端共用 UniVTAC 的 OpenPI 观测与动作适配工具。

streaming 配置使用一个 `history_size` 同时控制 state 历史窗口和启用 tactile 历史时的左右 marker 窗口；旧的 `tactile_history_size` 仍可作为兼容回退项。

```bash
bash eval_policy.sh lift_bottle demo openpi/deploy_single 0
bash eval_policy.sh lift_bottle demo openpi/deploy_streaming 0
bash eval_policy.sh pull_out_key openpi openpi/deploy_runtime 0
```

single 配置使用 `port`。streaming 配置使用 `fm_port`、`vlm_port`、`horizon` 和 `streaming_chunk_size`。两种模式均可设置 `use_tactile`、`marker_count` 和 `tactile_sensor_type`。

`runtime` 配置连接 `serve_runtime.py` 暴露的单一端口。client 每个控制周期上传当前 observation 并同步等待单个 action，同时携带单调递增的 `runtime_action_id`；下一控制周期用 `runtime_ack_action_id` 确认刚执行的 action。Runtime 在收到确认前保留完全相同的 action 响应。连接异常后，client 按 `reconnect_attempts` 重建 public WebSocket 并重试同一 action ID，不发送 episode reset；server 在进程内复用唯一活动 session。`reset` 只在 episode 开始时发送。client 不携带 prefix cache/version 调度信息。
