# OpenPI 适配器

`openpi` 使用单服务端 `infer` 接口。每次推理返回一个动作序列，客户端连续执行 `watermark` 个动作后重新规划。该适配器保持同步请求。

`streaming_openpi` 使用 VLM/FM 服务端组合。模型默认 `horizon=50`、`streaming_chunk_size=5`；剩余动作降至 `watermark=30` 时，客户端按 `horizon - watermark` 的间隔在后台刷新 VLM cache。FM chunk 耗尽时调用 `stream_infer`，并通过 `executed_action_id` 发送 UniVTAC 动作计数。VLM 刷新不会阻塞当前 FM chunk，FM chunk 请求仍在 chunk 边界同步等待，以匹配服务端的 streaming 状态协议。

两个适配器均使用 UniVTAC 数据集的 9 维约定：七个机械臂关节和两个同步手指关节。客户端选择第一个手指输出作为 UniVTAC 的第八维夹爪动作，模拟器将该标量写入两个手指。

启用 tactile checkpoint 时，`streaming_openpi` 还发送 `left_marker` 和 `right_marker`。当前 client 使用 GS Mini 原生的 63 个 marker，输入契约为 `(2,63,2)`；部署配置中的 `marker_count` 和仿真传感器类型需要与 checkpoint 保持一致。

`openpi` 连接普通 OpenPI 服务端。`streaming_openpi` 连接通过 `scripts/serve_policy.py --multi-process` 启动的服务，FM 使用 `fm_port`，VLM 使用 `vlm_port`。

Streaming tactile checkpoint 可设置 `tactile_history_enabled` 和 `tactile_history_size`。client 每个 action 都采集一帧 marker；开关打开时，FM 按“最远到最近”发送固定长度 `(N,2,63,2)` 历史，历史不足时重复最早一帧填充。开关关闭时仍保持相同采样频率，但 FM 只收到最近一帧 `(2,63,2)`。

```bash
bash eval_policy.sh lift_bottle demo openpi/deploy 0
bash eval_policy.sh lift_bottle demo streaming_openpi/deploy 0
```
