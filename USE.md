# UniVTAC 使用说明

## 环境

```bash
# 需要Ubuntu22.04、python3.10、CUDA12.4，需要在UniVTAC里面
# 先conda激活一下环境
conda activate base

# 使用UniVTAC的安装脚本
bash scripts/install.sh

# 下面手动安装之前先激活环境
conda activate UniVTAC

# vcpkg的哈希值是错的，需要手动安装
mkdir -p /root/Toolchain
git clone https://github.com/microsoft/vcpkg.git /root/Toolchain/vcpkg
cd /root/Toolchain/vcpkg
./bootstrap-vcpkg.sh -disableMetrics

# 后面一步是安装TacEx，也可以手动安装
cd /root/workspace/UniVTAC/third_party/TacEx
./tacex.sh -i

uv pip uninstall torch_scatter -y
uv pip install torch_scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

export CMAKE_TOOLCHAIN_FILE=/root/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake
uv pip install -e source/tacex_uipc -v --no-build-isolation

# 验证环境
python scripts/eval-env.py

# 服务器里面只能跑 headless 模式，注意x11客户端需要的依赖
apt-get install -y --no-install-recommends libsm6 libice6 libxt6 libglu1-mesa vulkan-tools
```

## 运行 client（仿真推理）

单进程评测：

```bash
bash eval_policy.sh <task_name> <task_config> <policy_config> <gpu_id>
```

论文/仓库仿真默认配置示例（默认评测 100 个 episode）：

```bash
bash eval_policy.sh lift_bottle demo ACT/deploy 0
```

并行评测：

```bash
bash parallel_eval.sh <task_name> <task_config> <policy_config> \
    <gpu_list> [num_processes] [total_num]
```

```bash
bash parallel_eval.sh lift_bottle demo ACT/deploy 0,1 2 100
```

评测结果默认保存到：

```text
eval_result/<policy_name>/<task_name>/<deploy_config>/<时间戳>/
```

## 仿真默认值

以下是当前 `task_config/demo.yml`/`contact.yml` 和 `BaseTaskCfg` 的仿真默认值，
不包含 ACT、Ablation、ViTAL 等 policy 的模型超参数。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `task_config` | `demo` | 使用 `task_config/demo.yml`；`contact` 也可选 |
| `decimation` | `1` | 每个控制步执行的物理子步数 |
| 仿真 `dt` | `1/120 s` | Isaac Lab 物理仿真步长 |
| `step_lim` | `300` | 单个评测 episode 的最大环境步数 |
| `scene.num_envs` | `1` | 单进程环境数量，评测脚本固定为 1 |
| `save_frequency` | `2` | 每 2 个环境步保存一次观测 |
| `video_frequency` | `2` | 每 2 个环境步保存一帧视频；`0` 表示不录制 |
| `random_texture` | `false` | 是否启用随机纹理 |
| `sensor_type` | `gsmini` | 默认触觉传感器；当前任务配置默认使用 GelSight Mini |
| `observations.camera` | `['rgb']` | 相机观测类型 |
| `observations.tactile` | `['rgb', 'rgb_marker', 'marker', 'depth', 'pose']` | 触觉观测类型 |
| `observations.embodiment` | `['joint', 'ee']` | 机器人关节与末端状态 |
| `observations.actor` | `true` | 是否返回场景物体状态 |
| 评测 `total_num` | `100` | 要完成的评测 episode 数 |
| 部署 `seed` | `0` | 起始 seed 偏移；默认首个 seed 为 `1000000` |

`demo` 与 `contact` 当前仅在数据采集数量上不同：`demo.yml` 的
`episode_num=5`，`contact.yml` 的 `episode_num=15`。`episode_num` 只由数据采集
脚本使用，评测数量由 `--total_num` 或并行脚本最后一个参数控制。

## 可选参数

### 位置参数

| 参数 | 可选值/范围 | 用法 |
|---|---|---|
| `task_name` | `insert_HDMI`、`insert_hole`、`insert_tube`、`lift_bottle`、`lift_can`、`pull_out_key`、`put_bottle_in_shelf`、`grasp_classify` 等 `envs/` 中的任务模块 | `bash eval_policy.sh lift_bottle ...` |
| `task_config` | `demo`、`contact`，或 `task_config/` 下自定义 YAML | `bash eval_policy.sh lift_bottle contact ...` |
| `policy_config` | `policy/` 下已有部署配置，格式为 `<Policy>/<config>` | `ACT/deploy` |
| `gpu_id` | GPU 编号字符串，如 `0`、`1` | `... ACT/deploy 0` |
| `gpu_list` | 一个或多个 GPU 编号，逗号分隔 | `0,1` |
| `num_processes` | 正整数，默认 `2` | `parallel_eval.sh ... 0,1 2 100` |
| `total_num` | 正整数，默认 `100` | `parallel_eval.sh ... 0,1 2 20` |

### Python 评测选项

需要额外控制时，直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py \
    <task_name> <task_config> <policy_config> [options]
```

| 选项 | 可选值/范围 | 用法 |
|---|---|---|
| `--total_num` | 正整数，默认 `100` | `--total_num 20` |
| `--start_seed` | `-1`（使用部署默认 seed）或非负整数 | `--start_seed 1000000` |
| `--max_seed` | `-1`（不限制）或非负整数 | `--max_seed 1000099` |
| `--expert_check` | 开关，默认关闭 | `--expert_check` |
| `--print_only` | 开关，默认关闭 | `--print_only` |
| `--device` | `cpu`、`cuda`、`cuda:N` | `--device cuda:0` |
| `--headless` | 开关，默认由 Isaac Lab 决定 | `--headless` |
| `--rendering_mode` | `performance`、`balanced`、`quality`、`xr` | `--rendering_mode performance` |

### 自定义 `task_config` 可改参数

在 `task_config/*.yml` 中修改后，用该文件名作为第二个位置参数。

| 参数 | 可选值/范围 | 用法 |
|---|---|---|
| `decimation` | 正整数；值越大，每个控制步仿真时间越长 | `decimation: 2` |
| `save_frequency` | 正整数 | `save_frequency: 4` |
| `video_frequency` | `0` 或正整数；`0` 关闭视频 | `video_frequency: 0` |
| `random_texture` | `true`、`false` | `random_texture: true` |
| `sensor_type` | `gsmini`、`gf225`、`xensews` | `sensor_type: gsmini` |
| `observations.camera` | 当前相机支持的观测类型：`rgb`、`depth` | `camera: ['rgb']` |
| `observations.tactile` | `rgb`、`rgb_marker`、`marker`、`depth`、`pose` 的组合 | `tactile: ['rgb', 'depth']` |
| `observations.embodiment` | `joint`、`ee` 的组合 | `embodiment: ['joint', 'ee']` |
| `observations.actor` | `true`、`false` | `actor: false` |

## pi-r2-UniVTAC 运行
```bash
## server，需要在pi-r2-GR00T下面
.venv/bin/python gr00t/eval/run_gr00t_server.py \
  --model-path ckpt/GR00T-N1.7-3B \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda:0 \
  --decoupled \
  --host 127.0.0.1 \
  --port 5555
# server这里需要cosmos的权重，代码使用的hf_home导入，需要export HF_HOME=/root/workspace/pi-r2-GR00T/.hf配合一下

## client，需要在UniVTAC下面
bash eval_policy.sh lift_bottle gr00t_rgb GR00T/deploy 0

## debug
# python -m debugpy --listen 5678 --wait-for-client python脚本

```