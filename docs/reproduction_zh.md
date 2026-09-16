# StreamPI 代码与复现导航

本文档面向论文复现，基于仓库提交 `5f1d875` 整理。它回答四个问题：核心方法在哪里、训练数据如何流动、训练与评测如何启动、当前代码与论文协议有哪些差异。

## 1. 仓库结构

```text
StreamPI/
├── src/openpi/
│   ├── models/              # JAX 模型；StreamPI 的注意力与 KV cache 核心实现
│   ├── policies/            # checkpoint 加载、输入输出变换、推理时记忆管理
│   ├── training/            # 数据集、训练配置、优化器、权重与 checkpoint
│   ├── serving/             # WebSocket policy server
│   └── transforms.py        # 历史帧抽样（TemporalJitter）及通用数据变换
├── scripts/
│   ├── train.py             # 单节点 JAX 训练入口
│   ├── train_multi_node.py  # 多节点 JAX 训练入口
│   ├── serve_policy.py      # 推理服务入口
│   └── compute_norm_stats.py
├── examples/
│   ├── calvin/              # CALVIN 客户端与评测循环
│   ├── libero/              # LIBERO 客户端与评测循环
│   └── aloha_real/          # 真机示例
├── assets/                  # normalization statistics
├── packages/openpi-client/  # WebSocket 客户端
├── third_party/             # ALOHA、LIBERO 等第三方代码
└── train_*.sh / run_*.sh    # 针对作者集群环境的快捷脚本
```

`lerobot/` 是上游数据工具的大型代码副本，不是理解 StreamPI 方法的首要入口。阅读时优先看 `src/openpi/`、`examples/calvin/` 和 `examples/libero/`。

## 2. 论文方法与代码映射

| 论文组件 | 代码位置 | 实现要点 |
| --- | --- | --- |
| 多帧输入配置 | `src/openpi/models/pi0_config.py` | `hist_horizon` 控制时间上下文长度 `T` |
| 历史帧数据窗口 | `src/openpi/training/data_loader.py` | 使用 `hist_interval` 构造历史时间戳窗口 |
| 历史帧抽样 | `src/openpi/transforms.py::TemporalJitter` | 从窗口中取 `T` 帧；本工作区已增加相邻间隔独立抽样模式 |
| instruction-anchored temporal unit | `src/openpi/models/pi0.py::embed_prefix` | 每个时刻拼接所有视角的视觉 token 与同一条指令 token |
| 单元内双向、单元间因果注意力 | `src/openpi/models/pi0.py::make_attn_mask` 与 `embed_prefix` | 每个时间单元首 token 将累计块编号加一，同一块内双向、后续块只看历史块 |
| temporal masking | `src/openpi/models/pi0.py::compute_loss` | 训练时随机屏蔽最早的 `0..T-1` 个时间单元 |
| streaming KV cache | `src/openpi/models/pi0.py::sample_actions` | 新观测只编码一次，并将 prefix KV 保存到 `memory` |
| cache 生命周期 | `src/openpi/policies/policy.py::Policy.infer` | 当 `step % hist_horizon == 0` 时清空记忆 |
| 训练配置 | `src/openpi/training/config.py` | LIBERO、CALVIN、真机的 `pi05_*stream*` 配置 |
| 训练入口 | `scripts/train.py` / `scripts/train_multi_node.py` | 单节点与多节点 JAX 训练 |
| 服务入口 | `scripts/serve_policy.py` | 从 config + checkpoint 创建 WebSocket 服务 |
| CALVIN 评测 | `examples/calvin/main.py` | 客户端维护递增 `step`，按 `replan_steps` 执行动作 |
| LIBERO 评测 | `examples/libero/main.py` | 当前客户端固定发送 `step=0`，见第 5 节 |

## 3. 数据到损失的调用链

```text
TrainConfig
  -> create_data_loader
  -> LeRobotDataset(delta_timestamps=历史连续窗口)
  -> PromptFromLeRobotTask
  -> TemporalJitter(从窗口抽取 T 帧)
  -> policy-specific repack / normalize / tokenize
  -> Pi0.compute_loss
       -> embed_prefix: [(V_1, language), ..., (V_T, language)]
       -> 随机屏蔽最早若干时间单元
       -> make_attn_mask: 单元内双向、单元间因果
       -> flow-matching action loss
```

注意：`data_loader.py` 先向 LeRobot 请求完整的连续历史窗口，`TemporalJitter` 再从这个窗口中挑选最终输入帧。修改随机间隔协议时，这两处必须一起检查。

## 4. 推理调用链

```text
benchmark / robot client
  -> observation + prompt + step
  -> WebSocketPolicyServer
  -> Policy.infer
       -> step 到达 T 的边界时 reset_memory
       -> Pi0.sample_actions
            -> 只编码当前 (visual, language) 单元
            -> 拼接历史 prefix mask 与 KV cache
            -> 保存更新后的 KV cache
            -> 对 action suffix 做 flow-matching 去噪
  -> action chunk
```

这里的 `step` 是“policy 调用次数”，不是环境原始帧号。`hist_interval` / `replan_steps` 决定相邻 policy 调用之间经过多少环境动作。

## 5. 严格复现前必须确认的差异

### 5.1 random-interval training 与原始提交的差异

论文协议是每个历史间隔独立采样：

```text
delta_i ~ Uniform{3, 4, 5, 6, 7}
历史索引为 0, -delta_1, -(delta_1 + delta_2), ...
```

仓库原始提交只为每路相机分别抽取一个 `base_offset`，实际索引为：

```text
0 + offset, -hist_interval + offset, -2 * hist_interval + offset, ...
```

这还会让不同相机抽到不同的偏移。本工作区已让固定间隔模式在所有相机间共享同一组索引，并新增 `hist_interval_range`：每一段间隔独立抽样，同时所有相机使用相同索引。原有 LIBERO、CALVIN 和 AgileX 配置没有设置该字段，行为保持不变。

此外，原始提交中的 `pi05_libero_stream3/5` 和 `pi05_calvin_stream3/5` 都设置了 `enable_jitter=False`。因此，原样运行只能复现固定间隔版本，不能称为论文 random-interval 主结果。

### 5.2 LIBERO 默认客户端未使用 streaming cache

`examples/libero/main.py` 每次都发送 `"step": 0`。服务端因此在每次调用前清空 cache，得到多帧训练模型的单帧式推理路径。要验证论文中的 streaming latency 或真正的增量推理，客户端应在每个 episode 内递增计数，并在 episode reset 时归零。

### 5.3 根目录 shell 脚本带有作者集群假设

`train_stream*.sh` 读取 `VC_WORKER_HOSTS`、`MA_NUM_HOSTS`、`VC_TASK_INDEX` 和 `MA_NUM_GPUS`。普通单机没有这些变量，不应直接把它们当作通用启动器。

`run_calvin_env.sh` 中的 `CALVIN_ROOT` 默认表达式不是常规的 Bash 默认值写法。当前环境应显式设置：

```bash
export CALVIN_ROOT=/home/zhouxn/calvin
```

### 5.4 `--resume` 与首次运行不能混用

部分快捷脚本固定传入 `--resume`。全新实验没有已有 checkpoint 时应使用新的 `--exp-name` 并传入 `--overwrite`；已有实验才使用 `--resume`。

### 5.5 复现实验硬件与 node2 不同

论文训练配置为 8 张 H100；当前 `node2` 可见 1 张 NVIDIA RTX PRO 5000 72GB。该机器适合先做环境验证、数据抽样检查、单步前向与小规模 smoke test，但完整训练的显存、吞吐和全局 batch 需要另行验证或使用多卡资源。

## 6. 官方配置一览

| 配置名 | 数据集 | `T` | 间隔 | 默认 jitter | 步数 | batch |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `pi05_libero` | `physical-intelligence/libero` | 1 | 5 | 关闭 | 30,000 | 256 |
| `pi05_libero_stream3` | `physical-intelligence/libero` | 3 | 5 | 关闭 | 30,000 | 256 |
| `pi05_libero_stream5` | `physical-intelligence/libero` | 5 | 5 | 关闭 | 30,000 | 256 |
| `pi05_calvin_stream3` | `InternRobotics/InternData-Calvin_ABC` | 3 | 5 | 关闭 | 30,000 | 256 |
| `pi05_calvin_stream5` | `InternRobotics/InternData-Calvin_ABC` | 5 | 5 | 关闭 | 30,000 | 256 |
| `pi05_stream5_aloha_shell_game` | 本地 AgileX 数据 | 5 | 5 | 开启 | 50,000 | 128 |

所有上述 StreamPI 配置从 `gs://openpi-assets/checkpoints/pi05_base/params` 初始化，不新增模型参数。

## 7. node2 当前状态（2026-09-08）

| 项目 | 状态 |
| --- | --- |
| 仓库 | `/home/zhouxn/StreamPI`，`main`，提交 `5f1d875` |
| Git 工作区 | 整理前无未提交改动 |
| GPU | 1 × NVIDIA RTX PRO 5000 72GB Blackwell |
| 系统 Python | 3.12.3 |
| `uv` | 未安装或不在 `PATH` |
| 项目 `.venv` | 不存在 |
| 可用磁盘 | 约 355GB |
| CALVIN | `/home/zhouxn/calvin` 已存在 |
| normalization stats | LIBERO 与 CALVIN 的 assets 已存在 |
| 训练 checkpoint | 仓库下未发现 |

这意味着下一阶段应先建隔离环境并跑 smoke test，不应直接启动 30,000 step 训练。

## 8. 建议的复现顺序

- [ ] 固定代码提交、记录 GPU/驱动/CUDA 和随机种子。
- [ ] 安装 `uv` 并创建仓库 `.venv`。
- [ ] 验证预训练 `pi05_base` 权重可下载。
- [ ] 验证 LIBERO/CALVIN normalization statistics 与数据集字段匹配。
- [ ] 给历史帧抽样写最小测试，检查索引顺序、边界裁剪和多相机同步。
- [ ] 实现论文所述“独立随机间隔”，并保留 fixed-interval 配置用于消融。
- [ ] 修正 LIBERO 客户端的 `step` 生命周期，验证 cache reset 不跨 episode 泄漏。
- [ ] 用 fake/small batch 完成数据加载、前向、反向和 checkpoint smoke test。
- [ ] 先复现 `T=1` baseline，再运行 `T=3`、`T=5`，避免把数据或评测问题误判为时序模块问题。
- [ ] 分别记录最终成功率与推理延迟；训练效果与 streaming 效率不要混为同一项验证。

## 9. 当前不建议立即重构的内容

`examples/libero/` 下多个 `main_*` 文件高度重复，`pi0.py` 与 `pi0_rtc.py` 也包含相似逻辑。这些代码可以后续参数化合并，但复现初期应先建立结果基线与测试，再重构。否则行为变化与清理改动会混在一起，难以定位差异来源。

## 10. 最小命令入口

以下命令用于说明入口；运行前仍需完成环境和数据检查。

```bash
# normalization statistics
python scripts/compute_norm_stats.py --config-name pi05_libero
python compute_norm_calvin.py

# 单节点训练（资源参数需按实际 GPU 调整）
python scripts/train.py pi05_libero_stream5 \
  --exp-name=<experiment-name> \
  --overwrite \
  --batch-size=<global-batch-size> \
  --fsdp-devices=<device-count>

# checkpoint 推理服务
python scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=pi05_calvin_stream5 \
  --policy.dir=<checkpoint-step-directory>

# CALVIN 客户端（另一个环境/进程）
python examples/calvin/main.py \
  --calvin_root=/home/zhouxn/calvin \
  --save_name=<evaluation-name> \
  --host=<policy-server-host> \
  --port=8000
```

不要在资源尚未核对时照抄论文的 batch 与 FSDP 数值。`--batch-size` 是全局 batch，`--fsdp-devices` 必须与实际设备 mesh 匹配。

## 11. Nero mission2 接入状态与待决事项

已完成的代码接入：

- `src/openpi/training/config.py` 新增 `LeRobotNeroDataConfig`，按 `ego_view`、`wrist_view`、26 维 state 和 19 维绝对 action 重打包。
- `src/openpi/policies/nero_policy.py` 新增 Nero 输入输出适配器。`wrist_view` 是物理右腕相机，映射到 `right_wrist_0_rgb`；缺失的左腕图像以黑图补齐并设置 `image_mask=False`。输出去掉模型 padding，只保留 19 维。
- `DataConfig.dataset_root` 会传给 LeRobot metadata 和 dataset，因而不需要依赖 Hugging Face cache 软链接。
- `hist_interval_range=(1, 2)` 可在 10 FPS 数据上为四段历史间隔分别抽样，并保持两路相机同步。

当前已经完成：

- 新增 `pi05_nero_stream5_mission2`：使用 `action_horizon=17`、`hist_horizon=5`、独立 1～2 帧历史间隔、`batch_size=1` 和 `fsdp_devices=1`。
- 在 `/home/zhouxn/StreamPI/data/nero/mission2_smooth_lerobot_v2_0` 建立 LeRobot v2.0 只读视图，只包含 episode 0～196。
- 视图含 197 个本地标准化 parquet 和 394 个视频链接。视频链接检查无断链，且均指向只读 NAS 的 mission2/smooth。
- 视图保存 `meta/provenance.json`，记录来源路径、源元数据 SHA-256、筛选范围和原数据 fingerprint。
- 197 个源 parquet 原本有三类 schema。派生视图丢弃仅在 episode 40～42 出现的 11 个未使用标注列，将 38 个 episode 的 state 从 float64 转成 float32，并将 194 个 episode 的 reward 从 float64 转成 float32；最终所有 parquet schema 一致。
- `meta/parquet_manifest.jsonl` 逐 episode 保存源/派生 SHA-256、列变化、dtype 转换和行数。全部 46,920 帧的 `task_index` 均为 0。

仍需注意：

1. mission2 当前共有 255 个 episode，但只有 0～196 的 episode 元数据带有 2026-07-21 的 action XYZ 对齐和 Rot6D 校验记录；197～254 来自 2026-08-18/19 的 recap 数据，缺少这两项记录。当前训练视图因此排除了后 58 个 episode。
2. `tasks.jsonl` 同时包含正式任务、`valid`、`invalid`。视图保留原始映射以避免篡改语义，但训练配置不读取 `task_index`，而是注入固定的 mission2 指令。
3. node2 已建立 Python 3.11/uv 环境；mission2 normalization 已完成，并已通过包含归一化、32 维 padding、双相机映射和 prompt tokenization 的最终 batch 验证。

生成脚本为 `scripts/prepare_nero_lerobot_view.py`。它默认拒绝覆盖已存在的目标目录；显式替换时会先核验原视图 provenance，再在同级临时目录完成转换、检查和视频链接，将旧视图保留为带时间戳的备份，最后原子切换。最终目录权限为 0555，parquet 与元数据权限为 0444。NAS canonical 数据不参与写入。

Normalization 的维数以当前 OpenPI 调用链为准：`scripts/compute_norm_stats.py` 只执行 repack 和 Nero data adapter，得到 26 维 state 与 19 维 actions；训练时先用这组 26/19 维统计归一化，随后 `PadStatesAndActions` 才补到模型的 32 维。因此需要重新计算 Nero 专用统计，但不应手工把 `norm_stats.json` 扩成 32 维。

## 12. Nero mission7 长程任务配置

`pi05_nero_stream5_mission7` 与 mission2 并行存在，不覆盖或删除已有配置。它使用 mission7 的完整长指令，而不是把三段阶段标注拆成三个短任务；阶段标注保留用于分阶段成功率分析。

- 数据来源：`/mnt/nero_nas/missions/nero/mission7/smooth` 中全部 93 条完整成功人类示范，共 41,626 帧。
- 标准视图：已在 `/home/zhouxn/StreamPI/data/nero/mission7_smooth_lerobot_v2_0` 创建 LeRobot v2.0 只读派生视图，包含 93 个本地标准化 parquet 和 186 个只读 NAS 视频链接。
- 接口：沿用 26 维 state、19 维绝对 action、主视角和物理右腕相机适配器。
- 时序：10 FPS，`hist_horizon=5`，四段相邻历史间隔分别从 1～2 帧独立抽样。
- 动作：`action_horizon=17`，约覆盖未来 1.7 秒；长任务通过持续重规划完成，而不是一次预测完整任务。
- 首轮训练：2 卡 FSDP、global batch 2、25,000 步、每 5,000 步保存一次。
- 统计：必须生成独立的 `assets/pi05_nero_stream5_mission7/nero_mission7/norm_stats.json`，不能复用 mission2。

标准视图已通过真实数据加载验证：dataset length 为 41,626；global batch 2 下 state 为 `(2, 32)`、actions 为 `(2, 17, 32)`，三路图像均为 `(2, 5, 224, 224, 3)`；主视角和右腕 mask 为真，补齐的左腕 mask 为假。该次验证跳过 normalization，mission7 专用统计生成后还需再做一次带 normalization 的最终验收。

标准视图工具不再固定主相机分辨率，只要求 `ego_view` 与 `wrist_view` 为三通道 RGB 视频，因此同时兼容 mission2 的 800×1280 主视角和 mission7 的 180×320 主视角。state/action 维度仍严格校验为 26/19。
