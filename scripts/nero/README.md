# Nero Mission 7 数据流水线

这组脚本把只读 NAS 中的 Mission 7 canonical 数据和 `meta/training_views.jsonl` 转换为 OpenPI 可直接读取的
LeRobot v2.0 本地训练视图。脚本不会修改 NAS 原始数据，也不会启动模型训练。

默认路径：

- 源数据：`/mnt/nero_nas/missions/nero/mission7/smooth`
- 输出数据：`<仓库>/data/nero/mission7_training_views_lerobot_v2_0`
- 训练配置：`pi05_nero_stream5_mission7_views`
- 视图类型：`full phase transition`

可通过环境变量覆盖：

```bash
export NERO_MISSION7_SOURCE_ROOT=/path/to/read-only/source
export NERO_MISSION7_DATASET_ROOT=/path/to/generated/view
export NERO_MISSION7_VIEW_TYPES="full phase transition"
```

分步执行：

```bash
scripts/nero/00_audit_mission7_source.sh
scripts/nero/10_build_mission7_training_views.sh
scripts/nero/20_validate_mission7_training_views.sh
scripts/nero/30_compute_mission7_norm_stats.sh
scripts/nero/40_validate_mission7_normalized_batch.sh
```

也可以在确认目标目录不存在、且希望连续执行所有数据步骤时运行：

```bash
scripts/nero/run_mission7_data_pipeline.sh
```

`10_build` 每条 `training_views.jsonl` 记录生成一个虚拟 LeRobot episode。Parquet 按视图边界切片；父视频只复制
一次，虚拟 episode 使用相对软链接复用完整父视频，时间戳保留在父视频坐标系中。输出目录是自包含的，迁移时必须
保留相对软链接，例如使用 `rsync -a`，不要再使用会展开链接并制造重复视频的 `rsync -L`。

脚本默认拒绝覆盖现有输出或 normalization 文件。需要重跑时，应先人工核对并移走旧输出，而不是直接覆盖。
