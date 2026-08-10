# RISE 手工复现指南

[项目首页](../README_zh-CN.md) | [English guide](reproduction.md) | [配置参考](configuration.md)

本文记录保留的 NavSim 手工工作流。它是一组由操作者逐条执行的命令，不是 DAG，也不是自动实验管理器。
所有命令均从仓库根目录运行；开始前请按[配置参考](configuration.md)检查并编辑七份平铺 YAML。

Predictor 是独立复现实验。控制链精确为
`P0 -> Field -> Calibration -> P1 -> Stop -> Oracle -> Gate`，随后才是唯一保留的最终评测：H=4 的
NavTest Full-controller EPDMS。

仓库不包含权重、Counterfactual 数据、论文或数值结果。公开发布过程没有执行下列数据/GPU 命令，本文
也不声称训练或评分已经成功完成。

## 1. 环境与显式路径

从 checkout 安装并设置导入路径：

```bash
cd /path/to/rise
python3 -m pip install -e .
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export RISE_FULL_ROOT=/path/to/rise/results/cvoi_manual_full
mkdir -p "$RISE_FULL_ROOT/handoff"
```

真实执行前，编辑
`configs/train/navsim/cvoi_manual_full/01_predictor_lewm_pure.yaml` 至
`configs/train/navsim/cvoi_manual_full/07_gate_full.yaml` 中的全部外部路径。一次一致部署应将中性示例替换
为如下绝对路径：

```text
/path/to/navsim/dataset
/path/to/counterfactual
/path/to/checkpoints/rise
/path/to/rise/results/cvoi_manual_full
```

未编辑的 `/path/to/` 值可以用于无数据解析，但会被 production preflight 拒绝；缺失输入不会回退到
其他资产。

## 2. Predictor 独立复现

Predictor 配置属于七文件发布集合，但它的输出不作为 P0 parent。

```bash
torchrun --standalone --nproc_per_node=8 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/01_predictor_lewm_pure.yaml \
  --train-script train_latent_predictor
```

## 3. P0 与人工候选交接

运行 uniform P0 planner：

```bash
torchrun --standalone --nproc_per_node=8 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/02_p0_uniform.yaml \
  --train-script train_predictor_rollout_planner
```

人工检查 P0 候选。选定配置结果目录中的普通 checkpoint 后，只把该候选发布到固定 handoff：

```bash
cp --remove-destination \
  "/path/to/rise/results/cvoi_manual_full/p0/<chosen-p0-checkpoint>" \
  "$RISE_FULL_ROOT/handoff/p0_selected.pt"
```

也可以显式创建软链接，但其解析目标必须是配置 P0 目录内的普通候选文件。RISE 不会自动选择、排序或
验收候选。

## 4. Counterfactual trajectory-quality sidecar

Field 训练需要相互独立的 train 与 validation trajectory-quality sidecar。每个输出只生成一次；目标
已经存在时工具会拒绝覆盖。

```bash
python3 tools/generate_navsim_cf_trajectory_quality.py \
  --pkl-root /path/to/counterfactual/navsim_logs/trainval \
  --pose-overlay-root /path/to/counterfactual/pose_overlay/trainval/pred_pose \
  --output "$RISE_FULL_ROOT/preflight/trajectory_quality/navsim_cf_train.json" \
  --timestep-sec 0.5 \
  --pose-overlay-coord-frame opencv_first_frame \
  --pose-overlay-txt-start-seconds 0.0 \
  --max-progress-m 20 \
  --pkl-fingerprint-scope relative_path_identity \
  --formal-v2-timeline \
  --formal-v2-annotations /path/to/counterfactual/annotations/navsim_train.json \
  --camera-name CAM_F0

python3 tools/generate_navsim_cf_trajectory_quality.py \
  --pkl-root /path/to/counterfactual/navsim_logs/test \
  --pose-overlay-root /path/to/counterfactual/pose_overlay/test/pred_pose \
  --output "$RISE_FULL_ROOT/preflight/trajectory_quality/navsim_cf_val.json" \
  --timestep-sec 0.5 \
  --pose-overlay-coord-frame opencv_first_frame \
  --pose-overlay-txt-start-seconds 0.0 \
  --max-progress-m 20 \
  --pkl-fingerprint-scope relative_path_identity \
  --formal-v2-timeline \
  --formal-v2-annotations /path/to/counterfactual/annotations/navsim_test.json \
  --camera-name CAM_F0
```

sidecar 记录数据输入身份，用于一致性校验；它不是阶段完成 receipt，也不会启动后续阶段。

## 5. Field 与 Calibration

两份 sidecar 和选定 P0 handoff 都存在后运行 Field：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/03_field_full.yaml \
  --train-script train_cvoi_offline
```

固定输出为 `handoff/field.pt`。随后运行 Calibration：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/04_calibration_full.yaml \
  --train-script train_cvoi_offline
```

固定输出为 `handoff/calibration.pt`。

## 6. P1 与人工候选交接

运行 guided P1 planner：

```bash
torchrun --standalone --nproc_per_node=8 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/05_p1_full.yaml \
  --train-script train_predictor_rollout_planner
```

人工检查配置 P1 目录中的候选，再显式发布选中的 checkpoint：

```bash
cp --remove-destination \
  "/path/to/rise/results/cvoi_manual_full/p1/<chosen-p1-checkpoint>" \
  "$RISE_FULL_ROOT/handoff/p1_selected.pt"
```

与 P0 相同，软链接只能解析到配置 P1 目录内的普通候选文件，不存在自动选择。

## 7. Stop

Calibration 与选定 P1 handoff 均存在后运行 Stop：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/06_stop_full.yaml \
  --train-script train_cvoi_offline
```

固定输出为 `handoff/stop.pt`。

## 8. NavTrain Oracle：五次显式评分

Oracle 构建需要六个显式环境变量，它们没有私有或机器相关默认值：

```bash
export CVOI_NAVSIM_DATA_ROOT=/path/to/navsim/dataset
export CVOI_NAVSIM_EXP_ROOT=/path/to/navsim/experiment
export CVOI_NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export CVOI_NAVSIM_METRIC_CACHE_ROOT=/path/to/navsim/metric_cache
export CVOI_NAVSIM_DEVKIT_ROOT=/path/to/navsim/devkit
export CVOI_NAVSIM_PYTHON_BIN=/path/to/python
```

先创建一次 raw NavTrain manifest：

```bash
python3 tools/run_cvoi_manual_oracle.py build-manifest \
  --results-root "$RISE_FULL_ROOT"
```

分别执行五个 horizon。任何一次失败都不会触发自动重试或下一个 horizon：

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/run_cvoi_manual_oracle.py score \
  --horizon 0 --results-root "$RISE_FULL_ROOT" \
  --source-config configs/train/navsim/cvoi_manual_full/05_p1_full.yaml

CUDA_VISIBLE_DEVICES=0 python3 tools/run_cvoi_manual_oracle.py score \
  --horizon 1 --results-root "$RISE_FULL_ROOT" \
  --source-config configs/train/navsim/cvoi_manual_full/05_p1_full.yaml

CUDA_VISIBLE_DEVICES=0 python3 tools/run_cvoi_manual_oracle.py score \
  --horizon 2 --results-root "$RISE_FULL_ROOT" \
  --source-config configs/train/navsim/cvoi_manual_full/05_p1_full.yaml

CUDA_VISIBLE_DEVICES=0 python3 tools/run_cvoi_manual_oracle.py score \
  --horizon 3 --results-root "$RISE_FULL_ROOT" \
  --source-config configs/train/navsim/cvoi_manual_full/05_p1_full.yaml

CUDA_VISIBLE_DEVICES=0 python3 tools/run_cvoi_manual_oracle.py score \
  --horizon 4 --results-root "$RISE_FULL_ROOT" \
  --source-config configs/train/navsim/cvoi_manual_full/05_p1_full.yaml
```

五次评分全部成功后，才聚合自包含 Oracle：

```bash
python3 tools/run_cvoi_manual_oracle.py aggregate \
  --results-root "$RISE_FULL_ROOT" \
  --source-config configs/train/navsim/cvoi_manual_full/05_p1_full.yaml
```

发布后的 Oracle handoff 为 `handoff/oracle_full.sqlite3`。

## 9. Gate

聚合 Oracle 存在后运行 Gate：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/07_gate_full.yaml \
  --train-script train_cvoi_offline
```

固定输出为 `handoff/gate.pt`。

## 10. 固定 handoff 契约

Full 链只通过同一配置 Full results root 下的以下稳定后缀交换产物：

```text
handoff/p0_selected.pt
handoff/field.pt
handoff/calibration.pt
handoff/p1_selected.pt
handoff/stop.pt
handoff/oracle_full.sqlite3
handoff/gate.pt
```

P0/P1 selection 始终由操作者维护。Field、Calibration、Stop、Oracle 与 Gate 通过严格运行边界发布固定
输出；执行时会校验文件类型、checkpoint 结构、SQLite 内容和相关输入指纹。

## 11. NavTest Full-controller EPDMS

编辑 `configs/eval/navsim/cvoi_manual_epdms/full_controller.yaml`，使其中所有 Full artifact 使用同一 Full
results root，并选择一个独立、尚不存在的 EPDMS output root。仓库相对的 `training_config_path` 保持为
`configs/train/navsim/cvoi_manual_full/05_p1_full.yaml`。

scorer shell 强制要求以下四个环境根目录，不提供默认值：

```bash
export OPENSCENE_DATA_ROOT=/path/to/navsim/dataset
export NAVSIM_EXP_ROOT=/path/to/navsim/experiment
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export NAVSIM_DEVKIT_ROOT=/path/to/navsim/devkit
```

可选显式覆盖包括 `METRIC_CACHE_PATH` 与 `PYTHON_BIN`。选中的 EPDMS 输出目录必须尚不存在。使用公开配置
的绝对路径启动唯一一次 Full-controller 评测：

```bash
python3 tools/run_cvoi_direct_epdms.py \
  --config /path/to/rise/configs/eval/navsim/cvoi_manual_epdms/full_controller.yaml
```

不要增加 horizon 参数。控制器在线选择 H0 至 H4，shell 将 agent 固定为 stage12，并在 NavTest 上调用
一次官方 `run_pdm_score_one_stage.py` scorer。

## 12. 无数据验证

以下命令不执行训练或评分：

```bash
python3 -m pytest -q tests/test_cvoi_manual_full_configs.py
python3 -m pytest -q tests/test_cvoi_direct_epdms_config.py
bash -n scripts/eval_navsim/eval_navsim_v2_pdms.sh
python3 tools/run_cvoi_manual_oracle.py --help
python3 tools/run_cvoi_direct_epdms.py --help
```
