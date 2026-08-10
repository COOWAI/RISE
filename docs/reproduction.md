# RISE manual reproduction guide

[Project overview](../README.md) | [中文复现指南](reproduction_zh-CN.md) |
[Configuration reference](configuration.md)

This guide documents the retained manual NavSim workflow. It is a sequence of operator-invoked commands, not a DAG or
an automatic experiment manager. Run every command from the repository root after reviewing and editing the seven flat
YAML files described in the [configuration reference](configuration.md).

The Predictor is an independent reproduction experiment. The controller path is exactly
`P0 -> Field -> Calibration -> P1 -> Stop -> Oracle -> Gate`, followed by the single maintained final evaluation,
NavTest Full-controller EPDMS at H=4.

The repository does not include weights, Counterfactual data, the paper, or numerical results. The commands below have
not been run as part of the public release, and this guide does not claim successful training or scoring.

## 1. Environment and explicit paths

Install from a checkout and make the repository importable:

```bash
cd /path/to/rise
python3 -m pip install -e .
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export RISE_FULL_ROOT=/path/to/rise/results/cvoi_manual_full
mkdir -p "$RISE_FULL_ROOT/handoff"
```

Before real execution, edit all external paths in
`configs/train/navsim/cvoi_manual_full/01_predictor_lewm_pure.yaml` through
`configs/train/navsim/cvoi_manual_full/07_gate_full.yaml`. A consistent deployment replaces the neutral roots with
absolute paths such as:

```text
/path/to/navsim/dataset
/path/to/counterfactual
/path/to/checkpoints/rise
/path/to/rise/results/cvoi_manual_full
```

Unedited `/path/to/` values are accepted for no-data parsing but rejected by production preflight. Missing inputs are
not replaced by fallback assets.

## 2. Predictor independent reproduction

The Predictor configuration is included in the seven-file release, but its output is not used as a P0 parent.

```bash
torchrun --standalone --nproc_per_node=8 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/01_predictor_lewm_pure.yaml \
  --train-script train_latent_predictor
```

## 3. P0 and manual candidate handoff

Run the uniform P0 planner stage:

```bash
torchrun --standalone --nproc_per_node=8 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/02_p0_uniform.yaml \
  --train-script train_predictor_rollout_planner
```

Inspect the produced P0 candidates manually. After choosing one ordinary checkpoint from the configured P0 result
directory, publish only that selection at the fixed handoff path:

```bash
cp --remove-destination \
  "/path/to/rise/results/cvoi_manual_full/p0/<chosen-p0-checkpoint>" \
  "$RISE_FULL_ROOT/handoff/p0_selected.pt"
```

The selected file may instead be an explicitly created symlink whose resolved target is a regular candidate file
inside the configured P0 directory. RISE does not choose, rank, or accept a candidate automatically.

## 4. Counterfactual trajectory-quality sidecars

Field training requires separate train and validation trajectory-quality sidecars. Generate each output once; the tool
rejects an existing target rather than overwriting it.

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

The sidecars record input identity for data-integrity checks. They are not stage-completion receipts and do not launch
later stages.

## 5. Field and Calibration

Run Field after both sidecars and the selected P0 handoff exist:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/03_field_full.yaml \
  --train-script train_cvoi_offline
```

Its fixed output is `handoff/field.pt`. Then run Calibration:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/04_calibration_full.yaml \
  --train-script train_cvoi_offline
```

Its fixed output is `handoff/calibration.pt`.

## 6. P1 and manual candidate handoff

Run the guided P1 planner stage:

```bash
torchrun --standalone --nproc_per_node=8 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/05_p1_full.yaml \
  --train-script train_predictor_rollout_planner
```

Inspect the configured P1 candidate directory and explicitly publish the chosen checkpoint:

```bash
cp --remove-destination \
  "/path/to/rise/results/cvoi_manual_full/p1/<chosen-p1-checkpoint>" \
  "$RISE_FULL_ROOT/handoff/p1_selected.pt"
```

As with P0, an explicit symlink is allowed only when it resolves to a regular candidate file inside the configured P1
directory. There is no automatic selection.

## 7. Stop

Run Stop after Calibration and the selected P1 handoff exist:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/06_stop_full.yaml \
  --train-script train_cvoi_offline
```

Its fixed output is `handoff/stop.pt`.

## 8. NavTrain Oracle: five explicit scorer runs

Oracle construction needs six explicit environment values. They have no private or machine-specific defaults:

```bash
export CVOI_NAVSIM_DATA_ROOT=/path/to/navsim/dataset
export CVOI_NAVSIM_EXP_ROOT=/path/to/navsim/experiment
export CVOI_NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export CVOI_NAVSIM_METRIC_CACHE_ROOT=/path/to/navsim/metric_cache
export CVOI_NAVSIM_DEVKIT_ROOT=/path/to/navsim/devkit
export CVOI_NAVSIM_PYTHON_BIN=/path/to/python
```

Build the raw NavTrain manifest once:

```bash
python3 tools/run_cvoi_manual_oracle.py build-manifest \
  --results-root "$RISE_FULL_ROOT"
```

Run the five horizons separately. A failure does not trigger a retry or a subsequent horizon:

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

Only after all five scorer runs succeed, aggregate the self-contained Oracle:

```bash
python3 tools/run_cvoi_manual_oracle.py aggregate \
  --results-root "$RISE_FULL_ROOT" \
  --source-config configs/train/navsim/cvoi_manual_full/05_p1_full.yaml
```

The published Oracle handoff is `handoff/oracle_full.sqlite3`.

## 9. Gate

Run Gate after the aggregated Oracle exists:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m app.main \
  --fname configs/train/navsim/cvoi_manual_full/07_gate_full.yaml \
  --train-script train_cvoi_offline
```

Its fixed output is `handoff/gate.pt`.

## 10. Fixed handoff contract

The Full chain exchanges artifacts only through these stable suffixes under one configured Full results root:

```text
handoff/p0_selected.pt
handoff/field.pt
handoff/calibration.pt
handoff/p1_selected.pt
handoff/stop.pt
handoff/oracle_full.sqlite3
handoff/gate.pt
```

P0 and P1 selections are operator-maintained. Field, Calibration, Stop, Oracle, and Gate publish their fixed outputs
through strict runtime boundaries. Execution validates file type, checkpoint structure, SQLite contents, and relevant
input fingerprints.

## 11. NavTest Full-controller EPDMS

Edit `configs/eval/navsim/cvoi_manual_epdms/full_controller.yaml` so its external paths match the same Full results
root and a separate, new EPDMS output root. The repository-relative `training_config_path` remains
`configs/train/navsim/cvoi_manual_full/05_p1_full.yaml`.

The scorer shell requires these four environment roots and provides no defaults:

```bash
export OPENSCENE_DATA_ROOT=/path/to/navsim/dataset
export NAVSIM_EXP_ROOT=/path/to/navsim/experiment
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export NAVSIM_DEVKIT_ROOT=/path/to/navsim/devkit
```

Optional explicit overrides include `METRIC_CACHE_PATH` and `PYTHON_BIN`. The selected EPDMS output directory must not
already exist. Launch exactly one Full-controller run with an absolute public-config path:

```bash
python3 tools/run_cvoi_direct_epdms.py \
  --config /path/to/rise/configs/eval/navsim/cvoi_manual_epdms/full_controller.yaml
```

Do not add a horizon argument. The controller selects H0 through H4 online, the shell fixes the agent to stage12, and
the boundary invokes the official `run_pdm_score_one_stage.py` scorer once on NavTest.

## 12. No-data validation

The following checks do not perform training or scoring:

```bash
python3 -m pytest -q tests/test_cvoi_manual_full_configs.py
python3 -m pytest -q tests/test_cvoi_direct_epdms_config.py
bash -n scripts/eval_navsim/eval_navsim_v2_pdms.sh
python3 tools/run_cvoi_manual_oracle.py --help
python3 tools/run_cvoi_direct_epdms.py --help
```
