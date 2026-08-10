# RISE NavSim manual-chain training configurations

[Project overview](../../../../README.md) |
[Configuration reference](../../../../docs/configuration.md) |
[English reproduction](../../../../docs/reproduction.md) |
[中文复现指南](../../../../docs/reproduction_zh-CN.md)

This directory contains exactly seven full, flat YAML files. They are manually invoked configurations, not a workflow
definition. Predictor is an independent reproduction experiment; the controller chain is
`P0 -> Field -> Calibration -> P1 -> Stop -> Oracle -> Gate`.

| Configuration | Public role |
| --- | --- |
| [`01_predictor_lewm_pure.yaml`](01_predictor_lewm_pure.yaml) | Independent Predictor reproduction |
| [`02_p0_uniform.yaml`](02_p0_uniform.yaml) | Uniform P0 planner candidates |
| [`03_field_full.yaml`](03_field_full.yaml) | Full Counterfactual Field value |
| [`04_calibration_full.yaml`](04_calibration_full.yaml) | Full local Calibration |
| [`05_p1_full.yaml`](05_p1_full.yaml) | Full guided P1 planner candidates |
| [`06_stop_full.yaml`](06_stop_full.yaml) | Full Stop value |
| [`07_gate_full.yaml`](07_gate_full.yaml) | Full Gate from the aggregated NavTrain Oracle |

External assets and output roots use neutral `/path/to/` examples and must be customized before production execution.
Repository-owned scene filters remain relative. Keep one consistent Full results root and the seven fixed handoff
suffixes documented in the canonical configuration reference.

The exact sidecar, training, manual P0/P1 selection, Oracle, and Gate commands live only in the reproduction guides.

For discoverability, the two required Field sidecar commands are indexed here as well:

```bash
export CVOI_RESULTS_ROOT=/path/to/rise/results/cvoi_manual_full
python3 tools/generate_navsim_cf_trajectory_quality.py \
  --pkl-root /path/to/counterfactual/navsim_logs/trainval \
  --pose-overlay-root /path/to/counterfactual/pose_overlay/trainval/pred_pose \
  --output "$CVOI_RESULTS_ROOT/preflight/trajectory_quality/navsim_cf_train.json" \
  --timestep-sec 0.5 --pose-overlay-coord-frame opencv_first_frame \
  --pose-overlay-txt-start-seconds 0.0 --max-progress-m 20 \
  --pkl-fingerprint-scope relative_path_identity --formal-v2-timeline \
  --formal-v2-annotations /path/to/counterfactual/annotations/navsim_train.json \
  --camera-name CAM_F0

python3 tools/generate_navsim_cf_trajectory_quality.py \
  --pkl-root /path/to/counterfactual/navsim_logs/test \
  --pose-overlay-root /path/to/counterfactual/pose_overlay/test/pred_pose \
  --output "$CVOI_RESULTS_ROOT/preflight/trajectory_quality/navsim_cf_val.json" \
  --timestep-sec 0.5 --pose-overlay-coord-frame opencv_first_frame \
  --pose-overlay-txt-start-seconds 0.0 --max-progress-m 20 \
  --pkl-fingerprint-scope relative_path_identity --formal-v2-timeline \
  --formal-v2-annotations /path/to/counterfactual/annotations/navsim_test.json \
  --camera-name CAM_F0
```
