# RISE configuration and path reference

[Project overview](../README.md) | [English reproduction](reproduction.md) |
[中文复现指南](reproduction_zh-CN.md)

RISE exposes seven full, flat YAML training configurations and one full, flat YAML evaluation configuration. The files
contain complete settings; the public release does not add environment-variable interpolation, inheritance, a DAG,
scheduler, automatic retry, automatic checkpoint selection, or a new execution mode.

## Public configuration inventory

The training set is fixed to these files:

1. [`01_predictor_lewm_pure.yaml`](../configs/train/navsim/cvoi_manual_full/01_predictor_lewm_pure.yaml) — independent
   Predictor reproduction.
2. [`02_p0_uniform.yaml`](../configs/train/navsim/cvoi_manual_full/02_p0_uniform.yaml) — unguided P0 planner candidates.
3. [`03_field_full.yaml`](../configs/train/navsim/cvoi_manual_full/03_field_full.yaml) — Counterfactual Field value.
4. [`04_calibration_full.yaml`](../configs/train/navsim/cvoi_manual_full/04_calibration_full.yaml) — local Calibration.
5. [`05_p1_full.yaml`](../configs/train/navsim/cvoi_manual_full/05_p1_full.yaml) — guided P1 planner candidates.
6. [`06_stop_full.yaml`](../configs/train/navsim/cvoi_manual_full/06_stop_full.yaml) — Stop value.
7. [`07_gate_full.yaml`](../configs/train/navsim/cvoi_manual_full/07_gate_full.yaml) — Gate trained from the embedded Oracle.

The only final evaluation configuration is
[`configs/eval/navsim/cvoi_manual_epdms/full_controller.yaml`](../configs/eval/navsim/cvoi_manual_epdms/full_controller.yaml).
It selects NavTest Full-controller EPDMS at H=4. It does not expose a forced-horizon field.

The Predictor file is part of this inventory but remains an independent experiment. The controller chain is
`P0 -> Field -> Calibration -> P1 -> Stop -> Oracle -> Gate`.

## Path grammar

Path values belong to one of two authorities:

- **External paths** identify user-provided data, checkpoints, environments, or writable outputs. They must be
  normalized absolute paths. Public examples start at `/path/to/`; replace them before execution. Do not use `..`
  traversal or rely on the process working directory.
- **Repository-owned paths** identify files tracked in this repository. They remain repository-relative and are
  resolved against the repository root, not the caller's current directory.

The neutral `/path/to/` examples intentionally parse in no-data configuration tests. Production preflight rejects any
unedited placeholder before launching data-dependent work or creating an evaluation output. Once edited, execution also
checks existence, regular-file and symlink policy, checkpoint schema and tensor roles, SQLite contents, and input
fingerprints. There is no path discovery or fallback.

## Data path group

The `data.navsim` group describes natural and Counterfactual samples. Review the following path-valued fields wherever
they appear in `train_roots` and `val_roots`:

- `data_path` — NavSim log or Counterfactual PKL root;
- `sensor_blobs_path` — matching camera/sensor blob root;
- `pose_overlay_path` — required Counterfactual pose overlay root;
- `annotations_path` — locked Counterfactual annotation JSON;
- `trajectory_quality_path` — generated Field sidecar for the matching split;
- `scene_filter_yaml` — repository-owned scene filter configuration.

Natural NavSim and Counterfactual entries have different domains and must not be pointed at the same directory merely
to make a check pass. Train and validation roots must preserve their declared split. Pose overlay coordinate-frame and
timestamp settings are non-path experiment semantics and should not be changed as part of path customization.

The two scene filter paths are repository-relative resources under the retained NavSim tree. They are resolved from the
repository root. Do not convert a scene filter into a machine mount path.

## Checkpoint and parameter path group

Planner stages identify their compatible world-model warm start through `cvoi.source_checkpoint.path`, the matching
parameter-source path, `world_model_checkpoint`, and `seed_planner_checkpoint`. For the released graph these values must
refer to the same user-supplied e120 checkpoint and parameter file, for example:

```text
/path/to/checkpoints/rise/e120.pt
/path/to/checkpoints/rise/params-pretrain.yaml
```

The Predictor uses its own configured pretraining input and does not supply a P0 handoff. Do not redirect P0 to the
Predictor output.

Stage-parent checkpoint fields are separate from warm-start fields:

- `unguided_planner_checkpoint` consumes the selected P0 handoff;
- `field_checkpoint` consumes Field for Calibration and Calibration for P1, according to the stage contract;
- `guided_planner_checkpoint` consumes the selected P1 handoff for Stop;
- `oracle_path` supplies the aggregated Oracle to Gate;
- `gate_checkpoint` is used only at controller evaluation.

Mixed Full roots, an incorrect suffix, a relative external checkpoint, and a checkpoint with the wrong runtime schema
are rejected rather than repaired.

## Output path group

Choose one custom, normalized absolute **Full results root** and replace every
`/path/to/rise/results/cvoi_manual_full` occurrence consistently. P0 and P1 first write candidate outputs beneath the
configured `p0/` and `p1/` directories. An operator then publishes the chosen candidates at their fixed selected
handoffs. Other stages publish their configured fixed output directly.

An ablation output root is independent from the Full results root. Each ablation writes below its separately configured
root, while its P0 input remains the selected P0 handoff from the Full root. Do not duplicate or relocate Full P0 into
an ablation root.

EPDMS uses another independent output root, such as `/path/to/rise/results/cvoi_manual_epdms/full`. The selected output
directory must not already exist when evaluation begins.

## Fixed Full handoff contract

All Full controller artifacts derive from the same Full results root and use these exact suffixes:

```text
handoff/p0_selected.pt
handoff/field.pt
handoff/calibration.pt
handoff/p1_selected.pt
handoff/stop.pt
handoff/oracle_full.sqlite3
handoff/gate.pt
```

The stage relationship is fixed:

| Producer or operator action | Stable handoff | Next consumer |
| --- | --- | --- |
| Manual P0 candidate selection | `handoff/p0_selected.pt` | Field and later Full stages |
| Field | `handoff/field.pt` | Calibration |
| Calibration | `handoff/calibration.pt` | P1 and Stop |
| Manual P1 candidate selection | `handoff/p1_selected.pt` | Stop and evaluation |
| Stop | `handoff/stop.pt` | Oracle/Gate runtime |
| Oracle aggregation | `handoff/oracle_full.sqlite3` | Gate and evaluation |
| Gate | `handoff/gate.pt` | Full-controller evaluation |

P0 and P1 selection is deliberately manual. The repository does not infer the best checkpoint, create a selection
receipt, or start the next stage.

## Sidecar, audit, and preflight path group

Field's two `trajectory_quality_path` values are derived beneath the Full root:

```text
/path/to/rise/results/cvoi_manual_full/preflight/trajectory_quality/navsim_cf_train.json
/path/to/rise/results/cvoi_manual_full/preflight/trajectory_quality/navsim_cf_val.json
```

Generate them with the commands in the [reproduction guide](reproduction.md). Each sidecar binds the Counterfactual
PKL, pose overlay, and locked annotation inputs used to construct it. The data loader rechecks those identities.

Audit and cohort paths that name tracked filters stay repository-relative. Generated manifests and Oracle working data
derive from the explicitly supplied results root. These fingerprints protect data identity; they are not workflow
receipts and do not authorize automatic stage progression.

## EPDMS path group

The public Full-controller file has four common paths:

- `training_config_path` is exactly the repository-relative
  `configs/train/navsim/cvoi_manual_full/05_p1_full.yaml`;
- `encoder_checkpoint_path` is a user-supplied absolute checkpoint;
- `scenario_manifest_path` is an absolute NavTest manifest under the configured Full preflight area;
- `output_root` is the separately configured absolute EPDMS output root.

Its controller artifact mapping must use one consistent Full root and these fixed suffixes:

```text
handoff/p0_selected.pt
handoff/calibration.pt
handoff/p1_selected.pt
handoff/stop.pt
handoff/gate.pt
handoff/oracle_full.sqlite3
```

Field is part of the seven-handoff training contract but is not loaded directly by the final controller projection.
The EPDMS loader rejects a mixed artifact root or any wrong suffix before launching the official scorer. The shell also
requires `OPENSCENE_DATA_ROOT`, `NAVSIM_EXP_ROOT`, `NUPLAN_MAPS_ROOT`, and `NAVSIM_DEVKIT_ROOT`; none has a default.

## Safe customization checklist

1. Copy or edit the public YAML deliberately; do not add interpolation or inheritance.
2. Replace every external `/path/to/` value with a normalized absolute path.
3. Keep repository-owned scene filters and the EPDMS P1 training config repository-relative.
4. Use one consistent Full results root for all seven handoffs.
5. Give ablations and EPDMS their own explicit output roots while retaining the Full P0 authority.
6. Preserve every fixed suffix and every non-path training value.
7. Parse the configuration with no data, then run the stage's production preflight in the target environment.

Changing optimization, sampling, architecture, dataset balance, guidance, seed, or schedule fields creates a different
experiment and is not path customization.
