# RISE NavSim Full-controller EPDMS configuration

[Project overview](../../../../README.md) |
[Configuration reference](../../../../docs/configuration.md) |
[English reproduction](../../../../docs/reproduction.md) |
[中文复现指南](../../../../docs/reproduction_zh-CN.md)

This directory exposes one public evaluation configuration:

- [`full_controller.yaml`](full_controller.yaml) — NavTest Full-controller EPDMS at H=4.

The controller selects H0 through H4 online. The CLI exposes no forced horizon option, the scorer boundary fixes
stage12, and only the official one-stage scorer is used.

Before execution, replace all external `/path/to/` values, keep the repository-relative
`configs/train/navsim/cvoi_manual_full/05_p1_full.yaml`, use one consistent Full artifact root with fixed handoff
suffixes, and choose a separate EPDMS output root. Exact environment and launch commands are in the reproduction
guides.

```bash
python3 tools/run_cvoi_direct_epdms.py \
  --config /path/to/rise/configs/eval/navsim/cvoi_manual_epdms/full_controller.yaml
```
