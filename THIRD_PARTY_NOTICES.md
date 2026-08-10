# Third-Party Notices

RISE includes code from the projects and authors listed below. These notices do not replace the license texts or
copyright headers shipped with the source. The repository-level MIT license applies to RISE code and to retained
Meta/Facebook code covered by that license; components identified as Apache-2.0 remain under Apache-2.0.

## Meta Platforms and Facebook

Source files under `app/`, `src/`, and `tests/` that carry a Meta Platforms, Inc. or Facebook, Inc. copyright header
are retained from or derived from Meta's V-JEPA code. They are distributed under the MIT terms in `LICENSE`, which
preserves the Meta copyright notice. Upstream project: <https://github.com/facebookresearch/vjepa2>.

## Apache-2.0 components

The complete Apache License, Version 2.0 is provided at `licenses/Apache-2.0.txt`.

- `src/datasets/utils/worker_init_fn.py` includes code copyrighted by the Lightning AI team. Its existing header
  identifies the file as Apache-2.0 licensed and records the exact upstream Lightning commit.
- `src/datasets/utils/video/randaugment.py` and `src/datasets/utils/video/randerase.py` include code copyrighted by
  Ross Wightman. Their existing headers identify those files as Apache-2.0 licensed and link to the upstream
  `pytorch-image-models` implementation: <https://github.com/huggingface/pytorch-image-models>.
- `configs/navsim/scene_filters/navtrain.yaml` and `configs/navsim/scene_filters/navtest.yaml` are Apache-2.0
  components covered by `licenses/Apache-2.0.txt`.

## DPM-Solver

`app/vjepa_cowa_world_model/diffusion_utils/dpm_solver_pytorch.py` is derived from the official DPM-Solver
implementation by Cheng Lu at <https://github.com/LuChengTHU/dpm-solver> and is distributed under the MIT License
below.

```text
MIT License

Copyright (c) 2022 Cheng Lu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## RISE trajectory diffusion implementation

The trajectory diffusion planner, linear VP-SDE, and sampling adapter at the four paths below are
independently written RISE code distributed under the root MIT license:

- `app/vjepa_cowa_world_model/models/diffusion_planner.py`
- `app/vjepa_cowa_world_model/diffusion_utils/__init__.py`
- `app/vjepa_cowa_world_model/diffusion_utils/sde.py`
- `app/vjepa_cowa_world_model/diffusion_utils/sampling.py`

Their technical basis is described by the following papers:

- DiT: <https://arxiv.org/abs/2212.09748>
- Score-SDE: <https://arxiv.org/abs/2011.13456>
- DPM-Solver++: <https://arxiv.org/abs/2211.01095>

These files do not contain source from the former private XTR adaptation.
