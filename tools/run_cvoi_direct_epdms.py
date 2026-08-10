#!/usr/bin/env python3
"""Run exactly one direct NavTest CVoI EPDMS evaluation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def run_cvoi_direct_epdms(*args: object, **kwargs: object) -> int:
    """Load the direct data plane only after command-line parsing succeeds."""

    from app.vjepa_cowa_world_model.evaluation.cvoi_direct_epdms import run_cvoi_direct_epdms as run_direct

    return run_direct(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_cvoi_direct_epdms(
        args.config,
        forced_horizon=None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
