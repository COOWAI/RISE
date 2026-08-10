#!/usr/bin/env python3
"""Run exactly one manual NavTrain Oracle data-plane action."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def load_manual_oracle_environment(*args: object, **kwargs: object) -> object:
    """Load the data-plane implementation only after CLI parsing succeeds."""

    from app.vjepa_cowa_world_model.training.cvoi_manual_navtrain_oracle import (
        load_manual_oracle_environment as load_environment,
    )

    return load_environment(*args, **kwargs)


def build_manual_navtrain_manifest(*args: object, **kwargs: object) -> object:
    """Dispatch manifest construction without importing it during ``--help``."""

    from app.vjepa_cowa_world_model.training.cvoi_manual_navtrain_oracle import (
        build_manual_navtrain_manifest as build_manifest,
    )

    return build_manifest(*args, **kwargs)


def score_manual_navtrain_horizon(*args: object, **kwargs: object) -> object:
    """Dispatch one scorer without importing it during ``--help``."""

    from app.vjepa_cowa_world_model.training.cvoi_manual_navtrain_oracle import (
        score_manual_navtrain_horizon as score_horizon,
    )

    return score_horizon(*args, **kwargs)


def aggregate_manual_navtrain_oracle(*args: object, **kwargs: object) -> object:
    """Dispatch aggregation without importing it during ``--help``."""

    from app.vjepa_cowa_world_model.training.cvoi_manual_navtrain_oracle import (
        aggregate_manual_navtrain_oracle as aggregate_oracle,
    )

    return aggregate_oracle(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_manifest = commands.add_parser("build-manifest")
    build_manifest.add_argument("--results-root", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--results-root", type=Path, required=True)
    aggregate.add_argument("--source-config", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--horizon", type=int, choices=range(5), required=True)
    score.add_argument("--results-root", type=Path, required=True)
    score.add_argument("--source-config", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build-manifest":
        environment = load_manual_oracle_environment(os.environ, require_cuda=False)
        build_manual_navtrain_manifest(args.results_root, environment)
        logger.info("Built raw manual NavTrain authority under %s", args.results_root)
    elif args.command == "score":
        environment = load_manual_oracle_environment(os.environ, require_cuda=True)
        score_manual_navtrain_horizon(
            args.results_root,
            environment,
            args.horizon,
            source_config_path=args.source_config,
        )
        logger.info("Imported manual NavTrain H%s under %s", args.horizon, args.results_root)
    elif args.command == "aggregate":
        aggregate_manual_navtrain_oracle(
            args.results_root,
            source_config_path=args.source_config,
        )
        logger.info("Published embedded manual Oracle under %s", args.results_root)
    else:
        raise RuntimeError(f"unsupported parsed command {args.command!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
