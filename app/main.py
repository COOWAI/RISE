# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pprint
from pathlib import Path

import yaml

from app.scaffold import RETAINED_TRAIN_SCRIPTS
from app.scaffold import main as app_main
from src.utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="flat Full-chain config file to load", required=True)
parser.add_argument(
    "--train-script",
    type=str,
    choices=RETAINED_TRAIN_SCRIPTS,
    help="retained Full-chain training entry",
    required=True,
)
parser.add_argument("--folder", type=str, default=None, help="folder to save logs and checkpoints (overrides config)")
parser.add_argument(
    "--set",
    action="append",
    metavar="KEY=VALUE",
    help="override a resolved config value with a dotted key path; may be repeated",
)


def process_main(fname, train_script):
    import logging

    from src.utils.logging import get_logger

    logger = get_logger(force=True)
    logger.setLevel(logging.INFO)

    logger.info(f"called-params {fname}")

    # Load config (resolving any `extends:` chain into a single fully-merged params dict)
    from app.vjepa_cowa_world_model.training.configs.overrides import apply_dotted_overrides, get_dotted_value
    from app.vjepa_cowa_world_model.training.configs.resolution import load_resolved_training_params

    params = load_resolved_training_params(fname)
    logger.info("loaded params...")
    set_overrides = args.set or []
    apply_dotted_overrides(params, set_overrides)
    for assignment in set_overrides:
        key = assignment.split("=", 1)[0]
        logger.info("applied config override: %s=%r", key, get_dotted_value(params, key))
    if args.folder is not None:
        params["folder"] = args.folder

    # Init distributed (read from environment variables when using torchrun)
    world_size, rank = init_distributed()
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Log config (only rank 0)
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, "params-pretrain.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    # Launch the app with loaded config
    app_main(params["app"], args=params, train_script=train_script)


if __name__ == "__main__":
    args = parser.parse_args()
    process_main(args.fname, args.train_script)
