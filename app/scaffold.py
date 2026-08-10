# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

RETAINED_TRAIN_SCRIPTS = (
    "train_cvoi_offline",
    "train_latent_predictor",
    "train_predictor_rollout_planner",
)


def resolve_train_script(train_script):
    """Validate one exact Full-chain training entry."""
    if train_script not in RETAINED_TRAIN_SCRIPTS:
        raise ValueError(f"unsupported --train-script {train_script!r}; expected one of {RETAINED_TRAIN_SCRIPTS}")
    return train_script


def main(app, args, train_script, resume_preempt=False):

    train_script = resolve_train_script(train_script)
    logger.info(f"Running pre-training of app: {app}, train_script: {train_script}")
    return importlib.import_module(f"app.{app}.{train_script}").main(args=args, resume_preempt=resume_preempt)
