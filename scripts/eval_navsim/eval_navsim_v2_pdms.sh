#!/bin/bash
# CVoI Full-only official NavSim scorer boundary.
#
# Callers must enable exactly one retained mode:
#   CVOI_DIRECT_EPDMS=1          for the NavTest Full-controller evaluation
#   CVOI_MANUAL_NAVTRAIN_GATE=1  for one NavTrain Oracle horizon
#
# The script always uses the stage12 agent boundary with one worker and no
# process pool.  Model authority is supplied by the mode-specific projected
# config; no caller-selectable forward mode or proposal checkpoint is accepted.

set -euo pipefail

DIRECT_EPDMS_MODE="${CVOI_DIRECT_EPDMS:-0}"
MANUAL_NAVTRAIN_GATE_MODE="${CVOI_MANUAL_NAVTRAIN_GATE:-0}"
if [[ "${DIRECT_EPDMS_MODE}" != "0" && "${DIRECT_EPDMS_MODE}" != "1" ]]; then
    echo "CVOI_DIRECT_EPDMS must be exactly 0 or 1" >&2
    exit 2
fi
if [[ "${MANUAL_NAVTRAIN_GATE_MODE}" != "0" && "${MANUAL_NAVTRAIN_GATE_MODE}" != "1" ]]; then
    echo "CVOI_MANUAL_NAVTRAIN_GATE must be exactly 0 or 1" >&2
    exit 2
fi
if [[ "${DIRECT_EPDMS_MODE}" == "${MANUAL_NAVTRAIN_GATE_MODE}" ]]; then
    echo "exactly one retained mode must be enabled: CVOI_DIRECT_EPDMS or CVOI_MANUAL_NAVTRAIN_GATE" >&2
    exit 2
fi
if [[ "${DIRECT_EPDMS_MODE}" == "1" ]]; then
    : "${CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH:?missing CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH}"
    if [[ "${CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH}" != /* ]]; then
        echo "CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH must be absolute" >&2
        exit 2
    fi
    for VARIABLE_NAME in ${!CVOI_DIRECT_EPDMS_@}; do
        if [[ "${VARIABLE_NAME}" != "CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH" ]]; then
            echo "unsupported direct EPDMS payload variable: ${VARIABLE_NAME}" >&2
            exit 2
        fi
    done
    DIRECT_CONFLICT=""
    for VARIABLE_NAME in \
        CVOI_MANUAL_NAVTRAIN_GATE \
        CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH \
        ${!CVOI_FORMAL_V2_NAVSIM_E120_@}; do
        if [[ ${!VARIABLE_NAME+x} == x ]]; then
            DIRECT_CONFLICT="${VARIABLE_NAME}"
            break
        fi
    done
    if [[ -n "${DIRECT_CONFLICT}" ]]; then
        echo "direct EPDMS and manual NavTrain Gate modes are mutually exclusive: ${DIRECT_CONFLICT}" >&2
        exit 2
    fi
elif [[ ${CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH+x} == x ]]; then
    echo "CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH requires CVOI_DIRECT_EPDMS=1" >&2
    exit 2
fi

# ---- 参数解析 ----
if [[ "${DIRECT_EPDMS_MODE}" == "1" ]]; then
    CHECKPOINT="${1:-}"
    TRAINING_CONFIG="${2:-}"
    EXPERIMENT_NAME="${3:-cvoi_direct_epdms}"
else
    CHECKPOINT="${1:?Error: 请提供 checkpoint 路径作为第一个参数}"
    TRAINING_CONFIG="${2:?Error: 请提供训练配置路径作为第二个参数}"
    EXPERIMENT_NAME="${3:-eval_vjepa_pdms_v2}"
fi

# ---- 项目根目录 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- 必需的 NavSim 环境根 ----
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NavSim dataset root}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to the NavSim experiment root}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the NuPlan maps root}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT to the official NavSim devkit root}"

# ---- PYTHONPATH: 确保项目根目录和 navsim devkit 在路径中 ----
CALLER_PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${NAVSIM_DEVKIT_ROOT}:${CALLER_PYTHONPATH}"

# ---- Metric cache 路径 ----
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache}"

# ---- 评估数据集分割 ----
TRAIN_TEST_SPLIT="navtest"

# ---- Retained scorer configuration ----
MAX_WORKERS="1"
USE_PROCESS_POOL="false"
FORWARD_MODE="stage12"
PROPOSAL_CHECKPOINT=""
PYTHON_BIN="${PYTHON_BIN:-python3}"

AGENT_OVERRIDES=()
AGENT_PATH_OVERRIDES=(
    "agent.checkpoint_path=${CHECKPOINT}"
    "agent.training_config_path=${TRAINING_CONFIG}"
)
OUTPUT_OVERRIDE=()
HYDRA_CONFIG_ARGS=(--config-dir "${PROJECT_ROOT}/configs/navsim/cvoi_manual")
AGENT_CONFIG_NAME="cvoi_manual_vjepa_world_model_agent"

if [[ "${MANUAL_NAVTRAIN_GATE_MODE}" == "1" ]]; then
    MANUAL_CONFLICT=""
    if [[ ${CVOI_FORMAL_V2_NAVSIM_E120+x} == x ]]; then
        MANUAL_CONFLICT="CVOI_FORMAL_V2_NAVSIM_E120"
    fi
    for VARIABLE_NAME in ${!CVOI_FORMAL_V2_NAVSIM_E120_@}; do
        MANUAL_CONFLICT="${VARIABLE_NAME}"
        break
    done
    if [[ -n "${MANUAL_CONFLICT}" ]]; then
        echo "manual NavTrain Gate and retired Formal variables are mutually exclusive: ${MANUAL_CONFLICT}" >&2
        exit 2
    fi
elif [[ -n "${CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH:-}" ]]; then
    echo "manual config path requires CVOI_MANUAL_NAVTRAIN_GATE=1" >&2
    exit 2
fi

if [[ "${DIRECT_EPDMS_MODE}" == "1" ]]; then
    NAVSIM_OUTPUT_DIR="${NAVSIM_OUTPUT_DIR:?direct EPDMS scoring requires NAVSIM_OUTPUT_DIR}"
    AGENT_OVERRIDES=(
        "++agent.cvoi_direct_epdms_config_path=${CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH}"
    )
    AGENT_PATH_OVERRIDES=()
    OUTPUT_OVERRIDE=("output_dir=${NAVSIM_OUTPUT_DIR}")
elif [[ "${MANUAL_NAVTRAIN_GATE_MODE}" == "1" ]]; then
    : "${CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH:?missing CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH}"
    NAVSIM_OUTPUT_DIR="${NAVSIM_OUTPUT_DIR:?manual NavTrain Gate scoring requires NAVSIM_OUTPUT_DIR}"
    TRAIN_TEST_SPLIT="navtrain"
    AGENT_OVERRIDES=(
        "++agent.cvoi_manual_navtrain_gate_config_path=${CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH}"
    )
    OUTPUT_OVERRIDE=("output_dir=${NAVSIM_OUTPUT_DIR}")
fi

echo "=============================================="
echo " NavSim V2 PDMS Evaluation (one-stage)"
echo "=============================================="
echo " Checkpoint:       ${CHECKPOINT}"
echo " Training Config:  ${TRAINING_CONFIG}"
echo " Experiment:       ${EXPERIMENT_NAME}"
echo " Devkit:           ${NAVSIM_DEVKIT_ROOT}"
echo " Data Root:        ${OPENSCENE_DATA_ROOT}"
echo " Metric Cache:     ${METRIC_CACHE_PATH}"
echo " Split:            ${TRAIN_TEST_SPLIT}"
echo " Workers:          ${MAX_WORKERS}"
echo " Forward Mode:     ${FORWARD_MODE}"
echo " Python:           ${PYTHON_BIN}"
echo "=============================================="

# ---- 运行评估 ----
# 使用官方 run_pdm_score_one_stage.py (单阶段 PDMS)
# traffic_agents 默认为 "non_reactive" (在 default_common.yaml 中已配置)
"${PYTHON_BIN}" -u "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage.py" \
    "${HYDRA_CONFIG_ARGS[@]}" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent="${AGENT_CONFIG_NAME}" \
    ${AGENT_PATH_OVERRIDES[@]+"${AGENT_PATH_OVERRIDES[@]}"} \
    ++agent.forward_mode="${FORWARD_MODE}" \
    ++agent.proposal_checkpoint_path="${PROPOSAL_CHECKPOINT}" \
    worker=single_machine_thread_pool \
    worker.max_workers="${MAX_WORKERS}" \
    worker.use_process_pool="${USE_PROCESS_POOL}" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    experiment_name="${EXPERIMENT_NAME}" \
    "${OUTPUT_OVERRIDE[@]}" \
    "${AGENT_OVERRIDES[@]}"
