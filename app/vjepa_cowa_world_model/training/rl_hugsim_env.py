import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from app.vjepa_cowa_world_model.training.config import RLConfig
from app.vjepa_cowa_world_model.utils import prepare_inference_consistent_status_vector, prepare_status_feature
from src.utils.logging import get_logger

logger = get_logger(__name__)


def require_info_field(info: Dict, key: str) -> float:
    """fail-loud: HUGSim 的 info 必须提供该 ego 动力学字段；缺失时报错而非伪造 0.0
    （静止/零速是有语义的状态，不能当作缺失值的默认）。"""
    value = info.get(key, None)
    if value is None:
        raise KeyError(
            f"HUGSIM info missing required field '{key}'; refusing to fabricate 0.0 "
            "(stationary is semantically meaningful)."
        )
    return float(value)


def _resolve_path(path: Optional[str], *, base_dir: Optional[str] = None) -> Optional[str]:
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    if base_dir is not None:
        return os.path.abspath(os.path.join(base_dir, path))
    return os.path.abspath(path)


def _read_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass(frozen=True)
class HUGSIMScenarioSpec:
    name: str
    scenario_path: str
    scene_name: str
    mode: str
    split: str = "both"
    weight: float = 1.0
    enabled: bool = True


def _normalize_split(split: Optional[str]) -> str:
    if split is None:
        return "both"
    normalized = str(split).strip().lower()
    if normalized in {"all", "both", "*", ""}:
        return "both"
    if normalized not in {"train", "eval", "both"}:
        raise ValueError(f"Unsupported scenario split: {split}")
    return normalized


def _split_matches(spec_split: str, requested_split: Optional[str]) -> bool:
    if requested_split is None:
        return True
    if spec_split == "both":
        return True
    return spec_split == requested_split


def _scenario_name_from_cfg(scenario_cfg: dict, scenario_path: str) -> Tuple[str, str, str]:
    scene_name = str(scenario_cfg.get("scene_name") or "unknown_scene")
    mode = str(scenario_cfg.get("mode") or "default")
    default_name = os.path.splitext(os.path.basename(scenario_path))[0]
    return default_name, scene_name, mode


def _scenario_output_name(spec: HUGSIMScenarioSpec) -> str:
    candidate = spec.name or f"{spec.scene_name}-{spec.mode}"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")
    return sanitized or f"{spec.scene_name}-{spec.mode}"


def _single_scenario_spec(rl_config: RLConfig) -> List[HUGSIMScenarioSpec]:
    scenario_path = _resolve_path(rl_config.scenario_path)
    if scenario_path is None:
        raise ValueError("rl.scenario_path or rl.scenario_manifest must be configured")
    scenario_cfg = _read_yaml(scenario_path)
    default_name, scene_name, mode = _scenario_name_from_cfg(scenario_cfg, scenario_path)
    return [
        HUGSIMScenarioSpec(
            name=default_name,
            scenario_path=scenario_path,
            scene_name=scene_name,
            mode=mode,
            split="both",
            weight=1.0,
            enabled=True,
        )
    ]


def _manifest_scenario_specs(rl_config: RLConfig) -> List[HUGSIMScenarioSpec]:
    manifest_path = _resolve_path(rl_config.scenario_manifest)
    if manifest_path is None:
        return []

    manifest_data = _read_yaml(manifest_path)
    if isinstance(manifest_data, dict):
        entries = manifest_data.get("scenarios")
    elif isinstance(manifest_data, list):
        entries = manifest_data
    else:
        raise ValueError(f"Scenario manifest must be a list or dict, got {type(manifest_data)!r}")

    if not entries:
        raise ValueError(f"Scenario manifest is empty: {manifest_path}")

    manifest_dir = os.path.dirname(manifest_path)
    specs: List[HUGSIMScenarioSpec] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Scenario manifest entries must be dicts, got {type(entry)!r}")
        enabled = bool(entry.get("enabled", True))
        if not enabled:
            continue
        raw_scenario_path = entry.get("scenario_path") or entry.get("path")
        scenario_path = _resolve_path(raw_scenario_path, base_dir=manifest_dir)
        if scenario_path is None:
            raise ValueError(f"Manifest entry is missing scenario_path: {entry}")
        scenario_cfg = _read_yaml(scenario_path)
        default_name, scene_name, mode = _scenario_name_from_cfg(scenario_cfg, scenario_path)
        split = _normalize_split(entry.get("split", "both"))
        weight = float(entry.get("weight", 1.0))
        specs.append(
            HUGSIMScenarioSpec(
                name=str(entry.get("name") or default_name),
                scenario_path=scenario_path,
                scene_name=scene_name,
                mode=mode,
                split=split,
                weight=weight,
                enabled=True,
            )
        )

    if not specs:
        raise ValueError(f"No enabled scenarios found in manifest: {manifest_path}")
    return specs


def load_hugsim_scenarios(
    rl_config: RLConfig,
    *,
    split: Optional[str] = None,
    fallback_to_all: bool = False,
) -> List[HUGSIMScenarioSpec]:
    requested_split = _normalize_split(split) if split is not None else None
    all_specs = (
        _manifest_scenario_specs(rl_config) if rl_config.scenario_manifest else _single_scenario_spec(rl_config)
    )

    if requested_split is None:
        return all_specs

    filtered = [spec for spec in all_specs if _split_matches(spec.split, requested_split)]
    if filtered:
        return filtered
    if fallback_to_all:
        return all_specs
    raise ValueError(f"No HUGSIM scenarios matched split='{requested_split}'")


def _ensure_hugsim_importable(repo_root: str):
    sim_root = os.path.join(repo_root, "sim")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if sim_root not in sys.path:
        sys.path.insert(0, sim_root)


def _load_hugsim_cfg(rl_config: RLConfig, scenario_path: Optional[str] = None):
    from omegaconf import OmegaConf  # noqa: WPS433

    repo_root = _resolve_path(rl_config.hugsim_repo_root)
    base_path = _resolve_path(rl_config.base_path)
    camera_path = _resolve_path(rl_config.camera_path)
    kinematic_path = _resolve_path(rl_config.kinematic_path)
    resolved_scenario_path = _resolve_path(scenario_path or rl_config.scenario_path)

    if repo_root is None:
        raise ValueError("rl.hugsim_repo_root must be configured")
    _ensure_hugsim_importable(repo_root)

    if not resolved_scenario_path or not base_path or not camera_path or not kinematic_path:
        raise ValueError("rl.scenario_path/base_path/camera_path/kinematic_path must all be configured")

    scenario_config = OmegaConf.load(resolved_scenario_path)
    base_config = OmegaConf.load(base_path)
    camera_config = OmegaConf.load(camera_path)
    kinematic_config = OmegaConf.load(kinematic_path)
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config},
    )

    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(model_path, "cfg.yaml"))
    cfg.update(model_config)
    cfg.model_path = model_path
    return cfg


def validate_hugsim_scenarios(
    rl_config: RLConfig,
    *,
    split: Optional[str] = None,
    fallback_to_all: bool = False,
) -> List[HUGSIMScenarioSpec]:
    scenarios = load_hugsim_scenarios(rl_config, split=split, fallback_to_all=fallback_to_all)
    errors: List[str] = []

    for spec in scenarios:
        try:
            cfg = _load_hugsim_cfg(rl_config, scenario_path=spec.scenario_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{spec.name}: failed to load merged HUGSIM config ({exc})")
            continue

        scene_dir = str(cfg.model_path)
        for rel_path in ("cfg.yaml", "scene.pth", "ground_param.pkl"):
            full_path = os.path.join(scene_dir, rel_path)
            if not os.path.exists(full_path):
                errors.append(f"{spec.name}: missing scene asset {full_path}")

        realcar_root = str(cfg.base.realcar_path)
        for actor in list(cfg.scenario.plan_list or []):
            if len(actor) < 6:
                errors.append(f"{spec.name}: invalid plan_list entry {actor}")
                continue
            vehicle_id = str(actor[5])
            vehicle_dir = os.path.join(realcar_root, vehicle_id)
            for rel_path in ("gs.pth", "wlh.json"):
                full_path = os.path.join(vehicle_dir, rel_path)
                if not os.path.exists(full_path):
                    errors.append(f"{spec.name}: missing vehicle asset {full_path}")

    if errors:
        raise FileNotFoundError("HUGSIM scenario asset preflight failed:\n- " + "\n- ".join(errors))
    return scenarios


def create_hugsim_env(rl_config: RLConfig, output_dir: str, scenario_path: Optional[str] = None):
    cfg = _load_hugsim_cfg(rl_config, scenario_path=scenario_path)
    os.makedirs(output_dir, exist_ok=True)

    import gymnasium  # noqa: WPS433
    import hugsim_env  # noqa: F401,WPS433

    env = gymnasium.make("hugsim_env/HUGSim-v0", cfg=cfg, output=output_dir)
    return env, cfg


class HUGSIMScenarioManager:
    def __init__(
        self,
        rl_config: RLConfig,
        output_root: str,
        *,
        split: Optional[str],
        seed: int = 0,
        fallback_to_all: bool = False,
    ):
        self.rl_config = rl_config
        self.output_root = output_root
        self.scenarios = validate_hugsim_scenarios(
            rl_config,
            split=split,
            fallback_to_all=fallback_to_all,
        )
        self.rng = random.Random(seed)
        self.single_scenario_passthrough = rl_config.scenario_manifest is None and len(self.scenarios) == 1
        self.current_scenario: Optional[HUGSIMScenarioSpec] = None
        self.current_env = None
        self.current_cfg = None
        self.current_output_dir: Optional[str] = None

    def sample(self) -> HUGSIMScenarioSpec:
        if not self.scenarios:
            raise ValueError("No HUGSIM scenarios are available for sampling")
        weights = [max(0.0, float(spec.weight)) for spec in self.scenarios]
        if sum(weights) <= 0.0:
            raise ValueError("Scenario weights must sum to a positive value")
        return self.rng.choices(self.scenarios, weights=weights, k=1)[0]

    def activate(self, scenario: HUGSIMScenarioSpec):
        if (
            self.current_env is not None
            and self.current_scenario is not None
            and self.current_scenario.scenario_path == scenario.scenario_path
        ):
            return self.current_env, self.current_cfg, self.current_scenario, self.current_output_dir

        self.close()
        if self.single_scenario_passthrough:
            output_dir = self.output_root
        else:
            output_dir = os.path.join(self.output_root, _scenario_output_name(scenario))
        env, cfg = create_hugsim_env(self.rl_config, output_dir, scenario_path=scenario.scenario_path)
        self.current_env = env
        self.current_cfg = cfg
        self.current_scenario = scenario
        self.current_output_dir = output_dir
        return env, cfg, scenario, output_dir

    def close(self):
        if self.current_env is not None:
            try:
                self.current_env.close()
            except Exception as exc:  # noqa: BLE001
                # 关闭失败不阻断后续 teardown，但必须显式记录而非静默吞掉（fail-loud：不静默）。
                logger.warning("Error while closing HUGSim env (continuing teardown): %s", exc)
        self.current_env = None
        self.current_cfg = None
        self.current_scenario = None
        self.current_output_dir = None


class HUGSIMObservationAdapter:
    def __init__(self, crop_size: int, camera_name: str = "CAM_FRONT"):
        self.crop_size = int(crop_size)
        self.camera_name = camera_name
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def encode_command(self, command_value) -> np.ndarray:
        cmd = np.zeros(4, dtype=np.float32)
        if command_value is None:
            cmd[0] = 1.0
            return cmd

        try:
            cmd_idx = int(command_value)
        except (TypeError, ValueError):
            if isinstance(command_value, str):
                normalized = command_value.strip().lower()
                mapping = {
                    "straight": 0,
                    "go_straight": 0,
                    "left": 1,
                    "turn_left": 1,
                    "right": 2,
                    "turn_right": 2,
                    "uturn": 3,
                    "u_turn": 3,
                }
                cmd_idx = mapping.get(normalized, 0)
            else:
                cmd_idx = 0

        cmd[min(max(cmd_idx, 0), 3)] = 1.0
        return cmd

    def preprocess_rgb(self, observation) -> torch.Tensor:
        rgb_dict = observation["rgb"]
        if self.camera_name in rgb_dict:
            rgb = rgb_dict[self.camera_name]
        else:
            fallback_key = sorted(rgb_dict.keys())[0]
            rgb = rgb_dict[fallback_key]
        tensor = torch.from_numpy(rgb).to(torch.float32) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor.squeeze(0)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def build_status_feature(self, info) -> torch.Tensor:
        ego_pos = info.get("ego_pos", [0.0, 0.0, 0.0])
        ego_rot = info.get("ego_rot", [0.0, 0.0, 0.0])
        vel = require_info_field(info, "ego_velo")
        steer = require_info_field(info, "ego_steer")
        x = float(ego_pos[0])
        y = float(ego_pos[2] if len(ego_pos) > 2 else ego_pos[1])
        yaw = float(ego_rot[1] if len(ego_rot) > 1 else 0.0)
        return torch.tensor([vel, steer, yaw, x, y], dtype=torch.float32)

    def adapt(self, observation, info) -> Dict[str, torch.Tensor]:
        return {
            "rgb": self.preprocess_rgb(observation),
            "status_feature": self.build_status_feature(info),
            "command": torch.from_numpy(self.encode_command(info.get("command"))),
        }


def get_action_bounds(env) -> Tuple[torch.Tensor, torch.Tensor]:
    steer_space = env.action_space["steer_rate"]
    acc_space = env.action_space["acc"]
    steer_low = float(np.asarray(steer_space.low).reshape(-1)[0])
    steer_high = float(np.asarray(steer_space.high).reshape(-1)[0])
    acc_low = float(np.asarray(acc_space.low).reshape(-1)[0])
    acc_high = float(np.asarray(acc_space.high).reshape(-1)[0])
    low = torch.tensor([steer_low, acc_low], dtype=torch.float32)
    high = torch.tensor([steer_high, acc_high], dtype=torch.float32)
    return low, high


def tensor_action_to_env(action_tensor):
    action_np = action_tensor.detach().cpu().numpy()
    return {
        "steer_rate": float(action_np[0]),
        "acc": float(action_np[1]),
    }


def rollout_info_to_metrics(reward, terminated, truncated, info):
    return {
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "collision": float(bool(info.get("collision", False))),
        "route_completion": float(info.get("rc", 0.0)),
    }


def info_to_state_vector(info) -> torch.Tensor:
    ego_pos = info.get("ego_pos", [0.0, 0.0, 0.0])
    ego_rot = info.get("ego_rot", [0.0, 0.0, 0.0])
    velo = require_info_field(info, "ego_velo")
    yaw = float(ego_rot[1] if len(ego_rot) > 1 else 0.0)
    state = torch.tensor(
        [
            float(ego_pos[0]),
            float(ego_pos[2] if len(ego_pos) > 2 else ego_pos[1]),
            float(ego_pos[1] if len(ego_pos) > 1 else 0.0),
            float(ego_rot[0] if len(ego_rot) > 0 else 0.0),
            0.0,
            yaw,
            velo,
        ],
        dtype=torch.float32,
    )
    return state


def build_status_feature_from_history(
    state_history,
    *,
    status_mode,
    num_context_frames,
    device,
    action_dim=7,
    predictor_inference_consistent=False,
    num_observed_frames=None,
):
    if len(state_history) == 0:
        raise ValueError("state_history is empty")

    target_len = max(1, int(num_context_frames))
    states = build_state_history_tensor(state_history, target_len).unsqueeze(0).to(device)
    if predictor_inference_consistent:
        observed = max(1, int(num_observed_frames or target_len))
        observed = min(observed, states.shape[1])
        return prepare_inference_consistent_status_vector(states, num_observed=observed)

    dummy_actions = torch.zeros(
        1,
        max(1, target_len - 1),
        action_dim,
        device=device,
        dtype=states.dtype,
    )
    status_feature = prepare_status_feature(
        states,
        dummy_actions,
        status_mode=status_mode,
        num_context_frames=target_len,
    )
    return status_feature


def build_state_history_tensor(state_history, target_len):
    if len(state_history) == 0:
        raise ValueError("state_history is empty")

    selected = [state.clone() for state in list(state_history)[-target_len:]]
    while len(selected) < target_len:
        selected.insert(0, selected[0].clone())
    return torch.stack(selected, dim=0)


def build_frame_history_tensor(frame_history, num_context_frames):
    if len(frame_history) == 0:
        raise ValueError("frame_history is empty")

    target_len = max(1, int(num_context_frames))
    selected = [frame.clone() for frame in list(frame_history)[-target_len:]]
    while len(selected) < target_len:
        selected.insert(0, selected[0].clone())
    return torch.stack(selected, dim=0)
