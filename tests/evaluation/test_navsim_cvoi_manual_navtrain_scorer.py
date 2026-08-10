"""Proof-free manual NavTrain scorer integration for the NavSim Agent."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.vjepa_cowa_world_model.evaluation.cvoi_navsim_identity import observation_key_tensor, unsigned_seed_tensor
from app.vjepa_cowa_world_model.training import cvoi_manual_lineage
from app.vjepa_cowa_world_model.training.cvoi_execution import cvoi_sample_seed
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import FORMAL_V2_NAVSIM_E120_LAMBDA_GRID
from app.vjepa_cowa_world_model.training.cvoi_manual_navtrain_oracle import ManualOracleSource
from tests.evaluation.test_navsim_stage3_alignment import _load_navsim_agent


def _module_globals() -> dict[str, object]:
    return _load_navsim_agent().initialize.__globals__


def _planner_payload(stage: str, branch_id: str, *, encoder_value: float = 1.0) -> dict[str, object]:
    states = {
        "encoder": {"weight": torch.tensor([encoder_value])},
        "predictor": {"weight": torch.tensor([2.0])},
        "planner": {"weight": torch.tensor([3.0])},
    }
    return {
        "stage": stage,
        "lineage": {"stage": stage, "branch_id": branch_id},
        **states,
        "role_state_shapes": {
            role: {key: list(value.shape) for key, value in state.items()} for role, state in states.items()
        },
    }


def _manual_artifact_paths(
    *,
    lineage: str = "p1_full",
    full_root: Path | None = None,
    no_cf_root: Path | None = None,
) -> dict[str, Path]:
    full_root = cvoi_manual_lineage.CVOI_MANUAL_FULL_RESULTS_ROOT if full_root is None else full_root
    no_cf_root = cvoi_manual_lineage.CVOI_MANUAL_ABLATION_RESULTS_ROOT / "no_cf" if no_cf_root is None else no_cf_root
    if lineage == "p1_full":
        value_root = full_root
    elif lineage == "p1_no_cf":
        value_root = no_cf_root
    else:
        raise ValueError(f"unsupported test lineage: {lineage!r}")
    return {
        "p0_planner_checkpoint": full_root / "handoff/p0_selected.pt",
        "field_checkpoint": value_root / "handoff/field.pt",
        "calibration_checkpoint": value_root / "handoff/calibration.pt",
        "p1_planner_checkpoint": value_root / "handoff/p1_selected.pt",
        "stop_checkpoint": value_root / "handoff/stop.pt",
    }


@pytest.mark.parametrize("value", ["", " ", 7])
def test_manual_navtrain_constructor_requires_a_nonempty_config_path(value: object) -> None:
    agent_class = _load_navsim_agent()

    with pytest.raises(ValueError, match="cvoi_manual_navtrain_gate_config_path.*non-empty"):
        agent_class(device="cpu", cvoi_manual_navtrain_gate_config_path=value)


def test_manual_navtrain_constructor_has_no_legacy_official_surface() -> None:
    agent_class = _load_navsim_agent()
    parameters = inspect.signature(agent_class.__init__).parameters

    assert not any(name.startswith("cvoi_official_") for name in parameters)
    assert {
        "CvoiOfficialEffectivePolicy",
        "resolve_cvoi_official_effective_policy",
        "open_cvoi_navsim_artifact_handles",
    }.isdisjoint(_module_globals())


def test_manual_navtrain_constructor_preserves_the_exact_requested_path() -> None:
    agent_class = _load_navsim_agent()
    path = str(Path("/manual/scorer_config.json"))

    agent = agent_class(device="cpu", cvoi_manual_navtrain_gate_config_path=path)

    assert agent._cvoi_manual_navtrain_gate_config_path == path
    assert not hasattr(agent, "_cvoi_official_request")
    assert not hasattr(agent, "_cvoi_official_policy")
    assert agent._cvoi_manual_runtime is None


def test_manual_navtrain_runtime_has_no_retired_policy_registry_dependency() -> None:
    globals_dict = _module_globals()

    assert {
        "CvoiFormalV2NavSimE120Policy",
        "build_cvoi_formal_v2_navsim_e120_policy_registry",
        "_resolve_manual_navtrain_policy_point",
    }.isdisjoint(globals_dict)
    assert globals_dict["_ManualNavTrainRuntimeContext"]._fields[:3] == (
        "policy_id",
        "lineage",
        "planner_stage",
    )


def test_manual_navtrain_preflight_reads_all_five_direct_artifacts_with_exact_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    globals_dict = _module_globals()
    preflight = globals_dict["_read_manual_navtrain_artifacts"]
    artifacts = _manual_artifact_paths()
    planner_calls: list[Path] = []
    value_calls: list[tuple[Path, str, str, object]] = []

    def planner_reader(path: Path) -> dict[str, object]:
        planner_calls.append(path)
        if path == artifacts["p0_planner_checkpoint"]:
            return _planner_payload("p0", "p0_uniform")
        return _planner_payload("p1", "p1_full")

    def value_reader(
        path: Path,
        *,
        required_phase: str,
        required_branch_id: str,
        map_location: object,
    ) -> dict[str, object]:
        value_calls.append((path, required_phase, required_branch_id, map_location))
        return {
            "phase": required_phase,
            "branch_id": required_branch_id,
            "architecture": {"embed_dim": 4, "hidden_dim": 8},
            "state_dict": {"weight": torch.ones(1)},
        }

    monkeypatch.setitem(globals_dict, "read_formal_v2_navsim_e120_direct_checkpoint", planner_reader)
    monkeypatch.setitem(globals_dict, "read_cvoi_navsim_e120_direct_value_checkpoint", value_reader)
    monkeypatch.setitem(
        globals_dict,
        "resolve_formal_v2_navsim_e120_selected_checkpoint",
        lambda path, **kwargs: path,
    )

    bundle = preflight(artifacts, expected_lineage="p1_full")

    assert planner_calls == [
        artifacts["p0_planner_checkpoint"],
        artifacts["p1_planner_checkpoint"],
    ]
    assert value_calls == [
        (artifacts["field_checkpoint"], "field_warmup", "field_full", "cpu"),
        (artifacts["calibration_checkpoint"], "field_calibrated", "calibration_full", "cpu"),
        (artifacts["stop_checkpoint"], "stop_calibrated", "stop_full", "cpu"),
    ]
    assert bundle.p1_checkpoint["stage"] == "p1"


def test_manual_navtrain_no_cf_preflight_uses_full_p0_and_exact_no_cf_value_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    globals_dict = _module_globals()
    preflight = globals_dict["_read_manual_navtrain_artifacts"]
    full_root = (tmp_path / "configured-full").resolve()
    no_cf_root = (tmp_path / "configured-ablation/no_cf").resolve()
    artifacts = _manual_artifact_paths(
        lineage="p1_no_cf",
        full_root=full_root,
        no_cf_root=no_cf_root,
    )
    monkeypatch.setattr(
        cvoi_manual_lineage,
        "CVOI_MANUAL_FULL_RESULTS_ROOT",
        (tmp_path / "poisoned-default-full").resolve(),
    )
    monkeypatch.setattr(
        cvoi_manual_lineage,
        "CVOI_MANUAL_ABLATION_RESULTS_ROOT",
        (tmp_path / "poisoned-default-ablation").resolve(),
    )
    planner_calls: list[Path] = []
    value_calls: list[tuple[Path, str, str, object]] = []
    selected_calls: list[tuple[Path, Path, str]] = []

    def selected_reader(path: Path, *, results_root: Path, stage: str) -> Path:
        selected_calls.append((path, results_root, stage))
        return path

    def planner_reader(path: Path) -> dict[str, object]:
        planner_calls.append(path)
        if path == artifacts["p0_planner_checkpoint"]:
            return _planner_payload("p0", "p0_uniform")
        return _planner_payload("p1", "p1_no_cf")

    def value_reader(
        path: Path,
        *,
        required_phase: str,
        required_branch_id: str,
        map_location: object,
    ) -> dict[str, object]:
        value_calls.append((path, required_phase, required_branch_id, map_location))
        return {
            "phase": required_phase,
            "branch_id": required_branch_id,
            "architecture": {"embed_dim": 4, "hidden_dim": 8},
            "state_dict": {"weight": torch.ones(1)},
        }

    monkeypatch.setitem(globals_dict, "resolve_formal_v2_navsim_e120_selected_checkpoint", selected_reader)
    monkeypatch.setitem(globals_dict, "read_formal_v2_navsim_e120_direct_checkpoint", planner_reader)
    monkeypatch.setitem(globals_dict, "read_cvoi_navsim_e120_direct_value_checkpoint", value_reader)

    bundle = preflight(artifacts, expected_lineage="p1_no_cf")

    assert selected_calls == [
        (artifacts["p0_planner_checkpoint"], full_root, "p0"),
        (artifacts["p1_planner_checkpoint"], no_cf_root, "p1"),
    ]
    assert planner_calls == [
        artifacts["p0_planner_checkpoint"],
        artifacts["p1_planner_checkpoint"],
    ]
    assert value_calls == [
        (artifacts["field_checkpoint"], "field_warmup", "field_no_cf", "cpu"),
        (artifacts["calibration_checkpoint"], "field_calibrated", "calibration_no_cf", "cpu"),
        (artifacts["stop_checkpoint"], "stop_calibrated", "stop_no_cf", "cpu"),
    ]
    assert bundle.p1_checkpoint["lineage"]["branch_id"] == "p1_no_cf"


@pytest.mark.parametrize(
    "mutation",
    [
        "full_p1",
        "full_calibration",
        "wrong_root_p0",
        "encoder_state",
        "value_architecture",
    ],
)
def test_manual_navtrain_no_cf_preflight_rejects_cross_lineage_or_incompatible_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    globals_dict = _module_globals()
    preflight = globals_dict["_read_manual_navtrain_artifacts"]
    artifacts = _manual_artifact_paths(lineage="p1_no_cf")
    if mutation == "wrong_root_p0":
        artifacts["p0_planner_checkpoint"] = (
            cvoi_manual_lineage.CVOI_MANUAL_ABLATION_RESULTS_ROOT / "no_cf/handoff/p0_selected.pt"
        )

    def planner_reader(path: Path) -> dict[str, object]:
        if path == artifacts["p0_planner_checkpoint"]:
            return _planner_payload("p0", "p0_uniform")
        branch_id = "p1_full" if mutation == "full_p1" else "p1_no_cf"
        return _planner_payload(
            "p1",
            branch_id,
            encoder_value=9.0 if mutation == "encoder_state" else 1.0,
        )

    def value_reader(
        path: Path,
        *,
        required_phase: str,
        required_branch_id: str,
        map_location: object,
    ) -> dict[str, object]:
        del map_location
        branch_id = (
            "calibration_full"
            if mutation == "full_calibration" and path == artifacts["calibration_checkpoint"]
            else required_branch_id
        )
        hidden_dim = 16 if mutation == "value_architecture" and path == artifacts["stop_checkpoint"] else 8
        return {
            "phase": required_phase,
            "branch_id": branch_id,
            "architecture": {"embed_dim": 4, "hidden_dim": hidden_dim},
            "state_dict": {"weight": torch.ones(1)},
        }

    monkeypatch.setitem(
        globals_dict,
        "resolve_formal_v2_navsim_e120_selected_checkpoint",
        lambda path, **kwargs: path,
    )
    monkeypatch.setitem(globals_dict, "read_formal_v2_navsim_e120_direct_checkpoint", planner_reader)
    monkeypatch.setitem(globals_dict, "read_cvoi_navsim_e120_direct_value_checkpoint", value_reader)

    with pytest.raises(ValueError):
        preflight(artifacts, expected_lineage="p1_no_cf")


@pytest.mark.parametrize("lineage", ["p1_hazard_only", "p1_quality_only", "p1_unknown"])
def test_manual_navtrain_preflight_rejects_lineages_without_a_manual_navtrain_stop_oracle(
    monkeypatch: pytest.MonkeyPatch,
    lineage: str,
) -> None:
    globals_dict = _module_globals()
    preflight = globals_dict["_read_manual_navtrain_artifacts"]

    def forbidden_io(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("unsupported lineage must be rejected before artifact I/O")

    for name in (
        "resolve_formal_v2_navsim_e120_selected_checkpoint",
        "read_formal_v2_navsim_e120_direct_checkpoint",
        "read_cvoi_navsim_e120_direct_value_checkpoint",
    ):
        monkeypatch.setitem(globals_dict, name, forbidden_io)

    with pytest.raises(ValueError):
        preflight(_manual_artifact_paths(lineage="p1_no_cf"), expected_lineage=lineage)


def test_manual_navtrain_no_cf_selected_links_stay_inside_their_distinct_lineage_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    globals_dict = _module_globals()
    preflight = globals_dict["_read_manual_navtrain_artifacts"]
    full_root = (tmp_path / "cvoi_manual_full").resolve()
    ablation_root = (tmp_path / "cvoi_manual_ablation").resolve()
    no_cf_root = ablation_root / "no_cf"
    for directory in (
        full_root / "handoff",
        full_root / "p0",
        full_root / "p1",
        no_cf_root / "handoff",
        no_cf_root / "p1",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_FULL_RESULTS_ROOT", full_root)
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_ABLATION_RESULTS_ROOT", ablation_root)
    for name, value in (
        ("CVOI_MANUAL_FULL_RESULTS_ROOT", full_root),
        ("CVOI_MANUAL_ABLATION_RESULTS_ROOT", ablation_root),
    ):
        if name in globals_dict:
            monkeypatch.setitem(globals_dict, name, value)

    p0_target = full_root / "p0/candidate.pt"
    p1_target = no_cf_root / "p1/candidate.pt"
    cross_root_target = full_root / "p1/candidate.pt"
    for target in (p0_target, p1_target, cross_root_target):
        target.write_bytes(target.name.encode("ascii"))
    p0_selected = full_root / "handoff/p0_selected.pt"
    p1_selected = no_cf_root / "handoff/p1_selected.pt"
    p0_selected.symlink_to(p0_target)
    p1_selected.symlink_to(p1_target)
    artifacts = _manual_artifact_paths(
        lineage="p1_no_cf",
        full_root=full_root,
        no_cf_root=no_cf_root,
    )
    planner_reads: list[Path] = []

    def planner_reader(path: Path) -> dict[str, object]:
        planner_reads.append(path)
        return _planner_payload("p0", "p0_uniform") if path == p0_target else _planner_payload("p1", "p1_no_cf")

    def value_reader(
        path: Path,
        *,
        required_phase: str,
        required_branch_id: str,
        map_location: object,
    ) -> dict[str, object]:
        del path, map_location
        return {
            "phase": required_phase,
            "branch_id": required_branch_id,
            "architecture": {"embed_dim": 4, "hidden_dim": 8},
            "state_dict": {"weight": torch.ones(1)},
        }

    monkeypatch.setitem(globals_dict, "read_formal_v2_navsim_e120_direct_checkpoint", planner_reader)
    monkeypatch.setitem(globals_dict, "read_cvoi_navsim_e120_direct_value_checkpoint", value_reader)

    preflight(artifacts, expected_lineage="p1_no_cf")
    assert planner_reads == [p0_target, p1_target]

    p1_selected.unlink()
    p1_selected.symlink_to(cross_root_target)
    with pytest.raises(ValueError, match="p1|P1|root|contained"):
        preflight(artifacts, expected_lineage="p1_no_cf")


def test_manual_navtrain_preflight_resolves_selected_links_inside_their_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    globals_dict = _module_globals()
    preflight = globals_dict["_read_manual_navtrain_artifacts"]
    results_root = tmp_path.resolve()
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_FULL_RESULTS_ROOT", results_root)
    for name, value in (
        ("CVOI_MANUAL_FULL_RESULTS_ROOT", results_root),
        ("CVOI_MANUAL_ABLATION_RESULTS_ROOT", results_root / "ablations"),
    ):
        if name in globals_dict:
            monkeypatch.setitem(globals_dict, name, value)
    handoff = results_root / "handoff"
    p0_dir = results_root / "p0"
    p1_dir = results_root / "p1"
    handoff.mkdir()
    p0_dir.mkdir()
    p1_dir.mkdir()
    p0_target = p0_dir / "candidate.pt"
    p1_target = p1_dir / "candidate.pt"
    p0_target.write_bytes(b"p0")
    p1_target.write_bytes(b"p1")
    (handoff / "p0_selected.pt").symlink_to(p0_target)
    (handoff / "p1_selected.pt").symlink_to(p1_target)
    artifacts = {
        "p0_planner_checkpoint": handoff / "p0_selected.pt",
        "field_checkpoint": handoff / "field.pt",
        "calibration_checkpoint": handoff / "calibration.pt",
        "p1_planner_checkpoint": handoff / "p1_selected.pt",
        "stop_checkpoint": handoff / "stop.pt",
    }
    planner_reads: list[Path] = []

    def planner_reader(path: Path) -> dict[str, object]:
        planner_reads.append(path)
        return _planner_payload("p0", "p0_uniform") if path == p0_target else _planner_payload("p1", "p1_full")

    def value_reader(
        path: Path,
        *,
        required_phase: str,
        required_branch_id: str,
        map_location: object,
    ) -> dict[str, object]:
        del path, map_location
        return {
            "phase": required_phase,
            "branch_id": required_branch_id,
            "architecture": {"embed_dim": 4, "hidden_dim": 8},
            "state_dict": {"weight": torch.ones(1)},
        }

    monkeypatch.setitem(globals_dict, "read_formal_v2_navsim_e120_direct_checkpoint", planner_reader)
    monkeypatch.setitem(globals_dict, "read_cvoi_navsim_e120_direct_value_checkpoint", value_reader)

    preflight(artifacts, expected_lineage="p1_full")

    assert planner_reads == [p0_target, p1_target]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("p0_branch", "p0_uniform"),
        ("missing_role", "roles"),
        ("declared_shape", "shape"),
        ("encoder_value", "encoder"),
        ("architecture", "architecture"),
    ],
)
def test_manual_navtrain_preflight_fails_closed_on_direct_artifact_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    globals_dict = _module_globals()
    preflight = globals_dict["_read_manual_navtrain_artifacts"]
    artifacts = _manual_artifact_paths()

    def planner_reader(path: Path) -> dict[str, object]:
        if path == artifacts["p0_planner_checkpoint"]:
            payload = _planner_payload("p0", "wrong" if mutation == "p0_branch" else "p0_uniform")
            if mutation == "missing_role":
                del payload["planner"]
            if mutation == "declared_shape":
                payload["role_state_shapes"]["encoder"]["weight"] = [2]
            return payload
        return _planner_payload("p1", "p1_full", encoder_value=9.0 if mutation == "encoder_value" else 1.0)

    def value_reader(
        path: Path,
        *,
        required_phase: str,
        required_branch_id: str,
        map_location: object,
    ) -> dict[str, object]:
        del map_location
        hidden_dim = 16 if mutation == "architecture" and path == artifacts["stop_checkpoint"] else 8
        return {
            "phase": required_phase,
            "branch_id": required_branch_id,
            "architecture": {"embed_dim": 4, "hidden_dim": hidden_dim},
            "state_dict": {"weight": torch.ones(1)},
        }

    monkeypatch.setitem(globals_dict, "read_formal_v2_navsim_e120_direct_checkpoint", planner_reader)
    monkeypatch.setitem(globals_dict, "read_cvoi_navsim_e120_direct_value_checkpoint", value_reader)
    monkeypatch.setitem(
        globals_dict,
        "resolve_formal_v2_navsim_e120_selected_checkpoint",
        lambda path, **kwargs: path,
    )

    with pytest.raises(ValueError, match=message):
        preflight(artifacts, expected_lineage="p1_full")


@pytest.mark.parametrize("horizon", [0, 3])
def test_manual_navtrain_compute_writes_only_the_exact_nine_trace_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    horizon: int,
) -> None:
    agent_class = _load_navsim_agent()
    globals_dict = agent_class.compute_trajectory.__globals__
    runtime_type = globals_dict["_ManualNavTrainRuntimeContext"]
    guidance_steps = 0 if horizon == 0 else 2
    policy_id = f"p1_full__fixed_h{horizon}_k{guidance_steps}"
    trace_dir = (tmp_path / "traces").resolve()
    trace_dir.mkdir()
    trace_stat = trace_dir.stat(follow_symlinks=False)
    observation_key = "a" * 64
    sample_seed = cvoi_sample_seed(239, observation_key)
    features = {
        "video_clip": torch.ones(3, 2, 4, 4),
        "cvoi_observation_key": observation_key_tensor(observation_key),
        "cvoi_rng_seed_bytes": unsigned_seed_tensor(sample_seed),
    }

    class FeatureBuilder:
        def compute_features(self, agent_input: object) -> dict[str, torch.Tensor]:
            del agent_input
            return {name: value.clone() for name, value in features.items()}

    class Trajectory:
        def __init__(self, poses: np.ndarray, trajectory_sampling: object) -> None:
            self.poses = poses
            self.trajectory_sampling = trajectory_sampling

    monkeypatch.setitem(globals_dict, "Trajectory", Trajectory)
    agent = object.__new__(agent_class)
    agent._cvoi_manual_runtime = runtime_type(
        policy_id=policy_id,
        lineage="p1_full",
        planner_stage="p1",
        scenario_manifest=SimpleNamespace(
            token_for_observation_key=lambda key: (
                "scenario-a" if key == observation_key else (_ for _ in ()).throw(KeyError(key))
            )
        ),
        forced_horizon=horizon,
        guidance_steps=guidance_steps,
        common_random_seed=239,
        trace_output_dir=trace_dir,
        trace_output_dir_identity=(trace_stat.st_dev, trace_stat.st_ino),
        **_manual_artifact_paths(),
    )
    agent._feature_builder = FeatureBuilder()
    agent._cvoi_trajectory_sampling = SimpleNamespace(num_poses=8)
    agent._cvoi_gate = None
    agent._cvoi_evaluation_gate_feature_mode = None
    agent._last_cvoi_navtrain_gate_features = None
    agent.eval = lambda: agent
    agent.get_feature_builders = lambda: [agent._feature_builder]
    agent.prepare_cvoi_features = lambda values: values
    agent.set_cvoi_evaluation_guidance_steps = lambda value: setattr(agent, "_cvoi_evaluation_guidance_steps", value)
    agent.set_cvoi_evaluation_forced_horizon = lambda value: setattr(agent, "_cvoi_evaluation_forced_horizon", value)
    agent.set_cvoi_latency_mode = lambda value: setattr(agent, "_cvoi_latency_mode", value)

    def forward(model_features: dict[str, object]) -> dict[str, torch.Tensor]:
        assert model_features["cvoi_rng_seed"] == sample_seed
        agent._last_cvoi_trace = {
            "stop_horizon": horizon,
            "decisions": [],
            "predicted_deltas": [],
            "rollout_latency_ms": 0.0,
            "guidance": {"guidance_steps": float(guidance_steps)},
        }
        agent._last_cvoi_navtrain_gate_features = {
            "gate_features": [1.0, 2.0, float(horizon)],
            "observed_feature_sha256": "b" * 64,
            "horizon": horizon,
        }
        return {"trajectory": torch.zeros(1, 8, 3)}

    agent.forward = forward

    trajectory = agent.compute_trajectory(object(), object(), object(), 0.5)

    assert trajectory.poses.shape == (8, 3)
    trace = json.loads((trace_dir / f"{observation_key}.json").read_text(encoding="ascii"))
    assert set(trace) == {
        "schema",
        "protocol_id",
        "scenario_token",
        "observation_key",
        "policy_id",
        "lineage",
        "horizon",
        "gate_features",
        "observed_feature_sha256",
    }
    assert trace["schema"] == "cvoi_manual_navtrain_gate_policy_trace_v1"
    assert trace["protocol_id"] == "epdms_v2_one_stage_navtrain_gate_label_v1"
    assert trace["policy_id"] == policy_id
    assert trace["horizon"] == horizon
    assert not any("sha256" in key for key in trace if key != "observed_feature_sha256")


@pytest.mark.parametrize(
    ("device", "cuda_available", "error"),
    [
        ("cpu", True, "requires a CUDA device"),
        ("cuda", False, "requires available CUDA"),
    ],
)
def test_manual_navtrain_initialize_fails_before_reading_the_scorer_without_requested_cuda(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    cuda_available: bool,
    error: str,
) -> None:
    agent_class = _load_navsim_agent()
    globals_dict = agent_class.initialize.__globals__
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    def forbidden_reader(path: Path) -> object:
        del path
        raise AssertionError("manual scorer config must not be read before CUDA validation")

    monkeypatch.setitem(globals_dict, "read_manual_navtrain_scorer_config", forbidden_reader)
    agent = agent_class(
        training_config_path="/must/not/be/read.yaml",
        device=device,
        cvoi_manual_navtrain_gate_config_path="/must/not/be/read.json",
    )

    with pytest.raises((ValueError, RuntimeError), match=error):
        agent.initialize()


def test_manual_navtrain_initialize_rejects_an_out_of_range_cuda_index_before_reading_the_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_class = _load_navsim_agent()
    globals_dict = agent_class.initialize.__globals__
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def forbidden_reader(path: Path) -> object:
        del path
        raise AssertionError("manual scorer config must not be read before CUDA validation")

    monkeypatch.setitem(globals_dict, "read_manual_navtrain_scorer_config", forbidden_reader)
    agent = agent_class(
        training_config_path="/must/not/be/read.yaml",
        device="cuda:1",
        cvoi_manual_navtrain_gate_config_path="/must/not/be/read.json",
    )

    with pytest.raises(ValueError, match=r"CUDA device index 1.*device_count=1"):
        agent.initialize()


@pytest.mark.parametrize(
    ("lineage", "mutation"),
    [
        ("p1_full", None),
        ("p1_no_cf", None),
        ("p1_no_cf", "full_p1"),
        ("p1_no_cf", "full_calibration"),
        ("p1_no_cf", "full_signature"),
        ("p1_no_cf", "wrong_signature_mechanism"),
        ("p1_no_cf", "full_source"),
    ],
)
def test_manual_navtrain_initialize_binds_exact_lineage_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lineage: str,
    mutation: str | None,
) -> None:
    agent_class = _load_navsim_agent()
    globals_dict = agent_class.initialize.__globals__
    scorer_type = globals_dict["ManualNavTrainScorerConfig"]
    full_root = (tmp_path / "cvoi_manual_full").resolve()
    ablation_root = (tmp_path / "cvoi_manual_ablation").resolve()
    lineage_name = lineage.removeprefix("p1_")
    results_root = full_root if lineage == "p1_full" else ablation_root / lineage_name
    horizon_dir = results_root / "oracle/work/h2"
    full_handoff_dir = full_root / "handoff"
    handoff_dir = results_root / "handoff"
    p0_dir = full_root / "p0"
    p1_dir = results_root / "p1"
    for directory in (horizon_dir, full_handoff_dir, handoff_dir, p0_dir, p1_dir, full_root / "p1"):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_FULL_RESULTS_ROOT", full_root)
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_ABLATION_RESULTS_ROOT", ablation_root)
    for name, value in (
        ("CVOI_MANUAL_FULL_RESULTS_ROOT", full_root),
        ("CVOI_MANUAL_ABLATION_RESULTS_ROOT", ablation_root),
    ):
        if name in globals_dict:
            monkeypatch.setitem(globals_dict, name, value)
    output_dir = horizon_dir / "scorer_output"
    trace_dir = horizon_dir / "policy_traces"
    output_dir.mkdir()
    trace_dir.mkdir()
    p0_target = p0_dir / "candidate.pt"
    p1_target = p1_dir / "candidate.pt"
    p0_target.write_bytes(b"p0")
    p1_target.write_bytes(b"p1")
    p0_selected = full_handoff_dir / "p0_selected.pt"
    p1_selected = handoff_dir / "p1_selected.pt"
    p0_selected.symlink_to(p0_target)
    p1_selected.symlink_to(p1_target)
    full_p1_selected = p1_selected
    if lineage == "p1_no_cf":
        full_p1_target = full_root / "p1/candidate.pt"
        full_p1_target.write_bytes(b"full-p1")
        full_p1_selected = full_handoff_dir / "p1_selected.pt"
        full_p1_selected.symlink_to(full_p1_target)
    artifacts = {
        "p0_planner_checkpoint": p0_selected,
        "field_checkpoint": handoff_dir / "field.pt",
        "calibration_checkpoint": handoff_dir / "calibration.pt",
        "p1_planner_checkpoint": p1_selected,
        "stop_checkpoint": handoff_dir / "stop.pt",
    }
    for role in ("field_checkpoint", "calibration_checkpoint", "stop_checkpoint"):
        artifacts[role].write_bytes(role.encode("ascii"))
    full_calibration = artifacts["calibration_checkpoint"]
    if lineage == "p1_no_cf":
        full_calibration = full_handoff_dir / "calibration.pt"
        full_calibration.write_bytes(b"full-calibration")
    if mutation == "full_p1":
        artifacts["p1_planner_checkpoint"] = full_p1_selected
    elif mutation == "full_calibration":
        artifacts["calibration_checkpoint"] = full_calibration
    signature_branch = "p1_full" if mutation == "full_signature" else lineage
    signature = {
        "experiment_role": "main" if lineage == "p1_full" else "ablation",
        "cf_field_supervision": "hazard_quality" if lineage == "p1_full" else "none",
        "field_calibration_mode": "local_geometry",
        "p0_prefix_mode": "uniform",
        "gate_feature_mode": "full",
        "branch_id": signature_branch,
    }
    if mutation == "wrong_signature_mechanism":
        signature["experiment_role"] = "main"
        signature["cf_field_supervision"] = "hazard_quality"
    effective_config = horizon_dir / "effective.yaml"
    effective_config.write_text(
        json.dumps(
            {
                "cvoi": {
                    "stage": "evaluation",
                    "evaluation_mode": "p1_field_forced",
                    "controller_lineage": "value_guided",
                    "unguided_planner_checkpoint": str(p0_selected),
                    "field_checkpoint": str(artifacts["calibration_checkpoint"]),
                    "guided_planner_checkpoint": str(p1_selected),
                    "dual_value_checkpoint": str(artifacts["stop_checkpoint"]),
                    "output_checkpoint": None,
                    "ablation_signature": signature,
                }
            }
        ),
        encoding="ascii",
    )
    if mutation == "full_p1":
        effective_payload = json.loads(effective_config.read_text(encoding="ascii"))
        effective_payload["cvoi"]["guided_planner_checkpoint"] = str(artifacts["p1_planner_checkpoint"])
        effective_config.write_text(json.dumps(effective_payload), encoding="ascii")
    scorer_path = horizon_dir / "scorer_config.json"
    scorer_path.write_text("{}", encoding="ascii")
    source_config_path = (tmp_path / f"{lineage_name}_05_p1.yaml").resolve()
    source_config_path.write_text("stage: train\n", encoding="ascii")
    value_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
        phase="guided_planner",
        branch_id=lineage,
        full_results_root=full_root,
        ablation_results_root=ablation_root,
    )
    source = ManualOracleSource(
        source_config_path=source_config_path,
        results_root=results_root,
        value_lineage=value_lineage,
        lineage=lineage,
        p0_planner_checkpoint=p0_selected,
        field_checkpoint=handoff_dir / "field.pt",
        calibration_checkpoint=handoff_dir / "calibration.pt",
        p1_planner_checkpoint=p1_selected,
        stop_checkpoint=handoff_dir / "stop.pt",
        oracle_path=handoff_dir / "oracle_full.sqlite3",
        lambda_grid=FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
    )
    if mutation == "full_source":
        full_source_config_path = (tmp_path / "full_05_p1.yaml").resolve()
        full_source_config_path.write_text("stage: train\n", encoding="ascii")
        full_value_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
            phase="guided_planner",
            branch_id="p1_full",
            full_results_root=full_root,
            ablation_results_root=ablation_root,
        )
        for name in ("field.pt", "stop.pt"):
            (full_handoff_dir / name).write_bytes(name.encode("ascii"))
        source = ManualOracleSource(
            source_config_path=full_source_config_path,
            results_root=full_root,
            value_lineage=full_value_lineage,
            lineage="p1_full",
            p0_planner_checkpoint=p0_selected,
            field_checkpoint=full_handoff_dir / "field.pt",
            calibration_checkpoint=full_handoff_dir / "calibration.pt",
            p1_planner_checkpoint=full_p1_selected,
            stop_checkpoint=full_handoff_dir / "stop.pt",
            oracle_path=full_handoff_dir / "oracle_full.sqlite3",
            lambda_grid=FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
        )
    scorer = object.__new__(scorer_type)
    scorer_values = {
        "schema": "cvoi_manual_navtrain_gate_scorer_v1",
        "protocol_id": "epdms_v2_one_stage_navtrain_gate_label_v1",
        "policy_id": f"{lineage}__fixed_h2_k2",
        "lineage": lineage,
        "planner_stage": "p1",
        "policy_mode": "fixed_horizon",
        "forced_horizon": 2,
        "guidance_steps": 2,
        "common_random_seed": 239,
        "artifacts": artifacts,
        "environment": None,
        "authority": SimpleNamespace(scenario_manifest=object()),
        "source": source,
        "effective_config_path": effective_config,
        "output_dir": output_dir,
        "trace_output_dir": trace_dir,
        "score_store_path": horizon_dir / "score_store.sqlite3",
        "feature_store_path": horizon_dir / "feature_store.sqlite3",
    }
    for field, value in scorer_values.items():
        object.__setattr__(scorer, field, value)

    poisoned_full_root = (tmp_path / "poisoned-default-full").resolve()
    poisoned_ablation_root = (tmp_path / "poisoned-default-ablation").resolve()
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_FULL_RESULTS_ROOT", poisoned_full_root)
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_ABLATION_RESULTS_ROOT", poisoned_ablation_root)
    for name, value in (
        ("CVOI_MANUAL_FULL_RESULTS_ROOT", poisoned_full_root),
        ("CVOI_MANUAL_ABLATION_RESULTS_ROOT", poisoned_ablation_root),
    ):
        if name in globals_dict:
            monkeypatch.setitem(globals_dict, name, value)

    events: list[str] = []
    exact_validator_inputs: list[Mapping[str, object]] = []

    def typed_reader(path: Path) -> object:
        assert path == scorer_path
        events.append("typed_scorer")
        return scorer

    def planner_reader(path: Path) -> dict[str, object]:
        events.append(f"direct_{path.parent.name}")
        return _planner_payload("p0", "p0_uniform") if path == p0_target else _planner_payload("p1", lineage)

    def value_reader(
        path: Path,
        *,
        required_phase: str,
        required_branch_id: str,
        map_location: object,
    ) -> dict[str, object]:
        del path
        assert map_location == "cpu"
        events.append(required_phase)
        return {
            "phase": required_phase,
            "branch_id": required_branch_id,
            "architecture": {"embed_dim": 1, "hidden_dim": 2},
            "state_dict": {"weight": torch.ones(1)},
        }

    def track_exact_direct_validator(checkpoint: Mapping[str, object]) -> Mapping[str, object]:
        exact_validator_inputs.append(checkpoint)
        return checkpoint

    removed_e120_apis = {
        "load_cvoi_formal_v2_navsim_e120_official_scorer_config",
        "load_cvoi_formal_v2_navsim_e120_selection_scorer_config",
        "load_cvoi_formal_v2_navsim_e120_navtrain_gate_scorer_config",
        "resolve_cvoi_formal_v2_navsim_e120_effective_policy",
        "resolve_cvoi_formal_v2_navsim_e120_selection_effective_policy",
        "resolve_cvoi_formal_v2_navsim_e120_navtrain_gate_effective_policy",
        "open_cvoi_formal_v2_navsim_e120_artifact_handles",
        "validate_formal_v2_navsim_e120_checkpoint",
        "validate_formal_v2_navsim_e120_lineage",
        "validate_formal_v2_navsim_e120_runtime_signature",
    }
    assert removed_e120_apis.isdisjoint(globals_dict)
    assert {
        "resolve_cvoi_official_effective_policy",
        "open_cvoi_navsim_artifact_handles",
    }.isdisjoint(globals_dict)
    monkeypatch.setitem(globals_dict, "read_manual_navtrain_scorer_config", typed_reader)
    monkeypatch.setitem(globals_dict, "read_formal_v2_navsim_e120_direct_checkpoint", planner_reader)
    monkeypatch.setitem(globals_dict, "read_cvoi_navsim_e120_direct_value_checkpoint", value_reader)
    monkeypatch.setitem(
        globals_dict,
        "validate_formal_v2_navsim_e120_direct_checkpoint",
        track_exact_direct_validator,
    )

    config = SimpleNamespace(
        cvoi=SimpleNamespace(
            enabled=True,
            stage="evaluation",
            evaluation_mode="p1_field_forced",
            controller_lineage="value_guided",
            unguided_planner_checkpoint=str(p0_selected),
            field_checkpoint=str(artifacts["calibration_checkpoint"]),
            guided_planner_checkpoint=str(p1_selected),
            dual_value_checkpoint=str(artifacts["stop_checkpoint"]),
            output_checkpoint=None,
            ablation_signature=SimpleNamespace(**signature),
        ),
        multiview=SimpleNamespace(enabled=False),
    )
    monkeypatch.setitem(globals_dict, "parse_training_config", lambda raw: config)
    monkeypatch.setitem(globals_dict, "require_cvoi_stage_for_entry", lambda *args, **kwargs: None)
    monkeypatch.setitem(globals_dict, "validate_cvoi_sequential_runtime_config", lambda config: None)
    monkeypatch.setitem(globals_dict, "cvoi_enabled", lambda config: True)
    monkeypatch.setitem(globals_dict, "value_planning_enabled", lambda config: False)

    def init_encoder(config: object, device: torch.device) -> tuple[torch.nn.Module, None]:
        del config, device
        if mutation is not None:
            raise AssertionError("manual scorer drift must fail before model construction")
        assert "stop_calibrated" in events
        events.append("init_encoder")
        return torch.nn.Linear(1, 1), None

    monkeypatch.setitem(globals_dict, "init_encoder", init_encoder)
    monkeypatch.setitem(globals_dict, "get_encoder_embed_dim", lambda encoder: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    agent = agent_class(
        checkpoint_path=str(artifacts["p1_planner_checkpoint"]),
        training_config_path=str(effective_config),
        device="cuda",
        cvoi_manual_navtrain_gate_config_path=str(scorer_path),
    )
    agent._parse_inference_params = lambda: None

    def initialize_stage12(device: torch.device, encoder: torch.nn.Module, embed_dim: int) -> None:
        del device, encoder, embed_dim
        events.append("initialize_stage12")
        agent._cvoi_dual_value = torch.nn.Identity()
        agent._cvoi_dual_value_adapter = torch.nn.Identity()
        agent._cvoi_navtrain_stop_value = torch.nn.Identity()
        agent._cvoi_navtrain_stop_value_adapter = torch.nn.Identity()
        agent._cvoi_gate = None
        agent._validate_configured_cvoi_planner_checkpoint(agent._cvoi_manual_artifacts.p1_checkpoint)

    agent._initialize_stage12 = initialize_stage12
    agent._build_feature_builder = lambda: object()

    if mutation is not None:
        expected_error = {
            "full_p1": "p1_planner_checkpoint|P1",
            "full_calibration": "calibration_checkpoint|Calibration",
            "full_signature": "p1_no_cf|signature|branch",
            "wrong_signature_mechanism": "p1_no_cf|signature|mechanism|experiment_role|cf_field_supervision",
            "full_source": "p1_no_cf|source|lineage",
        }[mutation]
        with pytest.raises(ValueError, match=expected_error):
            agent.initialize()
        assert "init_encoder" not in events
        return

    agent.initialize()

    assert events.index("typed_scorer") < events.index("direct_p0")
    assert events.index("stop_calibrated") < events.index("init_encoder")
    assert agent._cvoi_manual_runtime.lineage == lineage
    assert agent._cvoi_manual_runtime.p0_planner_checkpoint == p0_selected
    assert agent._cvoi_manual_runtime.p1_planner_checkpoint == p1_selected
    assert agent._cvoi_manual_runtime.scenario_manifest is scorer.authority.scenario_manifest
    assert len(exact_validator_inputs) == 1
    assert exact_validator_inputs[0] is agent._cvoi_manual_artifacts.p1_checkpoint
    correct_runtime_payload = _planner_payload("p1", lineage)
    validated = agent._validate_configured_cvoi_planner_checkpoint(correct_runtime_payload)
    assert validated["lineage"]["branch_id"] == lineage
    assert len(exact_validator_inputs) == 2
    assert exact_validator_inputs[-1] is correct_runtime_payload

    wrong_lineage = "p1_no_cf" if lineage == "p1_full" else "p1_full"
    wrong_runtime_payload = _planner_payload("p1", wrong_lineage)
    exact_calls_before_wrong_lineage = len(exact_validator_inputs)
    with pytest.raises(ValueError, match=lineage):
        agent._validate_configured_cvoi_planner_checkpoint(wrong_runtime_payload)
    assert len(exact_validator_inputs) == exact_calls_before_wrong_lineage + 1
    assert exact_validator_inputs[-1] is wrong_runtime_payload
