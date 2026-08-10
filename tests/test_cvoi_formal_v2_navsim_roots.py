"""Self-contained root-contract tests for the CVoI Formal-v2 NavSim study."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from app.vjepa_cowa_world_model.training import cvoi_formal_v2_navsim_roots as roots_module
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import (
    build_formal_v2_navsim_direct_task_projection,
    build_formal_v2_navsim_root_catalog,
    validate_formal_v2_navsim_direct_task_projection,
)
from app.vjepa_cowa_world_model.training.navsim_scene_filter_contract import load_navsim_scene_filter_contract


def _domains(projection: dict[str, object], field: str) -> list[str]:
    return [root["domain"] for root in projection[field]]


def _names(projection: dict[str, object], field: str) -> list[str]:
    return [root["name"] for root in projection[field]]


def _projection(stage: str, branch: str, preflight_root: str = "/results/cvoi/preflight") -> dict[str, object]:
    return build_formal_v2_navsim_direct_task_projection(
        stage,
        branch,
        build_formal_v2_navsim_root_catalog(),
        preflight_root,
    )


def test_root_catalog_is_embedded_and_contains_no_reference_or_digest_projection() -> None:
    catalog = build_formal_v2_navsim_root_catalog()

    assert set(catalog) == {"schema", "roots"}
    assert tuple(catalog["roots"]) == ("real_train", "cf_train", "real_navtest", "cf_val")
    assert [root["name"] for root in catalog["roots"].values()] == [
        "navsim_real_train",
        "navsim_cf_train",
        "navsim_real_navtest",
        "navsim_cf_val",
    ]
    source = roots_module.__file__
    assert source is not None
    for removed_name in (
        "FORMAL_V2_NAVSIM_REFERENCE_CONFIG",
        "FORMAL_V2_NAVSIM_REFERENCE_CONFIG_SHA256",
        "FORMAL_V2_NAVSIM_REFERENCE_DATA_SHA256",
        "FORMAL_V2_NAVSIM_REFERENCE_ROOTS_SHA256",
    ):
        assert not hasattr(roots_module, removed_name)


def test_root_catalog_uses_neutral_portable_defaults() -> None:
    catalog = build_formal_v2_navsim_root_catalog()
    roots = catalog["roots"]

    assert roots_module.FORMAL_V2_NAVSIM_DIRECT_PREFLIGHT_ROOT == Path(
        "/path/to/rise/results/cvoi_manual_full/preflight"
    )
    assert roots["real_train"]["data_path"] == "/path/to/navsim/dataset/navsim_logs/trainval"
    assert roots["real_train"]["sensor_blobs_path"] == "/path/to/navsim/dataset/sensor_blobs/trainval"
    assert roots["real_train"]["scene_filter_yaml"] == "configs/navsim/scene_filters/navtrain.yaml"
    assert roots["real_navtest"]["data_path"] == "/path/to/navsim/dataset/navsim_logs/test"
    assert roots["real_navtest"]["sensor_blobs_path"] == "/path/to/navsim/dataset/sensor_blobs/test"
    assert roots["real_navtest"]["scene_filter_yaml"] == "configs/navsim/scene_filters/navtest.yaml"
    assert roots["cf_train"]["data_path"] == "/path/to/counterfactual/navsim_logs/trainval"
    assert roots["cf_train"]["sensor_blobs_path"] == "/path/to/counterfactual/sensor_blobs/trainval"
    assert roots["cf_train"]["pose_overlay_path"] == ("/path/to/counterfactual/pose_overlay/trainval/pred_pose")
    assert roots["cf_train"]["annotations_path"] == ("/path/to/counterfactual/annotations/navsim_train.json")
    assert roots["cf_val"]["data_path"] == "/path/to/counterfactual/navsim_logs/test"
    assert roots["cf_val"]["sensor_blobs_path"] == "/path/to/counterfactual/sensor_blobs/test"
    assert roots["cf_val"]["pose_overlay_path"] == "/path/to/counterfactual/pose_overlay/test/pred_pose"
    assert roots["cf_val"]["annotations_path"] == "/path/to/counterfactual/annotations/navsim_test.json"


@pytest.mark.parametrize(
    "scene_filter_yaml",
    (
        "configs/navsim/scene_filters/navtrain.yaml",
        "configs/navsim/scene_filters/navtest.yaml",
    ),
)
def test_formal_scene_filter_resolver_is_repo_rooted_and_cwd_independent(
    scene_filter_yaml: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(roots_module.__file__).resolve().parents[3]
    monkeypatch.chdir(tmp_path)

    resolved = roots_module.resolve_formal_v2_navsim_scene_filter_path(scene_filter_yaml)

    assert resolved == repository_root / scene_filter_yaml
    assert resolved.is_file()
    assert load_navsim_scene_filter_contract(scene_filter_yaml).path == resolved.resolve()
    with pytest.raises(ValueError, match="exact repository-relative path"):
        roots_module.resolve_formal_v2_navsim_scene_filter_path("scene_filter/navtrain.yaml")


def test_catalog_returns_fresh_copies_and_rejects_field_level_root_drift() -> None:
    first = build_formal_v2_navsim_root_catalog()
    second = build_formal_v2_navsim_root_catalog()
    first["roots"]["real_train"]["data_path"] = "/mutated"

    assert second["roots"]["real_train"]["data_path"].endswith("/navsim_logs/trainval")
    drifted = copy.deepcopy(second)
    drifted["roots"]["real_train"]["sensor_blobs_path"] = "/wrong/blobs"
    with pytest.raises(ValueError, match=r"real_train\.sensor_blobs_path"):
        build_formal_v2_navsim_direct_task_projection("p0", "uniform", drifted, "/results/preflight")


def test_catalog_accepts_a_second_canonical_absolute_prefix() -> None:
    catalog = copy.deepcopy(build_formal_v2_navsim_root_catalog())
    catalog["roots"]["real_train"]["data_path"] = "/opt/rise-user/navsim/dataset/navsim_logs/trainval"
    catalog["roots"]["real_train"]["sensor_blobs_path"] = "/opt/rise-user/navsim/dataset/sensor_blobs/trainval"

    projection = build_formal_v2_navsim_direct_task_projection(
        "p0",
        "uniform",
        catalog,
        "/results/preflight",
    )

    assert projection["train_roots"][0]["data_path"] == catalog["roots"]["real_train"]["data_path"]
    assert projection["train_roots"][0]["sensor_blobs_path"] == catalog["roots"]["real_train"]["sensor_blobs_path"]


def test_catalog_relocation_preserves_all_external_paths_through_projection_validation() -> None:
    catalog = copy.deepcopy(build_formal_v2_navsim_root_catalog())
    relocated_paths = {
        "real_train": {
            "data_path": "/opt/rise-user/real/navsim_logs/trainval",
            "sensor_blobs_path": "/opt/rise-user/real/sensor_blobs/trainval",
        },
        "cf_train": {
            "data_path": "/opt/rise-user/counterfactual/navsim_logs/trainval",
            "sensor_blobs_path": "/opt/rise-user/counterfactual/sensor_blobs/trainval",
            "pose_overlay_path": "/opt/rise-user/counterfactual/pose_overlay/trainval/pred_pose",
            "annotations_path": "/opt/rise-user/counterfactual/annotations/navsim_train.json",
        },
        "real_navtest": {
            "data_path": "/opt/rise-user/real/navsim_logs/test",
            "sensor_blobs_path": "/opt/rise-user/real/sensor_blobs/test",
        },
        "cf_val": {
            "data_path": "/opt/rise-user/counterfactual/navsim_logs/test",
            "sensor_blobs_path": "/opt/rise-user/counterfactual/sensor_blobs/test",
            "pose_overlay_path": "/opt/rise-user/counterfactual/pose_overlay/test/pred_pose",
            "annotations_path": "/opt/rise-user/counterfactual/annotations/navsim_test.json",
        },
    }
    for role, paths in relocated_paths.items():
        catalog["roots"][role].update(paths)

    projection = build_formal_v2_navsim_direct_task_projection(
        "field",
        "full",
        catalog,
        "/opt/rise-user/results/preflight",
    )

    assert (
        validate_formal_v2_navsim_direct_task_projection(
            "field",
            "full",
            projection,
            catalog,
            "/opt/rise-user/results/preflight",
        )
        == projection
    )
    projected_by_name = {root["name"]: root for field in ("train_roots", "val_roots") for root in projection[field]}
    for role, paths in relocated_paths.items():
        projected = projected_by_name[catalog["roots"][role]["name"]]
        assert {field: projected[field] for field in paths} == paths


@pytest.mark.parametrize(
    ("role", "field", "value"),
    (
        ("real_train", "data_path", "relative/navsim_logs/trainval"),
        ("real_train", "sensor_blobs_path", "/opt/data/../sensor_blobs/trainval"),
        ("real_train", "data_path", "/opt/rise/./navsim_logs/trainval"),
        ("real_train", "data_path", "/opt/rise/navsim_logs/trainval/"),
        ("real_train", "data_path", "/opt/data/navsim_logs/test"),
        ("real_navtest", "sensor_blobs_path", "/opt/data/sensor_blobs/trainval"),
        ("cf_train", "pose_overlay_path", "/opt/data/pose_overlay/trainval/not_pred_pose"),
        ("cf_train", "pose_overlay_path", "/opt/data/pose_overlay/test/pred_pose"),
        ("cf_val", "pose_overlay_path", "/opt/data/pose_overlay/trainval/pred_pose"),
        ("cf_train", "annotations_path", "/opt/data/annotations/navsim_test.json"),
        ("cf_val", "annotations_path", "/opt/data/annotations/navsim_train.json"),
        ("cf_val", "data_path", ""),
    ),
)
def test_catalog_rejects_noncanonical_external_path_structure(role: str, field: str, value: str) -> None:
    catalog = copy.deepcopy(build_formal_v2_navsim_root_catalog())
    catalog["roots"][role][field] = value

    with pytest.raises(ValueError, match=rf"{role}\.{field}"):
        build_formal_v2_navsim_direct_task_projection(
            "field",
            "full",
            catalog,
            "/results/preflight",
        )


def test_direct_projection_preserves_exact_root_names_paths_and_runtime_fields() -> None:
    projection = _projection("field", "full")
    real_train, cf_train = projection["train_roots"]
    real_val, cf_val = projection["val_roots"]

    assert [real_train["name"], cf_train["name"], real_val["name"], cf_val["name"]] == [
        "navsim_real_train",
        "navsim_cf_train",
        "navsim_real_navtest",
        "navsim_cf_val",
    ]
    assert real_train["data_path"].endswith("/navsim_logs/trainval")
    assert real_train["scene_filter_yaml"] == "configs/navsim/scene_filters/navtrain.yaml"
    assert real_val["data_path"].endswith("/navsim_logs/test")
    assert real_val["scene_filter_yaml"] == "configs/navsim/scene_filters/navtest.yaml"
    for root in (real_train, cf_train, real_val, cf_val):
        assert root["window_stride"] == 4
        assert root["camera_name"] == "CAM_F0"
        assert root["camera_names"] == ["CAM_F0"]
        assert root["fps"] == 2
        assert root["num_target_frames"] == 12
        assert root["num_observed_frames"] == 4
        assert root["max_agents"] == 1024
        assert root["image_require_policy"] == "observed_only"
        assert "source_root_sha256" not in root
        assert "effective_runtime_root_sha256" not in root
        assert "runtime_overlay_schema" not in root
    assert cf_train["trajectory_quality_path"] == ("/results/cvoi/preflight/trajectory_quality/navsim_cf_train.json")
    assert cf_val["trajectory_quality_path"] == "/results/cvoi/preflight/trajectory_quality/navsim_cf_val.json"
    assert cf_train["annotations_accident_type_allowlist"] == ["自车行为引起", "非自车行为引起"]


@pytest.mark.parametrize("branch", ["uniform", "extremes", "short_heavy", "no_full"])
def test_p0_projection_is_real_only(branch: str) -> None:
    projection = _projection("p0", branch)

    assert _domains(projection, "train_roots") == ["real"]
    assert _names(projection, "val_roots") == ["navsim_real_navtest"]
    assert projection["balance_train_roots"] is False


@pytest.mark.parametrize("branch", ["full", "hazard_only", "quality_only"])
def test_counterfactual_field_projection_is_balanced(branch: str) -> None:
    projection = _projection("field", branch)

    assert _domains(projection, "train_roots") == ["real", "counterfactual"]
    assert _domains(projection, "val_roots") == ["real", "counterfactual"]
    assert projection["balance_train_roots"] is True


@pytest.mark.parametrize(
    ("stage", "branch"),
    [
        ("field", "strict_real_only"),
        ("calibration", "full"),
        ("p1", "full"),
        ("stop", "full"),
        ("oracle", "full"),
    ],
)
def test_real_only_stages_keep_real_train_and_navtest(stage: str, branch: str) -> None:
    projection = _projection(stage, branch)

    assert _domains(projection, "train_roots") == ["real"]
    assert _domains(projection, "val_roots") == ["real"]
    assert projection["balance_train_roots"] is False


def test_gate_projection_is_artifact_only() -> None:
    assert _projection("gate", "full") == {
        "train_roots": [],
        "val_roots": [],
        "balance_train_roots": False,
    }


def test_direct_validation_rejects_semantic_balance_and_proof_field_drift() -> None:
    catalog = build_formal_v2_navsim_root_catalog()
    expected = _projection("field", "full")

    semantic_drift = copy.deepcopy(expected)
    semantic_drift["train_roots"][0]["window_stride"] = 1
    with pytest.raises(ValueError, match="window_stride"):
        validate_formal_v2_navsim_direct_task_projection(
            "field",
            "full",
            semantic_drift,
            catalog,
            "/results/cvoi/preflight",
        )

    balance_drift = copy.deepcopy(expected)
    balance_drift["balance_train_roots"] = False
    with pytest.raises(ValueError, match="balance_train_roots"):
        validate_formal_v2_navsim_direct_task_projection(
            "field",
            "full",
            balance_drift,
            catalog,
            "/results/cvoi/preflight",
        )

    proof_drift = copy.deepcopy(expected)
    proof_drift["train_roots"][0]["source_root_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unexpected=.*source_root_sha256"):
        validate_formal_v2_navsim_direct_task_projection(
            "field",
            "full",
            proof_drift,
            catalog,
            "/results/cvoi/preflight",
        )


def test_projection_rejects_unknown_stage_and_relative_preflight_root() -> None:
    catalog = build_formal_v2_navsim_root_catalog()
    with pytest.raises(ValueError, match="unsupported Formal-v2 NavSim stage/branch"):
        build_formal_v2_navsim_direct_task_projection("unknown", "full", catalog, "/results/preflight")
    with pytest.raises(ValueError, match="preflight_root must be an absolute path"):
        build_formal_v2_navsim_direct_task_projection("field", "full", catalog, "relative/preflight")
    with pytest.raises(ValueError, match="preflight_root must not contain.*traversal"):
        build_formal_v2_navsim_direct_task_projection("field", "full", catalog, "/results/../preflight")


def test_locked_counterfactual_annotation_source_validates_bytes_and_record_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation_path = tmp_path / "navsim_train.json"
    payload = [{"scene": "scene-a", "annos": {"trajectory_match": True}}]
    annotation_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    annotation_path.write_bytes(annotation_bytes)
    expected_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    monkeypatch.setattr(
        roots_module,
        "FORMAL_V2_NAVSIM_CF_ANNOTATION_CONTRACTS",
        {
            "cf_train": {
                "source_sha256": expected_sha256,
                "source_record_count": 1,
                "selected_record_count": 1,
            }
        },
    )

    summary = roots_module.validate_formal_v2_navsim_cf_annotation_source(
        annotation_path,
        root_name="navsim_cf_train",
    )

    assert summary == {
        "role": "cf_train",
        "root_name": "navsim_cf_train",
        "source_sha256": expected_sha256,
        "source_record_count": 1,
        "selected_record_count": 1,
    }

    annotation_path.write_bytes(annotation_bytes + b"\n")
    with pytest.raises(ValueError, match="source_sha256 mismatch"):
        roots_module.validate_formal_v2_navsim_cf_annotation_source(
            annotation_path,
            root_name="navsim_cf_train",
        )


def test_locked_counterfactual_annotation_source_rejects_record_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation_path = tmp_path / "navsim_test.json"
    annotation_bytes = b"[]"
    annotation_path.write_bytes(annotation_bytes)
    monkeypatch.setattr(
        roots_module,
        "FORMAL_V2_NAVSIM_CF_ANNOTATION_CONTRACTS",
        {
            "cf_val": {
                "source_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                "source_record_count": 1,
                "selected_record_count": 0,
            }
        },
    )

    with pytest.raises(ValueError, match="source_record_count mismatch"):
        roots_module.validate_formal_v2_navsim_cf_annotation_source(
            annotation_path,
            root_name="navsim_cf_val",
        )
