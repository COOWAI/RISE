"""Repository-surface contract for the minimal NavSim CVoI Full branch."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 preparation environment
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG_ROOT = REPO_ROOT / "configs/train/navsim/cvoi_manual_full"
AGENT_ROOT = REPO_ROOT / "configs/navsim/cvoi_manual/agent"
EPDMS_ROOT = REPO_ROOT / "configs/eval/navsim/cvoi_manual_epdms"

FULL_CONFIG_NAMES = {
    "01_predictor_lewm_pure.yaml",
    "02_p0_uniform.yaml",
    "03_field_full.yaml",
    "04_calibration_full.yaml",
    "05_p1_full.yaml",
    "06_stop_full.yaml",
    "07_gate_full.yaml",
}
RETAINED_APP_ROOT_FILES = {
    "__init__.py",
    "train_cvoi_offline.py",
    "train_latent_predictor.py",
    "train_predictor_rollout_planner.py",
    "val_command.py",
}
RETAINED_TRAIN_SCRIPTS = {
    "train_cvoi_offline",
    "train_latent_predictor",
    "train_predictor_rollout_planner",
}
EXPERIMENT_TOOLS = {
    "build_cvoi_navsim_scenario_manifest.py",
    "generate_navsim_cf_trajectory_quality.py",
    "run_cvoi_direct_epdms.py",
    "run_cvoi_manual_oracle.py",
}
RELEASE_TOOLS = {
    "check_package.py",
    "check_public_surface.py",
    "export_public_tree.py",
}
RETAINED_SCRIPT = "scripts/eval_navsim/eval_navsim_v2_pdms.sh"
FORBIDDEN_ROOTS = (
    "assets",
    "ddddetection_torchcv",
    "_".join(("Drive", "JEPA")),
)
FORBIDDEN_APP_PUBLIC_PATHS = (
    "app/vjepa_cowa_world_model/EXTENDING.md",
    "app/vjepa_cowa_world_model/README.md",
    "app/vjepa_cowa_world_model/script",
    "app/vjepa_cowa_world_model/tests",
)
DOCUMENTED_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_./-])((?:app|configs|docs|scripts|src|tests|tools)/[A-Za-z0-9_./-]+)"
)
CORE_COLD_IMPORT_MODULES = (
    "app.vjepa_cowa_world_model.train_cvoi_offline",
    "app.vjepa_cowa_world_model.train_latent_predictor",
    "app.vjepa_cowa_world_model.train_predictor_rollout_planner",
    "app.vjepa_cowa_world_model.training.cvoi_manual_navtrain_oracle",
    "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime",
    "app.vjepa_cowa_world_model.training.navsim_cvoi_offline_adapter",
    "app.vjepa_cowa_world_model.evaluation.cvoi_direct_epdms",
)
RETAINED_CLI_COMMANDS = (
    (sys.executable, "-m", "app.main", "--help"),
    (sys.executable, "tools/build_cvoi_navsim_scenario_manifest.py", "--help"),
    (sys.executable, "tools/generate_navsim_cf_trajectory_quality.py", "--help"),
    (sys.executable, "tools/run_cvoi_direct_epdms.py", "--help"),
    (sys.executable, "tools/run_cvoi_manual_oracle.py", "--help"),
    ("bash", "-n", RETAINED_SCRIPT),
)


def _run_repository_command(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{inherited_pythonpath}" if inherited_pythonpath else str(REPO_ROOT)
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=120,
    )
    assert (
        completed.returncode == 0
    ), f"command failed: {command!r}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    return completed


def test_repository_exposes_only_the_full_training_configs() -> None:
    assert {path.name for path in FULL_CONFIG_ROOT.glob("*.yaml")} == FULL_CONFIG_NAMES
    all_training_yamls = set((REPO_ROOT / "configs/train").rglob("*.yaml"))
    assert all_training_yamls == {FULL_CONFIG_ROOT / name for name in FULL_CONFIG_NAMES}


def test_repository_exposes_one_manual_agent_and_one_full_epdms_config() -> None:
    assert {path.name for path in AGENT_ROOT.glob("*.yaml")} == {"cvoi_manual_vjepa_world_model_agent.yaml"}
    assert {path.name for path in EPDMS_ROOT.glob("*.yaml")} == {"full_controller.yaml"}
    payload = yaml.safe_load((EPDMS_ROOT / "full_controller.yaml").read_text(encoding="utf-8"))
    assert payload["branch"] == "full"
    assert payload["run_kind"] == "controller"


def test_retained_configs_are_flat() -> None:
    paths = (*FULL_CONFIG_ROOT.glob("*.yaml"), *AGENT_ROOT.glob("*.yaml"), *EPDMS_ROOT.glob("*.yaml"))
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert "extends" not in payload


def test_only_subject_launch_shims_remain() -> None:
    app_root = REPO_ROOT / "app/vjepa_cowa_world_model"
    actual = {path.name for path in app_root.glob("*.py")}
    assert actual == RETAINED_APP_ROOT_FILES
    assert not (REPO_ROOT / "app/main_coops.py").exists()


def test_stale_internal_entry_points_and_guides_are_absent() -> None:
    for relative_path in FORBIDDEN_APP_PUBLIC_PATHS:
        assert not (REPO_ROOT / relative_path).exists()


def test_training_cli_requires_exactly_the_retained_launch_shims() -> None:
    from app.main import parser
    from app.scaffold import RETAINED_TRAIN_SCRIPTS as SCAFFOLD_TRAIN_SCRIPTS

    retained_modules = {path.removesuffix(".py") for path in RETAINED_APP_ROOT_FILES}
    assert RETAINED_TRAIN_SCRIPTS <= retained_modules
    assert set(SCAFFOLD_TRAIN_SCRIPTS) == RETAINED_TRAIN_SCRIPTS

    actions = {action.dest: action for action in parser._actions}
    assert actions["fname"].required is True
    assert actions["train_script"].required is True
    assert set(actions["train_script"].choices) == RETAINED_TRAIN_SCRIPTS

    help_text = parser.format_help()
    for train_script in RETAINED_TRAIN_SCRIPTS:
        assert train_script in help_text
    for stale_surface in ("train_seg", "train_temporal", "train_world_model"):
        assert stale_surface not in help_text


def test_direct_runtime_and_test_dependencies_are_declared() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    runtime_names = {
        re.split(r"[<>=!~ ;\[]", requirement, maxsplit=1)[0].lower() for requirement in project["dependencies"]
    }
    train_names = {
        re.split(r"[<>=!~ ;\[]", requirement, maxsplit=1)[0].lower()
        for requirement in project["optional-dependencies"]["train"]
    }
    assert {"hydra-core", "matplotlib", "omegaconf", "pillow", "scipy"} <= runtime_names | train_names

    test_names = {
        re.split(r"[<>=!~ ;\[]", requirement, maxsplit=1)[0].lower()
        for requirement in project["optional-dependencies"]["test"]
    }
    assert {"black", "build", "flake8", "isort", "pytest"} <= test_names


def test_experiment_and_release_tools_are_separate_exact_categories() -> None:
    actual_tools = {path.name for path in (REPO_ROOT / "tools").glob("*") if path.is_file()}
    assert EXPERIMENT_TOOLS.isdisjoint(RELEASE_TOOLS)
    assert actual_tools & EXPERIMENT_TOOLS == EXPERIMENT_TOOLS
    assert actual_tools - EXPERIMENT_TOOLS == RELEASE_TOOLS
    assert {path for path in (REPO_ROOT / "scripts").rglob("*") if path.is_file()} == {REPO_ROOT / RETAINED_SCRIPT}


def test_retained_scorer_shell_is_executable() -> None:
    assert os.access(REPO_ROOT / RETAINED_SCRIPT, os.X_OK)


def test_retained_scorer_shell_only_exposes_the_stage12_boundary() -> None:
    source = (REPO_ROOT / RETAINED_SCRIPT).read_text(encoding="utf-8")
    assert 'FORWARD_MODE="stage12"' in source
    for stale_surface in (
        "configs/train/navsim/vitl16",
        "encoder_direct",
        "stage2",
        "stage3",
        "train_planner_world_model",
    ):
        assert stale_surface not in source


def test_readme_references_only_existing_repository_paths() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    documented_paths = {match.group(1).rstrip(".,:;") for match in DOCUMENTED_PATH_PATTERN.finditer(readme)}
    missing = sorted(path for path in documented_paths if not (REPO_ROOT / path).exists())
    assert not missing, f"README.md references deleted repository paths: {missing}"


def test_core_modules_cold_import_in_a_fresh_interpreter() -> None:
    source = "import importlib\n" + "\n".join(
        f"importlib.import_module({module_name!r})" for module_name in CORE_COLD_IMPORT_MODULES
    )
    _run_repository_command((sys.executable, "-c", source))


def test_retained_training_lines_do_not_prepend_the_deleted_vendor() -> None:
    for relative_path in (
        "app/vjepa_cowa_world_model/training/lines/planner_world_model.py",
        "app/vjepa_cowa_world_model/training/lines/world_model.py",
    ):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "ddddetection_torchcv" not in source


def test_predictor_train_step_updates_the_enclosing_nonfinite_streak() -> None:
    source_path = REPO_ROOT / "app/vjepa_cowa_world_model/training/lines/world_model.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    run_epoch = next(
        node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "run_epoch"
    )
    train_step = next(
        node for node in ast.walk(run_epoch) if isinstance(node, ast.FunctionDef) and node.name == "train_step"
    )
    captured_names = {name for node in train_step.body if isinstance(node, ast.Nonlocal) for name in node.names}
    assert "consecutive_nonfinite" in captured_names


def test_navsim_agent_imports_with_external_api_test_doubles() -> None:
    source = (
        "from tests.evaluation.test_navsim_stage3_alignment import _load_navsim_agent_module\n"
        "assert _load_navsim_agent_module().VJEPAWorldModelAgent"
    )
    _run_repository_command((sys.executable, "-c", source))


def test_retained_command_line_interfaces_start() -> None:
    for command in RETAINED_CLI_COMMANDS:
        _run_repository_command(command)


def test_direct_epdms_cli_does_not_advertise_a_forced_horizon() -> None:
    completed = _run_repository_command((sys.executable, "tools/run_cvoi_direct_epdms.py", "--help"))
    assert "--horizon" not in completed.stdout


def test_unrelated_vendored_and_asset_roots_are_absent() -> None:
    for relative in FORBIDDEN_ROOTS:
        assert not (REPO_ROOT / relative).exists()
    scene_filter_root = REPO_ROOT / "configs/navsim/scene_filters"
    assert {path.relative_to(scene_filter_root).as_posix() for path in scene_filter_root.rglob("*.yaml")} == {
        "navtrain.yaml",
        "navtest.yaml",
    }
