"""Semantic public-boundary and deterministic Git export contracts."""

from __future__ import annotations

import json
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from tools.check_public_surface import MAX_PUBLIC_FILE_BYTES, PublicSurfaceError
from tools.check_public_surface import main as scan_main
from tools.check_public_surface import scan_public_surface
from tools.export_public_tree import export_public_tree

REPO_ROOT = Path(__file__).resolve().parents[1]
_RETIRED_PROJECT_PATTERN = re.compile(r"[-_]?".join(("drive", "jepa")), flags=re.IGNORECASE)
FULL_CONFIG_NAMES = (
    "01_predictor_lewm_pure.yaml",
    "02_p0_uniform.yaml",
    "03_field_full.yaml",
    "04_calibration_full.yaml",
    "05_p1_full.yaml",
    "06_stop_full.yaml",
    "07_gate_full.yaml",
)
PUBLIC_DIFFUSION_MARKER = "# RISE provenance: independent-diffusion-v1"
PUBLIC_DIFFUSION_PATHS = (
    "app/vjepa_cowa_world_model/models/diffusion_planner.py",
    "app/vjepa_cowa_world_model/diffusion_utils/__init__.py",
    "app/vjepa_cowa_world_model/diffusion_utils/sde.py",
    "app/vjepa_cowa_world_model/diffusion_utils/sampling.py",
)
INVALID_DIFFUSION_PROVENANCE_CONTENTS = (
    pytest.param(
        "# independently written fixture without the required provenance marker\n",
        id="missing-marker",
    ),
    pytest.param(
        f"{PUBLIC_DIFFUSION_MARKER}\n# adapted from XTR\n",
        id="xtr-marker",
    ),
    pytest.param(
        f"{PUBLIC_DIFFUSION_MARKER}\n# private-XtR attribution\n",
        id="mixed-case-private-xtr-marker",
    ),
    pytest.param(
        f"{PUBLIC_DIFFUSION_MARKER}\n# privateXTR attribution\n",
        id="joined-private-xtr-marker",
    ),
    pytest.param(
        f"{PUBLIC_DIFFUSION_MARKER}\n# XTRderived source\n",
        id="joined-xtr-derived-marker",
    ),
)
REQUIRED_FIXTURE_PATHS = (
    "README.md",
    "README_zh-CN.md",
    "docs/configuration.md",
    "docs/reproduction.md",
    "docs/reproduction_zh-CN.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "licenses/Apache-2.0.txt",
    "Makefile",
    "pyproject.toml",
    "app/main.py",
    "app/vjepa_cowa_world_model/train_cvoi_offline.py",
    "app/vjepa_cowa_world_model/train_latent_predictor.py",
    "app/vjepa_cowa_world_model/train_predictor_rollout_planner.py",
    "configs/navsim/cvoi_manual/agent/cvoi_manual_vjepa_world_model_agent.yaml",
    "configs/navsim/scene_filters/navtrain.yaml",
    "configs/navsim/scene_filters/navtest.yaml",
    "configs/eval/navsim/cvoi_manual_epdms/full_controller.yaml",
    "scripts/eval_navsim/eval_navsim_v2_pdms.sh",
    "tools/build_cvoi_navsim_scenario_manifest.py",
    "tools/generate_navsim_cf_trajectory_quality.py",
    "tools/run_cvoi_direct_epdms.py",
    "tools/run_cvoi_manual_oracle.py",
    "tools/check_package.py",
    "tools/check_public_surface.py",
    "tools/export_public_tree.py",
)


def _write(root: Path, relative_path: str, content: str = "public fixture\n") -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_required_surface(root: Path) -> None:
    for relative_path in REQUIRED_FIXTURE_PATHS:
        _write(root, relative_path)
    for relative_path in PUBLIC_DIFFUSION_PATHS:
        _write(root, relative_path, f"{PUBLIC_DIFFUSION_MARKER}\n")
    for config_name in FULL_CONFIG_NAMES:
        _write(root, f"configs/train/navsim/cvoi_manual_full/{config_name}", "stage: fixture\n")


def _rules(root: Path) -> dict[str, set[str]]:
    rules: dict[str, set[str]] = {}
    for violation in scan_public_surface(root):
        rules.setdefault(violation.path, set()).add(violation.rule)
    return rules


def _run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_tracked_sources_contain_no_retired_project_identifiers() -> None:
    violations: list[str] = []
    for relative_path in _run_git(REPO_ROOT, "ls-files").splitlines():
        if _RETIRED_PROJECT_PATTERN.search(relative_path):
            violations.append(relative_path)
            continue
        try:
            content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if _RETIRED_PROJECT_PATTERN.search(content):
            violations.append(relative_path)

    assert not violations, "Retired project identifiers found:\n" + "\n".join(sorted(violations))


def test_worktree_artifacts_contain_no_retired_project_identifiers() -> None:
    retired_pattern = r"[-_]?".join(("drive", "jepa"))
    content_scan = subprocess.run(
        ["rg", "--no-ignore", "--hidden", "-a", "-l", "-i", "-g", "!.git/**", retired_pattern, "."],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert content_scan.returncode in {0, 1}, content_scan.stderr.decode(errors="replace")

    filename_listing = subprocess.run(
        ["rg", "--files", "--no-ignore", "--hidden", "-0", "-g", "!.git/**"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert filename_listing.returncode == 0, filename_listing.stderr.decode(errors="replace")
    filename_pattern = re.compile(retired_pattern, flags=re.IGNORECASE)
    filename_matches = [
        raw_path.decode(errors="replace")
        for raw_path in filename_listing.stdout.split(b"\0")
        if raw_path and filename_pattern.search(raw_path.decode(errors="replace"))
    ]
    content_matches = content_scan.stdout.decode(errors="replace").splitlines()

    assert (
        not content_matches and not filename_matches
    ), "Retired project identifiers found in worktree artifacts:\n" + "\n".join(
        sorted(content_matches + filename_matches)
    )


def _commit_fixture(repository: Path) -> str:
    _run_git(repository, "init", "-b", "main")
    _run_git(repository, "add", "--all")
    _run_git(
        repository,
        "-c",
        "user.name=Public Surface Test",
        "-c",
        "user.email=public-surface@invalid.example",
        "commit",
        "-m",
        "fixture",
    )
    return _run_git(repository, "rev-parse", "HEAD")


def test_scanner_allows_neutral_paths_and_normal_community_files(tmp_path: Path) -> None:
    _write_required_surface(tmp_path)
    _write(tmp_path, "docs/community-notes.md", "Results root: /" + "path/to/rise/results\n")
    _write(tmp_path, "configs/portable_example.py", 'DEFAULT_ROOT = "/' + 'path/to/rise/results"\n')
    _write(tmp_path, "docs/local-development.md", "Loopback: 127." + "0.0.1\n")
    _write(tmp_path, "app/vjepa_cowa_world_model/training/prefix_schedule.py")
    _write(tmp_path, "tests/preparation/test_public_sibling.py")

    assert scan_public_surface(tmp_path) == ()


@pytest.mark.parametrize(
    "relative_path",
    (
        "AGENTS.md",
        "MINIMAL_FILES.txt",
        "docs/superpowers/plans/internal.md",
        "tests/preparation/test_agent_guidance.py",
        "tests/preparation/test_diffusion_contract_recorder.py",
    ),
)
def test_scanner_rejects_preparation_only_paths(tmp_path: Path, relative_path: str) -> None:
    _write_required_surface(tmp_path)
    _write(tmp_path, relative_path)

    assert "preparation-only path" in _rules(tmp_path)[relative_path]


@pytest.mark.parametrize(
    "relative_path",
    (
        ".gitlab-ci.yml",
        ".github/workflows/ci.yml",
        ".agents/operator.md",
        "configs/generated/auto_a100/legacy.yaml",
        "scripts/auto_a100/run_pipeline.sh",
        "tools/cvoi_dag_runner.py",
        "tools/cvoi_scheduler.py",
        "tools/orchestrator.py",
        "tools/automation.py",
        "tools/internal_operations.py",
        "assets/checkpoint-index.txt",
        "ddddetection_torchcv/module.py",
    ),
)
def test_scanner_rejects_hosted_ci_legacy_automation_and_internal_operations(
    tmp_path: Path, relative_path: str
) -> None:
    _write_required_surface(tmp_path)
    _write(tmp_path, relative_path)

    assert "forbidden repository surface" in _rules(tmp_path)[relative_path]


def test_scanner_rejects_a_legacy_dag_directory(tmp_path: Path) -> None:
    _write_required_surface(tmp_path)
    _write(tmp_path, "scripts/cvoi_dag/run.py")

    assert "forbidden repository surface" in _rules(tmp_path)["scripts/cvoi_dag"]


@pytest.mark.parametrize(
    "content",
    (
        "root = '/" + "disk/private/results'\n",
        "owner = 'cheng" + "hao.he'\n",
        "url = 'https://harbor." + "cowa" + "robot" + ".cn/image'\n",
        "host = '172." + "16.1.10'\n",
        "link = 'https://github.com/fair" + "internal/repository'\n",
    ),
)
def test_scanner_rejects_internal_paths_identities_domains_and_ips(tmp_path: Path, content: str) -> None:
    _write_required_surface(tmp_path)
    _write(tmp_path, "notes/private.txt", content)

    assert "private deployment marker" in _rules(tmp_path)["notes/private.txt"]


def test_retained_weighted_sampler_contains_no_private_repository_marker(tmp_path: Path) -> None:
    _write_required_surface(tmp_path)
    relative_path = "src/datasets/utils/weighted_sampler.py"
    destination = tmp_path / relative_path
    destination.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / relative_path, destination)

    assert "private deployment marker" not in _rules(tmp_path).get(relative_path, set())


def test_scanner_rejects_credentials_without_echoing_the_value(tmp_path: Path) -> None:
    _write_required_surface(tmp_path)
    redaction_probe = "release-credential-value-12345"
    _write(tmp_path, "settings.py", "RISE_API_" + f'KEY = "{redaction_probe}"\n')

    violations = scan_public_surface(tmp_path)

    assert any(item.path == "settings.py" and item.rule == "credential-like assignment" for item in violations)
    assert all(redaction_probe not in str(item) for item in violations)


@pytest.mark.parametrize("key_kind", ("", "RSA ", "OPENSSH "))
def test_scanner_rejects_private_key_headers_without_echoing_key_material(tmp_path: Path, key_kind: str) -> None:
    _write_required_surface(tmp_path)
    key_material = "highly-sensitive-key-material"
    _write(
        tmp_path,
        "notes.txt",
        "-----BEGIN " + key_kind + "PRIVATE KEY-----\n" + key_material + "\n-----END PRIVATE KEY-----\n",
    )

    violations = scan_public_surface(tmp_path)

    assert any(item.path == "notes.txt" and item.rule == "private-key header" for item in violations)
    assert all(key_material not in str(item) for item in violations)


def test_scanner_rejects_symlinks(tmp_path: Path) -> None:
    _write_required_surface(tmp_path)
    target = _write(tmp_path, "notes/target.txt")
    (tmp_path / "notes/link.txt").symlink_to(target.name)

    assert "symbolic link" in _rules(tmp_path)["notes/link.txt"]


def test_scanner_rejects_oversized_files_without_reading_them(tmp_path: Path) -> None:
    _write_required_surface(tmp_path)
    oversized = tmp_path / "large.txt"
    with oversized.open("wb") as stream:
        stream.truncate(MAX_PUBLIC_FILE_BYTES + 1)

    assert "oversized file" in _rules(tmp_path)["large.txt"]


@pytest.mark.parametrize("relative_path", ("results/model.pt", "data/oracle.sqlite3", "keys/id_ed25519"))
def test_scanner_rejects_checkpoint_data_and_key_like_files(tmp_path: Path, relative_path: str) -> None:
    _write_required_surface(tmp_path)
    _write(tmp_path, relative_path)

    assert "prohibited release artifact" in _rules(tmp_path)[relative_path]


def test_scanner_does_not_hide_artifacts_just_because_gitignore_mentions_them(tmp_path: Path) -> None:
    _write_required_surface(tmp_path)
    _write(tmp_path, ".gitignore", "results/\n")
    _write(tmp_path, "results/model.ckpt")

    assert "prohibited release artifact" in _rules(tmp_path)["results/model.ckpt"]


def test_scanner_skips_only_documented_cache_directories(tmp_path: Path) -> None:
    _write_required_surface(tmp_path)
    _write(tmp_path, ".pytest_cache/private.pem", "-----BEGIN " + "PRIVATE KEY-----\n")
    _write(tmp_path, "build/model.pt")
    _write(tmp_path, "dist/data.sqlite3")
    _write(tmp_path, "rise_wam.egg-info/PKG-INFO", "owner = 'cheng" + "hao.he'\n")

    assert scan_public_surface(tmp_path) == ()


def test_scanner_skips_a_git_worktree_pointer_file(tmp_path: Path) -> None:
    _write_required_surface(tmp_path)
    _write(tmp_path, ".git", "gitdir: /" + "disk/private/worktree\n")

    assert scan_public_surface(tmp_path) == ()


@pytest.mark.parametrize("relative_path", ("README_zh-CN.md", "CONTRIBUTING.md", "Makefile"))
def test_scanner_reports_missing_required_public_files(tmp_path: Path, relative_path: str) -> None:
    _write_required_surface(tmp_path)
    (tmp_path / relative_path).unlink()

    assert "required public path is missing" in _rules(tmp_path)[relative_path]


def test_scanner_cli_reports_an_invalid_root_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = scan_main([str(tmp_path / "missing")])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "not a directory" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("relative_path", PUBLIC_DIFFUSION_PATHS)
def test_scanner_accepts_each_independent_diffusion_provenance_marker(tmp_path: Path, relative_path: str) -> None:
    _write_required_surface(tmp_path)

    assert _rules(tmp_path).get(relative_path, set()) == set()


@pytest.mark.parametrize("relative_path", PUBLIC_DIFFUSION_PATHS)
def test_scanner_accepts_verified_diffusion_provenance_with_ordinary_extra_text(
    tmp_path: Path,
    relative_path: str,
) -> None:
    _write_required_surface(tmp_path)
    _write(
        tmp_path,
        relative_path,
        f"{PUBLIC_DIFFUSION_MARKER}\n# ordinary extra validation text\n",
    )

    assert _rules(tmp_path).get(relative_path, set()) == set()


@pytest.mark.parametrize("relative_path", PUBLIC_DIFFUSION_PATHS)
def test_scanner_reports_each_missing_public_diffusion_source_as_required(tmp_path: Path, relative_path: str) -> None:
    _write_required_surface(tmp_path)
    (tmp_path / relative_path).unlink()

    assert _rules(tmp_path).get(relative_path, set()) == {"required public path is missing"}


@pytest.mark.parametrize("relative_path", PUBLIC_DIFFUSION_PATHS)
@pytest.mark.parametrize("invalid_content", INVALID_DIFFUSION_PROVENANCE_CONTENTS)
def test_scanner_rejects_each_unverified_diffusion_provenance(
    tmp_path: Path,
    relative_path: str,
    invalid_content: str,
) -> None:
    _write_required_surface(tmp_path)
    _write(tmp_path, relative_path, invalid_content)

    assert _rules(tmp_path).get(relative_path, set()) == {"unverified diffusion provenance"}


def test_export_reads_the_explicit_commit_not_the_worktree_and_preserves_modes(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    _write_required_surface(repository)
    _write(repository, "docs/community-notes.md", "committed\n")
    executable = repository / "scripts/eval_navsim/eval_navsim_v2_pdms.sh"
    executable.chmod(0o755)
    _write(repository, "AGENTS.md")
    _write(repository, "MINIMAL_FILES.txt")
    _write(repository, "docs/superpowers/plans/preparation.md")
    _write(repository, "tests/preparation/test_agent_guidance.py")
    _write(repository, "tests/preparation/test_diffusion_contract_recorder.py")
    _write(repository, "tests/preparation/test_public_sibling.py")
    _write(repository, "tests/models/test_diffusion_planner_behavior_contract.py")
    _write(repository, "tests/test_minimal_repository_surface.py")
    commit = _commit_fixture(repository)
    _write(repository, "docs/community-notes.md", "dirty worktree\n")
    _write(repository, "untracked.txt")
    destination = tmp_path / "export"

    facts = export_public_tree(repository=repository, source_ref=commit, destination=destination)

    assert facts.source_commit == commit
    assert (destination / "docs/community-notes.md").read_text(encoding="utf-8") == "committed\n"
    assert stat.S_IMODE((destination / "scripts/eval_navsim/eval_navsim_v2_pdms.sh").stat().st_mode) == 0o755
    assert not (destination / "untracked.txt").exists()
    assert not (destination / "AGENTS.md").exists()
    assert not (destination / "MINIMAL_FILES.txt").exists()
    assert not (destination / "docs/superpowers").exists()
    assert not (destination / "tests/preparation/test_agent_guidance.py").exists()
    assert not (destination / "tests/preparation/test_diffusion_contract_recorder.py").exists()
    assert (destination / "tests/preparation/test_public_sibling.py").is_file()
    assert (destination / "tests/models/test_diffusion_planner_behavior_contract.py").is_file()
    assert (destination / "tests/test_minimal_repository_surface.py").is_file()
    assert not (destination / ".git").exists()


def test_export_digest_and_report_are_deterministic_and_expect_report_is_read_only(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    _write_required_surface(repository)
    commit = _commit_fixture(repository)
    report = tmp_path / "audit.json"

    first = export_public_tree(
        repository=repository,
        source_ref="HEAD",
        destination=tmp_path / "first",
        report=report,
    )
    report_bytes = report.read_bytes()
    second = export_public_tree(
        repository=repository,
        source_ref=commit,
        destination=tmp_path / "second",
        expect_report=report,
    )

    assert first == second
    assert report.read_bytes() == report_bytes
    assert json.loads(report_bytes) == {
        "file_count": first.file_count,
        "public_tree_sha256": first.public_tree_sha256,
        "source_commit": commit,
    }


def test_export_refuses_to_overwrite_a_report_or_nonempty_destination(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    _write_required_surface(repository)
    _commit_fixture(repository)
    report = _write(tmp_path, "audit.json", "existing\n")
    destination = tmp_path / "destination"

    with pytest.raises(PublicSurfaceError, match="report already exists"):
        export_public_tree(repository=repository, source_ref="HEAD", destination=destination, report=report)
    assert not destination.exists()

    destination.mkdir()
    _write(destination, "keep.txt")
    with pytest.raises(PublicSurfaceError, match="destination must be absent or empty"):
        export_public_tree(repository=repository, source_ref="HEAD", destination=destination)
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "public fixture\n"


def test_export_report_mismatch_stops_before_creating_destination(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    _write_required_surface(repository)
    _commit_fixture(repository)
    expected = tmp_path / "expected.json"
    expected.write_text(
        json.dumps({"source_commit": "0" * 40, "public_tree_sha256": "1" * 64, "file_count": 1}),
        encoding="utf-8",
    )
    destination = tmp_path / "destination"

    with pytest.raises(PublicSurfaceError, match="does not match expected report"):
        export_public_tree(
            repository=repository,
            source_ref="HEAD",
            destination=destination,
            expect_report=expected,
        )

    assert not destination.exists()


@pytest.mark.parametrize("report_kind", ("missing_parent", "inside_destination", "aliased_inside_destination"))
def test_export_rejects_an_unsafe_report_location_before_publishing(tmp_path: Path, report_kind: str) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    _write_required_surface(repository)
    _commit_fixture(repository)
    destination = tmp_path / "destination"
    if report_kind == "missing_parent":
        report = tmp_path / "missing" / "audit.json"
    elif report_kind == "inside_destination":
        destination.mkdir()
        report = destination / "audit.json"
    else:
        destination.mkdir()
        alias = tmp_path / "alias"
        alias.mkdir()
        report = alias / ".." / destination.name / "audit.json"

    with pytest.raises(PublicSurfaceError, match="report"):
        export_public_tree(
            repository=repository,
            source_ref="HEAD",
            destination=destination,
            report=report,
        )

    assert not destination.exists() or not any(destination.iterdir())


def test_export_rejects_committed_symlinks_without_touching_destination(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    _write_required_surface(repository)
    (repository / "unsafe-link").symlink_to("README.md")
    _commit_fixture(repository)
    destination = tmp_path / "destination"

    with pytest.raises(PublicSurfaceError, match="symbolic link"):
        export_public_tree(repository=repository, source_ref="HEAD", destination=destination)

    assert not destination.exists()


@pytest.mark.parametrize("relative_path", PUBLIC_DIFFUSION_PATHS)
@pytest.mark.parametrize("invalid_content", INVALID_DIFFUSION_PROVENANCE_CONTENTS)
def test_export_rejects_unverified_diffusion_provenance_before_publishing(
    tmp_path: Path,
    relative_path: str,
    invalid_content: str,
) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    _write_required_surface(repository)
    _write(repository, relative_path, invalid_content)
    _commit_fixture(repository)
    destination = tmp_path / "destination"

    with pytest.raises(PublicSurfaceError) as error:
        export_public_tree(repository=repository, source_ref="HEAD", destination=destination)

    assert not destination.exists()
    message = str(error.value)
    assert message.count("unverified diffusion provenance") == 1
    assert relative_path in message


def test_export_aggregates_all_unverified_diffusion_paths_before_publishing(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    _write_required_surface(repository)
    for index, relative_path in enumerate(PUBLIC_DIFFUSION_PATHS):
        content = (
            "# independently written fixture without the required provenance marker\n"
            if index % 2 == 0
            else f"{PUBLIC_DIFFUSION_MARKER}\n# private-XtR attribution\n"
        )
        _write(repository, relative_path, content)
    _commit_fixture(repository)
    destination = tmp_path / "destination"

    with pytest.raises(PublicSurfaceError) as error:
        export_public_tree(repository=repository, source_ref="HEAD", destination=destination)

    assert not destination.exists()
    message = str(error.value)
    assert message.count("unverified diffusion provenance") == len(PUBLIC_DIFFUSION_PATHS)
    for relative_path in PUBLIC_DIFFUSION_PATHS:
        assert relative_path in message
