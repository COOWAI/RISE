"""Public packaging contract for the ``rise-wam`` source release."""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from wheel.wheelfile import WheelFile

from tools import check_package

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 preparation environment
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_FILTER_HASHES = {
    "configs/navsim/scene_filters/navtrain.yaml": "c37fea567a0cfdbc29076cca893d4f5dd32db59baec18ae214527206d6b64e6f",
    "configs/navsim/scene_filters/navtest.yaml": "61284edf5003c0291f843ce9817c822ba306609a62d54544223adae3fc7fc9cd",
}


def _write_minimal_smoke_wheel(wheel_path: Path, *, include_scene_filters: bool = True) -> None:
    roots_source = """\
from pathlib import Path

_ARTIFACT_ROOT = Path(__file__).resolve().parents[3]

def resolve_formal_v2_navsim_scene_filter_path(path):
    return _ARTIFACT_ROOT / Path(path)
"""
    modules = {
        "app/main.py": "if __name__ == '__main__':\n    pass\n",
        "app/vjepa_cowa_world_model/__init__.py": "",
        "app/vjepa_cowa_world_model/training/__init__.py": "",
        "app/vjepa_cowa_world_model/training/cvoi_manual_navtrain_oracle.py": "",
        "app/vjepa_cowa_world_model/training/cvoi_formal_v2_navsim_roots.py": roots_source,
        "src/models/vision_transformer.py": "",
        "tools/run_cvoi_manual_oracle.py": "if __name__ == '__main__':\n    pass\n",
        "tools/run_cvoi_direct_epdms.py": "if __name__ == '__main__':\n    pass\n",
        "rise_wam-0.1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: rise-wam\nVersion: 0.1.0\n",
        "rise_wam-0.1.0.dist-info/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: RISE tests\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    if include_scene_filters:
        modules.update(
            {member_name: (REPO_ROOT / member_name).read_text(encoding="utf-8") for member_name in SCENE_FILTER_HASHES}
        )
    with WheelFile(wheel_path, "w") as wheel:
        for member_name, source in modules.items():
            wheel.writestr(member_name, source.encode("utf-8"))


def _write_validation_wheel(
    wheel_path: Path,
    *,
    extra_members: tuple[tuple[zipfile.ZipInfo | str, bytes], ...] = (),
    payload_overrides: dict[str, bytes] | None = None,
) -> None:
    payloads = {
        member_name: (
            (REPO_ROOT / member_name).read_bytes() if member_name in SCENE_FILTER_HASHES else b"release placeholder\n"
        )
        for member_name in check_package.EXPECTED_WHEEL_MEMBERS
    }
    payloads.update(
        {
            "rise_wam-0.1.0.dist-info/METADATA": (b"Metadata-Version: 2.1\nName: rise-wam\nVersion: 0.1.0\n"),
            "rise_wam-0.1.0.dist-info/LICENSE": b"MIT License\n",
        }
    )
    if payload_overrides is not None:
        payloads.update(payload_overrides)
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for member_name, payload in payloads.items():
            wheel.writestr(member_name, payload)
        for member, payload in extra_members:
            wheel.writestr(member, payload)


def _build_release_wheel(tmp_path: Path) -> Path:
    source_root = tmp_path / "source"
    distribution_root = tmp_path / "dist"
    check_package._copy_source_tree(source_root)
    distribution_root.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(distribution_root),
            str(source_root),
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    wheels = list(distribution_root.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def _write_shadow_module(path: Path, marker: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "from pathlib import Path\n" f"Path({str(marker)!r}).write_text('shadowed\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )


def _add_tar_directory(stream: tarfile.TarFile, member_name: str) -> None:
    member = tarfile.TarInfo(member_name)
    member.type = tarfile.DIRTYPE
    member.mode = 0o755
    stream.addfile(member)


def _load_pyproject() -> dict[str, object]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _dependency_names(requirements: list[str]) -> set[str]:
    return {
        re.split(r"[\[<>=!~; ]", requirement.lower().replace("_", "-"), maxsplit=1)[0] for requirement in requirements
    }


def test_pyproject_is_the_single_rise_wam_metadata_authority() -> None:
    payload = _load_pyproject()
    assert payload["build-system"]["build-backend"] == "setuptools.build_meta"

    project = payload["project"]
    assert project["name"] == "rise-wam"
    assert project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.11"
    assert project["readme"] == "README.md"
    assert "RISE" in project["description"]
    assert project["license"] in ("MIT", {"text": "MIT"}, {"file": "LICENSE"})
    assert "urls" not in project

    discovery = payload["tool"]["setuptools"]["packages"]["find"]
    assert discovery["where"] == ["."]
    assert discovery["include"] == ["app*", "configs*", "src*", "tools*"]
    assert discovery["namespaces"] is True
    assert payload["tool"]["setuptools"]["package-data"] == {"configs.navsim.scene_filters": ["*.yaml"]}

    for legacy_authority in ("setup.py", "requirements.txt", "requirements-test.txt"):
        assert not (REPO_ROOT / legacy_authority).exists()


def test_dependency_scopes_are_explicit_and_test_is_self_sufficient() -> None:
    project = _load_pyproject()["project"]
    base = _dependency_names(project["dependencies"])
    optional = project["optional-dependencies"]
    train = _dependency_names(optional["train"])
    test = _dependency_names(optional["test"])

    assert {"torch", "torchvision", "pyyaml", "numpy", "iopath", "timm"} <= base
    assert {"tensorboard", "wandb", "transformers", "peft", "decord", "pandas", "opencv-python"} <= train
    assert base | train <= test
    assert {"black", "flake8", "isort", "pytest", "build", "setuptools", "wheel"} <= test

    test_requirements = optional["test"]
    assert "black==24.4.2" in test_requirements
    assert "flake8==7.0.0" in test_requirements
    assert "isort==5.13.2" in test_requirements


def test_isort_policy_does_not_depend_on_profile_version_defaults() -> None:
    isort_config = _load_pyproject()["tool"]["isort"]

    assert isort_config["profile"] == "black"
    assert isort_config["line_length"] == 119
    assert isort_config["split_on_trailing_comma"] is False


def test_public_source_tree_matches_declared_isort_policy() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "isort", "app", "src", "tests", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"


def test_clean_tree_gate_passes_only_when_git_status_succeeds_and_is_empty(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    makefile = REPO_ROOT / "Makefile"

    clean = subprocess.run(
        ["make", "-f", str(makefile), "clean-tree-check"],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )
    assert clean.returncode == 0, f"stdout:\n{clean.stdout}\nstderr:\n{clean.stderr}"

    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    dirty = subprocess.run(
        ["make", "-f", str(makefile), "clean-tree-check"],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )
    assert dirty.returncode != 0
    (repository / "untracked.txt").unlink()

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text('#!/bin/sh\n: > "$FAKE_GIT_MARKER"\nexit 23\n', encoding="utf-8")
    fake_git.chmod(0o755)
    fake_git_marker = tmp_path / "fake-git-called"
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["FAKE_GIT_MARKER"] = str(fake_git_marker)
    failed_git = subprocess.run(
        ["make", "-f", str(makefile), "clean-tree-check"],
        cwd=repository,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert failed_git.returncode != 0
    assert fake_git_marker.is_file()


def test_release_check_delegates_to_the_fail_closed_clean_tree_gate() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "\t$(MAKE) clean-tree-check\n" in makefile


def test_sdist_manifest_carries_public_docs_configs_and_notices() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    required_fragments = (
        "README.md",
        "README_zh-CN.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "licenses",
        "docs",
        "configs",
        "tools",
        "scripts/eval_navsim/eval_navsim_v2_pdms.sh",
    )
    for fragment in required_fragments:
        assert fragment in manifest
    assert "recursive-include configs *.md *.yaml *.yml" in manifest
    assert "_".join(("Drive", "JEPA")) not in manifest
    for preparation_only in ("AGENTS.md", "MINIMAL_FILES.txt", "docs/superpowers", "tests/preparation"):
        assert f"exclude {preparation_only}" in manifest or f"prune {preparation_only}" in manifest


def test_package_checker_requires_exact_scene_filter_hashes() -> None:
    assert set(SCENE_FILTER_HASHES) <= check_package.EXPECTED_WHEEL_MEMBERS
    assert check_package.EXPECTED_WHEEL_MEMBER_SHA256 == SCENE_FILTER_HASHES
    assert set(SCENE_FILTER_HASHES) <= check_package.EXPECTED_SDIST_MEMBERS
    assert check_package.EXPECTED_SDIST_MEMBER_SHA256 == SCENE_FILTER_HASHES


def test_built_wheel_contains_exact_scene_filters(tmp_path: Path) -> None:
    wheel_path = _build_release_wheel(tmp_path)

    with zipfile.ZipFile(wheel_path) as wheel:
        scene_filter_members = {name for name in wheel.namelist() if name.startswith("configs/navsim/scene_filters/")}
        assert scene_filter_members == set(SCENE_FILTER_HASHES)
        for member_name, expected_hash in SCENE_FILTER_HASHES.items():
            assert hashlib.sha256(wheel.read(member_name)).hexdigest() == expected_hash


def test_package_checker_rejects_modified_wheel_scene_filter_bytes(tmp_path: Path) -> None:
    wheel_path = tmp_path / "modified-scene-filter.whl"
    _write_validation_wheel(
        wheel_path,
        payload_overrides={"configs/navsim/scene_filters/navtrain.yaml": b"modified\n"},
    )

    with pytest.raises(RuntimeError, match=r"wheel member SHA256 mismatch.*navtrain\.yaml"):
        check_package._validate_wheel(wheel_path)


def test_package_checker_rejects_modified_scene_filter_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "rise-wam-0.1.0.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for member_name in sorted(check_package.EXPECTED_SDIST_MEMBERS):
            payload = (
                (REPO_ROOT / member_name).read_bytes()
                if member_name in SCENE_FILTER_HASHES
                else b"release placeholder\n"
            )
            if member_name == "configs/navsim/scene_filters/navtrain.yaml":
                payload += b"# modified\n"
            member = tarfile.TarInfo(f"rise-wam-0.1.0/{member_name}")
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match=r"SHA256 mismatch.*navtrain\.yaml"):
        check_package._validate_sdist(archive)


@pytest.mark.parametrize(
    ("member_prefix", "case_name"),
    [("/", "absolute"), ("../", "parent-traversal"), ("//server/share/", "unc")],
)
def test_package_checker_rejects_unsafe_sdist_member_roots(
    tmp_path: Path,
    member_prefix: str,
    case_name: str,
) -> None:
    archive = tmp_path / f"{case_name}.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for member_name in sorted(check_package.EXPECTED_SDIST_MEMBERS):
            payload = (
                (REPO_ROOT / member_name).read_bytes()
                if member_name in SCENE_FILTER_HASHES
                else b"release placeholder\n"
            )
            member = tarfile.TarInfo(f"{member_prefix}{member_name}")
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="unsafe sdist member path"):
        check_package._validate_sdist(archive)


@pytest.mark.parametrize(
    ("member_prefix", "case_name"),
    [
        pytest.param("C:/", "drive-rooted", id="drive-rooted"),
        pytest.param("C:rise-wam-0.1.0/", "drive-relative", id="drive-relative"),
    ],
)
def test_package_checker_rejects_drive_qualified_sdist_members(
    tmp_path: Path,
    member_prefix: str,
    case_name: str,
) -> None:
    archive = tmp_path / f"{case_name}.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for member_name in sorted(check_package.EXPECTED_SDIST_MEMBERS):
            payload = (
                (REPO_ROOT / member_name).read_bytes()
                if member_name in SCENE_FILTER_HASHES
                else b"release placeholder\n"
            )
            member = tarfile.TarInfo(f"{member_prefix}{member_name}")
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="unsafe sdist member path.*drive-qualified"):
        check_package._validate_sdist(archive)


@pytest.mark.parametrize(
    "case_name",
    ["duplicate-root-directory", "duplicate-nested-directory", "mixed-directory-file"],
)
def test_package_checker_rejects_duplicate_normalized_sdist_members(
    tmp_path: Path,
    case_name: str,
) -> None:
    archive = tmp_path / f"{case_name}.tar.gz"
    root = "rise-wam-0.1.0"
    with tarfile.open(archive, "w:gz") as stream:
        _add_tar_directory(stream, f"{root}/")
        if case_name == "duplicate-root-directory":
            _add_tar_directory(stream, f"{root}/")
        elif case_name == "duplicate-nested-directory":
            _add_tar_directory(stream, f"{root}/configs/")
            _add_tar_directory(stream, f"{root}/configs/")
        else:
            _add_tar_directory(stream, f"{root}/README.md")

        for member_name in sorted(check_package.EXPECTED_SDIST_MEMBERS):
            payload = (
                (REPO_ROOT / member_name).read_bytes()
                if member_name in SCENE_FILTER_HASHES
                else b"release placeholder\n"
            )
            member = tarfile.TarInfo(f"{root}/{member_name}")
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="duplicate sdist member"):
        check_package._validate_sdist(archive)


@pytest.mark.parametrize(
    "unsafe_name",
    [
        pytest.param(r"rise-wam-0.1.0/docs\..\..\escape.txt", id="backslash"),
        pytest.param("rise-wam-0.1.0/docs/\x01escape.txt", id="control-character"),
        pytest.param("rise-wam-0.1.0/docs/\x85escape.txt", id="c1-control-character"),
    ],
)
def test_package_checker_rejects_unsafe_sdist_member_names(tmp_path: Path, unsafe_name: str) -> None:
    archive = tmp_path / "unsafe-name.tar.gz"
    root = "rise-wam-0.1.0"
    with tarfile.open(archive, "w:gz") as stream:
        _add_tar_directory(stream, f"{root}/")
        for member_name in sorted(check_package.EXPECTED_SDIST_MEMBERS):
            payload = (
                (REPO_ROOT / member_name).read_bytes()
                if member_name in SCENE_FILTER_HASHES
                else b"release placeholder\n"
            )
            member = tarfile.TarInfo(f"{root}/{member_name}")
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
        unsafe_member = tarfile.TarInfo(unsafe_name)
        unsafe_member.size = len(b"unsafe\n")
        stream.addfile(unsafe_member, io.BytesIO(b"unsafe\n"))

    with pytest.raises(RuntimeError, match="unsafe sdist member path"):
        check_package._validate_sdist(archive)


def test_package_checker_rejects_sdist_links(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.tar.gz"
    root = "rise-wam-0.1.0"
    with tarfile.open(archive, "w:gz") as stream:
        _add_tar_directory(stream, f"{root}/")
        for member_name in sorted(check_package.EXPECTED_SDIST_MEMBERS):
            payload = (
                (REPO_ROOT / member_name).read_bytes()
                if member_name in SCENE_FILTER_HASHES
                else b"release placeholder\n"
            )
            member = tarfile.TarInfo(f"{root}/{member_name}")
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
        link = tarfile.TarInfo(f"{root}/unsafe-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../../../outside"
        stream.addfile(link)

    with pytest.raises(RuntimeError, match="unsupported sdist member type"):
        check_package._validate_sdist(archive)


@pytest.mark.parametrize(
    ("case_name", "member"),
    [
        pytest.param("parent-traversal", "../escape.py", id="parent-traversal"),
        pytest.param("backslash", r"docs\..\escape.py", id="backslash"),
        pytest.param("control-character", "docs/\x01escape.py", id="control-character"),
        pytest.param("c1-control-character", "docs/\x85escape.py", id="c1-control-character"),
        pytest.param("duplicate", "app/main.py", id="duplicate"),
    ],
)
def test_package_checker_rejects_unsafe_wheel_members(
    tmp_path: Path,
    case_name: str,
    member: str,
) -> None:
    wheel_path = tmp_path / f"{case_name}.whl"
    _write_validation_wheel(wheel_path, extra_members=((member, b"unsafe\n"),))

    with pytest.raises(RuntimeError, match="unsafe wheel member|duplicate wheel member"):
        check_package._validate_wheel(wheel_path)


def test_package_checker_rejects_wheel_symlinks(tmp_path: Path) -> None:
    wheel_path = tmp_path / "symlink.whl"
    link = zipfile.ZipInfo("unsafe-link.py")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    _write_validation_wheel(wheel_path, extra_members=((link, b"../../outside"),))

    with pytest.raises(RuntimeError, match="unsupported wheel member type"):
        check_package._validate_wheel(wheel_path)


def test_smoke_installed_wheel_precedes_caller_purelib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "rise_wam-0.1.0-py3-none-any.whl"
    _write_minimal_smoke_wheel(wheel)

    caller_purelib = tmp_path / "caller-purelib"
    marker_paths = {
        module_name: tmp_path / f"{module_name.replace('.', '-')}.marker"
        for module_name in (
            "app.main",
            "tools.run_cvoi_manual_oracle",
            "tools.run_cvoi_direct_epdms",
        )
    }
    for module_name, marker in marker_paths.items():
        _write_shadow_module(caller_purelib / f"{module_name.replace('.', '/')}.py", marker)
    monkeypatch.setattr(check_package.sysconfig, "get_path", lambda _name: str(caller_purelib))

    smoke_root = tmp_path / "smoke"
    smoke_root.mkdir()
    check_package._smoke_installed_wheel(wheel, smoke_root)

    executed_shadows = sorted(module_name for module_name, marker in marker_paths.items() if marker.exists())
    assert not executed_shadows, f"caller purelib shadow modules executed: {executed_shadows}"


def test_smoke_installed_wheel_rejects_missing_scene_filters(tmp_path: Path) -> None:
    wheel = tmp_path / "rise_wam-0.1.0-py3-none-any.whl"
    _write_minimal_smoke_wheel(wheel, include_scene_filters=False)

    smoke_root = tmp_path / "smoke"
    smoke_root.mkdir()
    with pytest.raises(RuntimeError, match="scene filter"):
        check_package._smoke_installed_wheel(wheel, smoke_root)


def test_package_checker_builds_and_smoke_tests_installed_artifacts() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/check_package.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    assert "wheel: ok" in completed.stdout
    assert "sdist: ok" in completed.stdout
    assert "installed artifact smoke checks: ok" in completed.stdout
