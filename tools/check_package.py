#!/usr/bin/env python3
"""Build and validate the RISE wheel and source distribution without network access."""

from __future__ import annotations

import hashlib
import ntpath
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WHEEL_MEMBERS = {
    "app/main.py",
    "app/vjepa_cowa_world_model/__init__.py",
    "configs/navsim/scene_filters/navtest.yaml",
    "configs/navsim/scene_filters/navtrain.yaml",
    "src/models/vision_transformer.py",
    "tools/run_cvoi_direct_epdms.py",
    "tools/run_cvoi_manual_oracle.py",
}
EXPECTED_SDIST_MEMBERS = {
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "README_zh-CN.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "configs/eval/navsim/cvoi_manual_epdms/full_controller.yaml",
    "configs/navsim/scene_filters/navtest.yaml",
    "configs/navsim/scene_filters/navtrain.yaml",
    "configs/train/navsim/cvoi_manual_full/01_predictor_lewm_pure.yaml",
    "configs/train/navsim/cvoi_manual_full/02_p0_uniform.yaml",
    "configs/train/navsim/cvoi_manual_full/03_field_full.yaml",
    "configs/train/navsim/cvoi_manual_full/04_calibration_full.yaml",
    "configs/train/navsim/cvoi_manual_full/05_p1_full.yaml",
    "configs/train/navsim/cvoi_manual_full/06_stop_full.yaml",
    "configs/train/navsim/cvoi_manual_full/07_gate_full.yaml",
    "docs/configuration.md",
    "docs/reproduction.md",
    "docs/reproduction_zh-CN.md",
    "licenses/Apache-2.0.txt",
    "scripts/eval_navsim/eval_navsim_v2_pdms.sh",
    "tools/check_package.py",
    "tools/run_cvoi_direct_epdms.py",
    "tools/run_cvoi_manual_oracle.py",
}
EXPECTED_SDIST_MEMBER_SHA256 = {
    "configs/navsim/scene_filters/navtrain.yaml": "c37fea567a0cfdbc29076cca893d4f5dd32db59baec18ae214527206d6b64e6f",
    "configs/navsim/scene_filters/navtest.yaml": "61284edf5003c0291f843ce9817c822ba306609a62d54544223adae3fc7fc9cd",
}
EXPECTED_WHEEL_MEMBER_SHA256 = {
    "configs/navsim/scene_filters/navtrain.yaml": "c37fea567a0cfdbc29076cca893d4f5dd32db59baec18ae214527206d6b64e6f",
    "configs/navsim/scene_filters/navtest.yaml": "61284edf5003c0291f843ce9817c822ba306609a62d54544223adae3fc7fc9cd",
}
FORBIDDEN_SDIST_MEMBERS = {
    "AGENTS.md",
    "MINIMAL_FILES.txt",
}


def _run(command: list[str], *, cwd: Path, environment: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"package check command failed ({rendered})\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _copy_source_tree(destination: Path) -> None:
    ignored = shutil.ignore_patterns(
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "*.egg-info",
        "__pycache__",
        "build",
        "dist",
    )
    shutil.copytree(REPOSITORY_ROOT, destination, ignore=ignored)


def _validated_posix_member_components(
    raw_name: str,
    *,
    archive_kind: str,
    is_directory: bool,
) -> tuple[str, ...]:
    drive, _ = ntpath.splitdrive(raw_name)
    if drive:
        raise RuntimeError(
            f"{archive_kind} contains unsafe {archive_kind} member path {raw_name!r}: drive-qualified path"
        )
    if not raw_name or "\\" in raw_name or any(unicodedata.category(character) == "Cc" for character in raw_name):
        raise RuntimeError(f"{archive_kind} contains unsafe {archive_kind} member path {raw_name!r}")

    raw_path = PurePosixPath(raw_name)
    if raw_path.is_absolute():
        raise RuntimeError(f"{archive_kind} contains unsafe {archive_kind} member path {raw_name!r}: absolute path")

    components = raw_name.split("/")
    if is_directory and raw_name.endswith("/"):
        components = components[:-1]
    invalid_component = next((component for component in components if component in {"", ".", ".."}), None)
    if invalid_component is not None:
        raise RuntimeError(
            f"{archive_kind} contains unsafe {archive_kind} member path {raw_name!r}: "
            f"invalid component {invalid_component!r}"
        )

    normalized_path = PurePosixPath(*components)
    if normalized_path.parts != tuple(components):
        raise RuntimeError(f"{archive_kind} contains unsafe {archive_kind} member path {raw_name!r}: not normalized")
    return tuple(components)


def _validated_sdist_file_members(stream: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    validated_members: list[tuple[tarfile.TarInfo, tuple[str, ...]]] = []
    normalized_member_paths: set[tuple[str, ...]] = set()
    roots: set[str] = set()
    for member in stream.getmembers():
        raw_name = member.name
        components = _validated_posix_member_components(
            raw_name,
            archive_kind="sdist",
            is_directory=member.isdir(),
        )
        if components in normalized_member_paths:
            raise RuntimeError(f"sdist contains duplicate sdist member: {PurePosixPath(*components).as_posix()}")
        normalized_member_paths.add(components)
        if not member.isfile() and not member.isdir():
            raise RuntimeError(f"sdist contains unsupported sdist member type: {raw_name!r}")
        roots.add(components[0])
        validated_members.append((member, components))

    if len(roots) != 1:
        raise RuntimeError(f"sdist must contain exactly one root directory, got {sorted(roots)}")

    files: dict[str, tarfile.TarInfo] = {}
    for member, components in validated_members:
        if len(components) == 1:
            if not member.isdir():
                raise RuntimeError(f"sdist top-level entry must be a directory: {member.name!r}")
            continue
        if member.isdir():
            continue
        relative_name = PurePosixPath(*components[1:]).as_posix()
        if relative_name in files:
            raise RuntimeError(f"sdist contains duplicate file member: {relative_name}")
        files[relative_name] = member
    return files


def _validated_wheel_file_members(stream: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    files: dict[str, zipfile.ZipInfo] = {}
    normalized_member_paths: set[tuple[str, ...]] = set()
    top_level_roots: set[str] = set()
    for member in stream.infolist():
        components = _validated_posix_member_components(
            member.filename,
            archive_kind="wheel",
            is_directory=member.is_dir(),
        )
        if components in normalized_member_paths:
            raise RuntimeError(f"wheel contains duplicate wheel member: {PurePosixPath(*components).as_posix()}")
        normalized_member_paths.add(components)

        unix_mode = (member.external_attr >> 16) & 0xFFFF
        member_type = stat.S_IFMT(unix_mode)
        allowed_types = {0, stat.S_IFDIR} if member.is_dir() else {0, stat.S_IFREG}
        if member_type not in allowed_types:
            raise RuntimeError(f"wheel contains unsupported wheel member type: {member.filename!r}")

        top_level_roots.add(components[0])
        if not member.is_dir():
            files[PurePosixPath(*components).as_posix()] = member

    dist_info_roots = {root for root in top_level_roots if root.endswith(".dist-info")}
    if len(dist_info_roots) != 1:
        raise RuntimeError(f"wheel must contain exactly one .dist-info root, got {sorted(dist_info_roots)}")
    expected_roots = {PurePosixPath(name).parts[0] for name in EXPECTED_WHEEL_MEMBERS}
    unexpected_roots = top_level_roots - expected_roots - dist_info_roots
    if unexpected_roots:
        raise RuntimeError(f"wheel contains unexpected top-level roots: {sorted(unexpected_roots)}")
    return files


def _validate_wheel(archive: Path) -> None:
    with zipfile.ZipFile(archive) as stream:
        archive_members = _validated_wheel_file_members(stream)
        members = set(archive_members)
        missing = sorted(EXPECTED_WHEEL_MEMBERS - members)
        if missing:
            raise RuntimeError(f"wheel is missing packaged modules: {missing}")
        if not any(name.endswith(".dist-info/METADATA") for name in members):
            raise RuntimeError("wheel is missing distribution metadata")
        if not any(
            name.endswith(".dist-info/LICENSE") or (".dist-info/licenses/" in name and name.endswith("LICENSE"))
            for name in members
        ):
            raise RuntimeError("wheel is missing MIT license metadata")

        mismatched_hashes = []
        for relative_name, expected_hash in EXPECTED_WHEEL_MEMBER_SHA256.items():
            actual_hash = hashlib.sha256(stream.read(archive_members[relative_name])).hexdigest()
            if actual_hash != expected_hash:
                mismatched_hashes.append(f"{relative_name}: expected {expected_hash}, got {actual_hash}")
    if mismatched_hashes:
        raise RuntimeError(f"wheel member SHA256 mismatch: {'; '.join(mismatched_hashes)}")


def _validate_sdist(archive: Path) -> None:
    with tarfile.open(archive, "r:gz") as stream:
        archive_members = _validated_sdist_file_members(stream)
        members = set(archive_members)
        missing = sorted(EXPECTED_SDIST_MEMBERS - members)
        if missing:
            raise RuntimeError(f"sdist is missing public release files: {missing}")
        forbidden = sorted(FORBIDDEN_SDIST_MEMBERS & members)
        forbidden.extend(sorted(name for name in members if name.startswith("docs/superpowers/")))
        forbidden.extend(sorted(name for name in members if name.startswith("tests/preparation/")))
        if forbidden:
            raise RuntimeError(f"sdist contains preparation-only files: {forbidden}")

        mismatched_hashes = []
        for relative_name, expected_hash in EXPECTED_SDIST_MEMBER_SHA256.items():
            member = archive_members[relative_name]
            extracted = stream.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"could not read sdist member: {relative_name}")
            actual_hash = hashlib.sha256(extracted.read()).hexdigest()
            if actual_hash != expected_hash:
                mismatched_hashes.append(f"{relative_name}: expected {expected_hash}, got {actual_hash}")
    if mismatched_hashes:
        raise RuntimeError(f"sdist member SHA256 mismatch: {'; '.join(mismatched_hashes)}")


def _venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts/python.exe"
    return venv_root / "bin/python"


def _python_purelib(python: Path, *, cwd: Path) -> Path:
    command = [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"]
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    location = completed.stdout.strip()
    if completed.returncode != 0 or not location:
        raise RuntimeError(
            "could not determine interpreter purelib "
            f"({python})\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    purelib = Path(location)
    if not purelib.is_absolute():
        raise RuntimeError(f"interpreter purelib must be absolute: {location!r}")
    return purelib.resolve()


def _smoke_installed_wheel(wheel: Path, work_root: Path) -> None:
    venv_root = work_root / "venv"
    _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_root)], cwd=work_root)
    python = _venv_python(venv_root)
    install_command = [str(python), "-m", "pip", "install", "--no-deps"]
    if sys.version_info < (3, 11):
        # The preparation host is older than the public contract. Metadata is checked independently;
        # this flag only permits the source-isolation smoke test to use its preinstalled PyTorch stack.
        install_command.append("--ignore-requires-python")
    install_command.append(str(wheel))
    _run(install_command, cwd=work_root)

    smoke_root = work_root / "outside-source"
    smoke_root.mkdir()
    nested_purelib = _python_purelib(python, cwd=work_root)
    if not nested_purelib.is_relative_to(venv_root.resolve()):
        raise RuntimeError(f"smoke interpreter purelib escaped its venv: {nested_purelib}")
    caller_purelib = Path(sysconfig.get_path("purelib")).resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(nested_purelib), str(caller_purelib)))
    environment.pop("PYTHONHOME", None)
    commands = (
        [str(python), "-m", "app.main", "--help"],
        [str(python), "-m", "tools.run_cvoi_manual_oracle", "--help"],
        [str(python), "-m", "tools.run_cvoi_direct_epdms", "--help"],
    )
    for command in commands:
        _run(command, cwd=smoke_root, environment=environment)

    import_check = """
import hashlib
from pathlib import Path
import sysconfig
import app.main as app_main
import app.vjepa_cowa_world_model as rise_app
import app.vjepa_cowa_world_model.training.cvoi_manual_navtrain_oracle as oracle
import app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots as formal_roots
import src.models.vision_transformer as vision_transformer
import tools.run_cvoi_direct_epdms as direct_cli
import tools.run_cvoi_manual_oracle as manual_cli

artifact_root = Path(sysconfig.get_path('purelib')).resolve()
for module in (app_main, manual_cli, direct_cli, rise_app, oracle, formal_roots, vision_transformer):
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(artifact_root):
        raise RuntimeError(f'{module.__name__} loaded outside installed artifact root: {module_path}')

scene_filter_hashes = {
    'configs/navsim/scene_filters/navtrain.yaml': 'c37fea567a0cfdbc29076cca893d4f5dd32db59baec18ae214527206d6b64e6f',
    'configs/navsim/scene_filters/navtest.yaml': '61284edf5003c0291f843ce9817c822ba306609a62d54544223adae3fc7fc9cd',
}
for relative_name, expected_hash in scene_filter_hashes.items():
    scene_filter = formal_roots.resolve_formal_v2_navsim_scene_filter_path(relative_name).resolve()
    if not scene_filter.is_relative_to(artifact_root):
        raise RuntimeError(f'scene filter resolved outside installed artifact root: {scene_filter}')
    if not scene_filter.is_file():
        raise RuntimeError(f'installed scene filter is missing: {scene_filter}')
    actual_hash = hashlib.sha256(scene_filter.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f'installed scene filter SHA256 mismatch for {relative_name}: expected {expected_hash}, got {actual_hash}'
        )
"""
    _run([str(python), "-c", import_check], cwd=smoke_root, environment=environment)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rise-package-check-") as temporary:
        work_root = Path(temporary)
        source_root = work_root / "source"
        distribution_root = work_root / "dist"
        _copy_source_tree(source_root)
        distribution_root.mkdir()
        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--outdir",
                str(distribution_root),
                str(source_root),
            ],
            cwd=work_root,
        )

        wheels = sorted(distribution_root.glob("*.whl"))
        sdists = sorted(distribution_root.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise RuntimeError(f"expected one wheel and one sdist, got wheels={wheels}, sdists={sdists}")
        _validate_wheel(wheels[0])
        print("wheel: ok")
        _validate_sdist(sdists[0])
        print("sdist: ok")
        _smoke_installed_wheel(wheels[0], work_root)
        print("installed artifact smoke checks: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
