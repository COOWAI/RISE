#!/usr/bin/env python3
"""Validate a candidate RISE public tree without consulting ignore files."""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

MAX_PUBLIC_FILE_BYTES = 10 * 1024 * 1024

FULL_CONFIG_NAMES = (
    "01_predictor_lewm_pure.yaml",
    "02_p0_uniform.yaml",
    "03_field_full.yaml",
    "04_calibration_full.yaml",
    "05_p1_full.yaml",
    "06_stop_full.yaml",
    "07_gate_full.yaml",
)
PUBLIC_DIFFUSION_PATHS = frozenset(
    {
        "app/vjepa_cowa_world_model/models/diffusion_planner.py",
        "app/vjepa_cowa_world_model/diffusion_utils/__init__.py",
        "app/vjepa_cowa_world_model/diffusion_utils/sde.py",
        "app/vjepa_cowa_world_model/diffusion_utils/sampling.py",
    }
)
PUBLIC_DIFFUSION_MARKER = "# RISE provenance: independent-diffusion-v1"
REQUIRED_PUBLIC_PATHS = frozenset(
    {
        "README.md",
        "README_zh-CN.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "docs/configuration.md",
        "docs/reproduction.md",
        "docs/reproduction_zh-CN.md",
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
        *PUBLIC_DIFFUSION_PATHS,
        *(f"configs/train/navsim/cvoi_manual_full/{name}" for name in FULL_CONFIG_NAMES),
    }
)

_SKIPPED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
    }
)
_PREPARATION_ONLY_PATHS = frozenset(
    {
        "AGENTS.md",
        "MINIMAL_FILES.txt",
        "tests/preparation/test_agent_guidance.py",
        "tests/preparation/test_diffusion_contract_recorder.py",
    }
)
_PREPARATION_ONLY_PREFIXES = ("docs/superpowers/",)
_FORBIDDEN_EXACT_PATHS = frozenset({".gitlab-ci.yml"})
_FORBIDDEN_PATH_PREFIXES = (
    ".agents/",
    ".codex/",
    ".github/workflows/",
    "assets/",
    "configs/generated/",
    "ddddetection_torchcv/",
    "scripts/auto_a100/",
    "scripts/internal/",
    "scripts/remote/",
)
_FORBIDDEN_ENTRY_NAME = re.compile(
    r"(?:^|[_-])(?:(?:cvoi[_-])?(?:dag|orchestrat(?:or|ion)|automat(?:e|ed|ic|ion))|"
    r"(?:cvoi|auto)[_-]schedul(?:e|er))(?:[_-]|$)",
    flags=re.IGNORECASE,
)
_INTERNAL_OPERATION_NAME = re.compile(r"(?:^|[_-])internal[_-](?:operation|ops)", flags=re.IGNORECASE)
_PROHIBITED_FILE_ENDINGS = (
    ".arrow",
    ".bin",
    ".ckpt",
    ".db",
    ".h5",
    ".hdf5",
    ".mdb",
    ".npy",
    ".npz",
    ".onnx",
    ".p12",
    ".parquet",
    ".pem",
    ".pfx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tar.gz",
)
_KEY_LIKE_BASENAMES = frozenset({"id_ed25519", "id_ecdsa", "id_rsa"})
_IPV4_CANDIDATE = re.compile(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])")
_PRIVATE_IPV4_NETWORKS = (
    ipaddress.ip_network("10" + ".0.0.0/8"),
    ipaddress.ip_network("172." + "16.0.0/12"),
    ipaddress.ip_network("192." + "168.0.0/16"),
)
_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_-]*)\s*(?::[^=]+)?=\s*(.*?)\s*$")
_YAML_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*?)\s*$")
_SENSITIVE_NAME_PARTS = (
    "pass" + "word",
    "pass" + "wd",
    "sec" + "ret",
    "api_" + "key",
    "api" + "key",
    "access_" + "key",
    "auth_" + "token",
    "github_" + "token",
    "private_" + "key",
)
_PRIVATE_TEXT_MARKERS = (
    "/" + "disk/",
    "/home/" + "gpualloc",
    "/home/" + "hecheng" + "hao",
    "project-" + "world-model",
    "hch_" + "workspace",
    "gyd_" + "workspace",
    "qlr_" + "workspace",
    "fair" + "internal",
    "cowa" + "robot" + ".cn",
    "harbor." + "cowa" + "robot",
    "cowa" + "@",
    "he" + "chenghao",
    "chenghao" + ".he",
)
_KEY_HEADER_PATTERN = re.compile(re.escape("-----BEGIN ") + r"(?:[A-Z0-9]+ )*" + re.escape("PRIVATE KEY-----"))
_XTR_PROVENANCE_MARKER = re.compile(
    r"(?:(?<![A-Z0-9])XTR(?![A-Z0-9])|PRIVATE[-_ ]?XTR|XTR[-_ ]?DERIVED)",
    flags=re.IGNORECASE,
)


class PublicSurfaceError(RuntimeError):
    """Raised when a candidate tree cannot be safely scanned or exported."""


@dataclass(frozen=True, order=True)
class PublicSurfaceViolation:
    """One path-local public-release policy violation."""

    path: str
    rule: str

    def __str__(self) -> str:
        return f"{self.path}: {self.rule}"


def _is_skipped_directory(name: str) -> bool:
    return name in _SKIPPED_DIRECTORY_NAMES or name.endswith(".egg-info")


def _path_rule(relative_path: str) -> str | None:
    if relative_path in _PREPARATION_ONLY_PATHS or relative_path.startswith(_PREPARATION_ONLY_PREFIXES):
        return "preparation-only path"
    if relative_path in _FORBIDDEN_EXACT_PATHS or relative_path.startswith(_FORBIDDEN_PATH_PREFIXES):
        return "forbidden repository surface"
    basename = relative_path.rstrip("/").rsplit("/", 1)[-1]
    entry_stem = basename.rsplit(".", 1)[0]
    if _FORBIDDEN_ENTRY_NAME.search(entry_stem) or _INTERNAL_OPERATION_NAME.search(entry_stem):
        return "forbidden repository surface"
    lowered = basename.lower()
    if lowered in _KEY_LIKE_BASENAMES or lowered == ".env" or lowered.endswith(_PROHIBITED_FILE_ENDINGS):
        return "prohibited release artifact"
    return None


def _has_private_ip(text: str) -> bool:
    for candidate in _IPV4_CANDIDATE.findall(text):
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if any(address in network for network in _PRIVATE_IPV4_NETWORKS):
            return True
    return False


def _has_credential_assignment(text: str) -> bool:
    for line in text.splitlines():
        assignment = _ASSIGNMENT.match(line) or _YAML_ASSIGNMENT.match(line)
        if assignment is None:
            continue
        name, raw_value = assignment.groups()
        normalized_name = name.lower().replace("-", "_")
        if not any(marker in normalized_name for marker in _SENSITIVE_NAME_PARTS):
            continue
        value = raw_value.strip().rstrip(",").strip()
        unquoted = value.strip("\"'").strip()
        lowered = unquoted.lower()
        if not unquoted:
            continue
        if lowered in {"none", "null", "false", "true", "redacted", "changeme"}:
            continue
        if unquoted.startswith(("/" + "path/to/", "${", "$", "<")):
            continue
        if lowered.startswith(("os.environ", "getenv(", "field(")):
            continue
        return True
    return False


def _content_rules(content: bytes) -> Iterable[str]:
    text = content.decode("utf-8", errors="ignore")
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in _PRIVATE_TEXT_MARKERS) or _has_private_ip(text):
        yield "private deployment marker"
    if _has_credential_assignment(text):
        yield "credential-like assignment"
    if _KEY_HEADER_PATTERN.search(text):
        yield "private-key header"


def _has_verified_diffusion_provenance(content: bytes) -> bool:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return PUBLIC_DIFFUSION_MARKER in text.splitlines() and _XTR_PROVENANCE_MARKER.search(text) is None


def _scan_regular_file(root: Path, path: Path, relative_path: str) -> Iterable[PublicSurfaceViolation]:
    path_rule = _path_rule(relative_path)
    if path_rule is not None:
        yield PublicSurfaceViolation(relative_path, path_rule)

    metadata = path.stat(follow_symlinks=False)
    if metadata.st_size > MAX_PUBLIC_FILE_BYTES:
        yield PublicSurfaceViolation(relative_path, "oversized file")
        return
    content = path.read_bytes()
    if relative_path in PUBLIC_DIFFUSION_PATHS and not _has_verified_diffusion_provenance(content):
        yield PublicSurfaceViolation(relative_path, "unverified diffusion provenance")
    for rule in _content_rules(content):
        yield PublicSurfaceViolation(relative_path, rule)


def _walk_candidate(root: Path) -> Iterable[PublicSurfaceViolation]:
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name, reverse=True)
        except OSError as error:
            relative = directory.relative_to(root).as_posix() or "."
            raise PublicSurfaceError(f"cannot scan {relative}: {error}") from error
        for entry in entries:
            path = Path(entry.path)
            relative_path = path.relative_to(root).as_posix()
            try:
                metadata = path.lstat()
            except OSError as error:
                raise PublicSurfaceError(f"cannot inspect {relative_path}: {error}") from error
            if stat.S_ISLNK(metadata.st_mode):
                yield PublicSurfaceViolation(relative_path, "symbolic link")
            elif stat.S_ISDIR(metadata.st_mode):
                if not _is_skipped_directory(entry.name):
                    path_rule = _path_rule(relative_path + "/")
                    if path_rule is not None:
                        yield PublicSurfaceViolation(relative_path, path_rule)
                    pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                if entry.name == ".git":
                    continue
                yield from _scan_regular_file(root, path, relative_path)
            else:
                yield PublicSurfaceViolation(relative_path, "non-regular filesystem entry")


def scan_public_surface(root: Path | str) -> tuple[PublicSurfaceViolation, ...]:
    """Return sorted violations for ``root`` without reading ``.gitignore``."""

    candidate_root = Path(root).resolve()
    if not candidate_root.is_dir():
        raise PublicSurfaceError(f"public surface root is not a directory: {candidate_root}")
    violations = set(_walk_candidate(candidate_root))
    for relative_path in REQUIRED_PUBLIC_PATHS:
        required = candidate_root / relative_path
        if not required.is_file() or required.is_symlink():
            violations.add(PublicSurfaceViolation(relative_path, "required public path is missing"))
    return tuple(sorted(violations))


def require_public_surface(root: Path | str) -> None:
    """Raise a redacted error if ``root`` violates the public boundary."""

    violations = scan_public_surface(root)
    if violations:
        rendered = "\n".join(str(violation) for violation in violations)
        raise PublicSurfaceError(f"public surface check failed:\n{rendered}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan a RISE public-tree candidate without changing it.")
    parser.add_argument("root", type=Path, help="candidate public tree to scan")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        violations = scan_public_surface(arguments.root)
    except PublicSurfaceError as error:
        print(f"public surface check failed: {error}", file=sys.stderr)
        return 2
    if violations:
        for violation in violations:
            print(violation)
        return 1
    print(f"public surface: ok ({arguments.root.resolve()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
