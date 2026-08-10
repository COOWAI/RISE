#!/usr/bin/env python3
"""Export a deterministic RISE public tree from an explicit Git commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

if __package__:
    from tools.check_public_surface import PublicSurfaceError, require_public_surface
else:  # pragma: no cover - exercised by CLI help and release commands
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.check_public_surface import PublicSurfaceError, require_public_surface

_EXCLUDED_PATHS = frozenset(
    {
        "AGENTS.md",
        "MINIMAL_FILES.txt",
        "tests/preparation/test_agent_guidance.py",
        "tests/preparation/test_diffusion_contract_recorder.py",
    }
)
_EXCLUDED_PREFIXES = ("docs/superpowers/",)
_ALLOWED_GIT_MODES = frozenset({"100644", "100755"})


@dataclass(frozen=True)
class ExportFacts:
    """Stable facts that bind a public export to one source commit."""

    source_commit: str
    public_tree_sha256: str
    file_count: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "file_count": self.file_count,
            "public_tree_sha256": self.public_tree_sha256,
            "source_commit": self.source_commit,
        }


@dataclass(frozen=True)
class _TreeEntry:
    path: str
    mode: str
    content: bytes


def _run_git(repository: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PublicSurfaceError(f"Git object read failed: {detail or 'unknown Git error'}")
    return completed.stdout


def _resolve_commit(repository: Path, source_ref: str) -> str:
    if not source_ref or source_ref.isspace():
        raise PublicSurfaceError("source ref must be an explicit nonempty Git ref")
    revision = f"{source_ref}^{{commit}}"
    output = _run_git(repository, ("rev-parse", "--verify", "--end-of-options", revision))
    commit = output.decode("ascii", errors="strict").strip()
    if not re_full_hex_commit(commit):
        raise PublicSurfaceError("Git did not resolve the source ref to one full commit ID")
    return commit


def re_full_hex_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdefABCDEF" for character in value)


def _validate_git_path(path: str) -> None:
    pure = PurePosixPath(path)
    if not path or pure.is_absolute() or ".." in pure.parts or path != pure.as_posix():
        raise PublicSurfaceError(f"unsafe Git tree path: {path!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise PublicSurfaceError("Git tree contains a path with control characters")


def _excluded_from_public_tree(path: str) -> bool:
    return path in _EXCLUDED_PATHS or path.startswith(_EXCLUDED_PREFIXES)


def _read_commit_entries(repository: Path, commit: str) -> tuple[_TreeEntry, ...]:
    listing = _run_git(repository, ("ls-tree", "-rz", "--full-tree", commit))
    entries: list[_TreeEntry] = []
    for record in listing.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise PublicSurfaceError("Git tree contains an unsupported path or metadata record") from error
        _validate_git_path(path)
        if _excluded_from_public_tree(path):
            continue
        if mode == "120000":
            raise PublicSurfaceError(f"symbolic link is not allowed in public export: {path}")
        if object_type != "blob" or mode not in _ALLOWED_GIT_MODES:
            raise PublicSurfaceError(
                f"unsupported Git tree entry for public export: {path} (mode={mode}, type={object_type})"
            )
        content = _run_git(repository, ("cat-file", "blob", object_id))
        entries.append(_TreeEntry(path=path, mode=mode, content=content))
    return tuple(sorted(entries, key=lambda entry: entry.path))


def _public_tree_digest(entries: Sequence[_TreeEntry]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        path_bytes = entry.path.encode("utf-8")
        mode_bytes = entry.mode.encode("ascii")
        digest.update(len(path_bytes).to_bytes(8, byteorder="big"))
        digest.update(path_bytes)
        digest.update(len(mode_bytes).to_bytes(8, byteorder="big"))
        digest.update(mode_bytes)
        digest.update(len(entry.content).to_bytes(8, byteorder="big"))
        digest.update(entry.content)
    return digest.hexdigest()


def _validate_destination(destination: Path) -> None:
    if destination.is_symlink():
        raise PublicSurfaceError(f"destination must not be a symbolic link: {destination}")
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise PublicSurfaceError(f"destination must be absent or empty: {destination}")
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise PublicSurfaceError(f"destination parent must be an existing real directory: {parent}")


def _canonical_output_path(value: Path | str, *, field: str) -> Path:
    absolute = Path(value).absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise PublicSurfaceError(f"{field} must not contain symbolic-link components: {absolute}")
    try:
        return absolute.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise PublicSurfaceError(f"cannot resolve {field}: {absolute}") from error


def _validate_new_report(report: Path, destination: Path) -> None:
    if report.exists() or report.is_symlink():
        raise PublicSurfaceError(f"report already exists: {report}")
    if report == destination or destination in report.parents:
        raise PublicSurfaceError("report must be outside the exported destination")
    if not report.parent.is_dir() or report.parent.is_symlink():
        raise PublicSurfaceError(f"report parent must be an existing real directory: {report.parent}")


def _materialize(entries: Sequence[_TreeEntry], destination: Path) -> None:
    destination.mkdir()
    for entry in entries:
        output_path = destination / entry.path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(entry.content)
        output_path.chmod(0o755 if entry.mode == "100755" else 0o644)


def _load_expected_report(report: Path) -> dict[str, object]:
    if not report.is_file() or report.is_symlink():
        raise PublicSurfaceError(f"expected report must be an existing regular file: {report}")
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicSurfaceError(f"cannot read expected report: {report}") from error
    if not isinstance(payload, dict):
        raise PublicSurfaceError(f"expected report must contain a JSON object: {report}")
    return payload


def _require_expected_facts(expected: dict[str, object], facts: ExportFacts) -> None:
    if (
        expected.get("source_commit") != facts.source_commit
        or expected.get("public_tree_sha256") != facts.public_tree_sha256
    ):
        raise PublicSurfaceError("resolved commit or public-tree digest does not match expected report")


def _write_new_report(report: Path, facts: ExportFacts) -> None:
    try:
        with report.open("x", encoding="utf-8") as stream:
            json.dump(facts.as_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise PublicSurfaceError(f"report already exists: {report}") from error
    except OSError as error:
        raise PublicSurfaceError(f"cannot write report: {report}") from error


def _publish_staged_tree(staging: Path, destination: Path) -> None:
    _validate_destination(destination)
    if destination.exists():
        for child in staging.iterdir():
            target = destination / child.name
            if child.is_dir():
                shutil.copytree(child, target, copy_function=shutil.copy2)
            else:
                shutil.copy2(child, target)
        return
    staging.rename(destination)


def export_public_tree(
    *,
    repository: Path | str,
    source_ref: str,
    destination: Path | str,
    report: Path | str | None = None,
    expect_report: Path | str | None = None,
) -> ExportFacts:
    """Export one validated commit tree while excluding preparation-only records."""

    repository_path = Path(repository).resolve()
    if not repository_path.is_dir():
        raise PublicSurfaceError(f"source repository is not a directory: {repository_path}")
    destination_path = _canonical_output_path(destination, field="destination")
    report_path = _canonical_output_path(report, field="report") if report is not None else None
    expected_report_path = (
        _canonical_output_path(expect_report, field="expected report") if expect_report is not None else None
    )
    if report_path is not None and expected_report_path is not None:
        raise PublicSurfaceError("--report and --expect-report are mutually exclusive")
    _validate_destination(destination_path)
    if report_path is not None:
        _validate_new_report(report_path, destination_path)

    commit = _resolve_commit(repository_path, source_ref)
    entries = _read_commit_entries(repository_path, commit)
    facts = ExportFacts(
        source_commit=commit,
        public_tree_sha256=_public_tree_digest(entries),
        file_count=len(entries),
    )
    if expected_report_path is not None:
        _require_expected_facts(_load_expected_report(expected_report_path), facts)

    with tempfile.TemporaryDirectory(prefix=".rise-public-export-", dir=destination_path.parent) as temporary:
        staging = Path(temporary) / "tree"
        _materialize(entries, staging)
        require_public_surface(staging)
        _publish_staged_tree(staging, destination_path)

    if report_path is not None:
        _write_new_report(report_path, facts)
    return facts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a validated public tree from an explicit Git commit.")
    parser.add_argument("--source-ref", required=True, help="Git ref that must resolve to a commit")
    parser.add_argument("--destination", required=True, type=Path, help="absent or empty export destination")
    reports = parser.add_mutually_exclusive_group()
    reports.add_argument("--report", type=Path, help="write export facts to a new JSON file")
    reports.add_argument("--expect-report", type=Path, help="require facts to match an existing JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        facts = export_public_tree(
            repository=Path.cwd(),
            source_ref=arguments.source_ref,
            destination=arguments.destination,
            report=arguments.report,
            expect_report=arguments.expect_report,
        )
    except PublicSurfaceError as error:
        print(f"public export failed: {error}", file=sys.stderr)
        return 1
    print(f"source commit: {facts.source_commit}")
    print(f"public tree sha256: {facts.public_tree_sha256}")
    print(f"destination: {arguments.destination.absolute()}")
    print(f"file count: {facts.file_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
