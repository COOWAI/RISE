"""Failure-atomic publication helpers for PyTorch artifacts."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch


def atomic_publish_staging_no_overwrite(staging_path: str | Path, target_path: str | Path) -> Path:
    """Atomically link a complete sibling artifact without replacing an existing target."""

    staging = Path(staging_path)
    target = Path(target_path)
    if not staging.is_file():
        raise FileNotFoundError(f"staging artifact does not exist: {staging}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(staging, target)
    except FileExistsError as exc:
        raise FileExistsError(f"artifact appeared while publishing; staging retained at {staging}") from exc
    staging.unlink()
    return target


def atomic_torch_save_no_overwrite(payload: Any, path: str | Path) -> Path:
    """Save to a unique sibling first and publish only after serialization succeeds."""

    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.staging-{uuid4().hex}")
    try:
        torch.save(payload, staging)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    return atomic_publish_staging_no_overwrite(staging, target)


def _absolute_lexically_normalized_path(path: str | Path) -> Path:
    raw_path = os.fspath(path)
    normalized_path = os.path.normpath(raw_path)
    if not os.path.isabs(raw_path) or raw_path != normalized_path or raw_path.startswith("//"):
        raise ValueError(f"path must be absolute lexical-normalized: {raw_path!r}")
    return Path(raw_path)


def _open_existing_parent_directory(parent: Path) -> int:
    try:
        parent_stat = parent.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"parent directory does not exist: {parent}") from exc
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise NotADirectoryError(f"parent path is not a directory: {parent}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(parent, flags)
    except OSError as exc:
        raise NotADirectoryError(f"parent path must be a non-symlink directory: {parent}") from exc
    try:
        if not stat.S_ISDIR(os.fstat(parent_fd).st_mode):
            raise NotADirectoryError(f"parent path is not a directory: {parent}")
    except BaseException as exc:
        try:
            os.close(parent_fd)
        except BaseException as close_error:
            _add_cleanup_note(exc, close_error)
        raise
    return parent_fd


def _regular_target_stat(parent_fd: int, target_name: str, *, display_path: Path) -> os.stat_result | None:
    try:
        target_stat = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(target_stat.st_mode):
        raise ValueError(f"target must be absent or a regular file: {display_path}")
    return target_stat


def _target_revision(target_stat: os.stat_result | None) -> tuple[int, ...] | None:
    if target_stat is None:
        return None
    return (
        int(target_stat.st_dev),
        int(target_stat.st_ino),
        int(target_stat.st_size),
        int(target_stat.st_mtime_ns),
        int(target_stat.st_ctime_ns),
    )


def _create_staging_file(parent_fd: int, target_name: str) -> tuple[int, str, tuple[int, int]]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _attempt in range(100):
        staging_name = f".{target_name}.staging-{uuid4().hex}"
        try:
            staging_fd = os.open(staging_name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            staging_stat = os.fstat(staging_fd)
        except BaseException as exc:
            try:
                os.close(staging_fd)
            except BaseException as close_error:
                _add_cleanup_note(exc, close_error)
            try:
                os.unlink(staging_name, dir_fd=parent_fd)
            except BaseException as unlink_error:
                _add_cleanup_note(exc, unlink_error)
            raise
        return staging_fd, staging_name, (int(staging_stat.st_dev), int(staging_stat.st_ino))
    raise FileExistsError(f"could not allocate a unique staging file for {target_name!r}")


def _unlink_owned_staging(
    parent_fd: int,
    staging_name: str,
    staging_identity: tuple[int, int],
) -> BaseException | None:
    try:
        current_stat = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except BaseException as exc:
        return exc
    if (int(current_stat.st_dev), int(current_stat.st_ino)) != staging_identity:
        return RuntimeError(f"refusing to remove staging path no longer owned by this call: {staging_name!r}")
    try:
        os.unlink(staging_name, dir_fd=parent_fd)
    except BaseException as exc:
        return exc
    return None


def _add_cleanup_note(error: BaseException, cleanup_error: BaseException) -> None:
    note = f"additional cleanup failure: {type(cleanup_error).__name__}: {cleanup_error}"
    cleanup_errors = tuple(getattr(error, "atomic_cleanup_errors", ())) + (cleanup_error,)
    setattr(error, "atomic_cleanup_errors", cleanup_errors)
    if hasattr(error, "add_note"):
        error.add_note(note)


def atomic_torch_save_replace(payload: Any, path: str | Path) -> Path:
    """Failure-atomically replace one fixed PyTorch artifact.

    The caller must provide an absolute, lexically normalized path whose parent
    already exists. Existing targets must be regular files. The rename is the
    commit point: pre-commit failures preserve the previous
    target, while a post-commit directory-fsync failure can expose only the
    complete old or complete new artifact. The parent directory must be trusted:
    callers must serialize writes and must not mutate the target or the helper's
    unique staging entry outside this helper while publication is in progress.
    """

    target = _absolute_lexically_normalized_path(path)
    parent_fd = _open_existing_parent_directory(target.parent)
    staging_fd = -1
    staging_name = ""
    staging_identity = (-1, -1)
    staging_owned = False
    active_error: BaseException | None = None
    try:
        initial_revision = _target_revision(
            _regular_target_stat(parent_fd, target.name, display_path=target),
        )
        staging_fd, staging_name, staging_identity = _create_staging_file(parent_fd, target.name)
        staging_owned = True
        stream = os.fdopen(staging_fd, "wb")
        staging_fd = -1
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException as exc:
            try:
                stream.close()
            except BaseException as close_error:
                _add_cleanup_note(exc, close_error)
            raise
        stream.close()

        current_revision = _target_revision(
            _regular_target_stat(parent_fd, target.name, display_path=target),
        )
        if current_revision != initial_revision:
            raise RuntimeError(f"target changed while publishing: {target}")

        os.replace(
            staging_name,
            target.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        staging_owned = False
        os.fsync(parent_fd)
        return target
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        if staging_fd >= 0:
            try:
                os.close(staging_fd)
            except BaseException as close_error:
                if active_error is None:
                    raise
                _add_cleanup_note(active_error, close_error)
        if staging_owned:
            cleanup_error = _unlink_owned_staging(parent_fd, staging_name, staging_identity)
            if cleanup_error is not None:
                if active_error is None:
                    raise cleanup_error
                _add_cleanup_note(active_error, cleanup_error)
        try:
            os.close(parent_fd)
        except BaseException as close_error:
            if active_error is None:
                raise
            _add_cleanup_note(active_error, close_error)
