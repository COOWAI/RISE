from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools import run_cvoi_manual_oracle as cli


def _poison(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("side effect must not run")


@pytest.mark.parametrize(
    ("arguments", "handler_name", "require_cuda"),
    [
        (["build-manifest", "--results-root", "/results"], "build_manual_navtrain_manifest", False),
        (
            [
                "score",
                "--results-root",
                "/results",
                "--horizon",
                "3",
                "--source-config",
                "/configs/no_cf_05_p1.yaml",
            ],
            "score_manual_navtrain_horizon",
            True,
        ),
    ],
)
def test_main_dispatches_one_environment_bound_action(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    handler_name: str,
    require_cuda: bool,
) -> None:
    environment = object()
    loader_calls: list[bool] = []
    handler_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_loader(_environ: object, *, require_cuda: bool) -> object:
        loader_calls.append(require_cuda)
        return environment

    monkeypatch.setattr(cli, "load_manual_oracle_environment", fake_loader)
    for candidate in (
        "build_manual_navtrain_manifest",
        "score_manual_navtrain_horizon",
        "aggregate_manual_navtrain_oracle",
    ):
        monkeypatch.setattr(
            cli,
            candidate,
            (
                (lambda *values, **options: handler_calls.append((values, options)))
                if candidate == handler_name
                else _poison
            ),
        )

    assert cli.main(arguments) == 0
    assert loader_calls == [require_cuda]
    expected = (
        (
            (Path("/results"), environment, 3),
            {"source_config_path": Path("/configs/no_cf_05_p1.yaml")},
        )
        if handler_name == "score_manual_navtrain_horizon"
        else ((Path("/results"), environment), {})
    )
    assert handler_calls == [expected]


def test_main_aggregate_loads_no_environment_and_calls_only_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(cli, "load_manual_oracle_environment", _poison)
    monkeypatch.setattr(cli, "build_manual_navtrain_manifest", _poison)
    monkeypatch.setattr(cli, "score_manual_navtrain_horizon", _poison)
    monkeypatch.setattr(
        cli,
        "aggregate_manual_navtrain_oracle",
        lambda path, *, source_config_path: calls.append((path, source_config_path)),
    )

    assert (
        cli.main(
            [
                "aggregate",
                "--results-root",
                "/results",
                "--source-config",
                "/configs/no_cf_05_p1.yaml",
            ]
        )
        == 0
    )
    assert calls == [(Path("/results"), Path("/configs/no_cf_05_p1.yaml"))]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["all", "--results-root", "/results"],
        ["run", "--results-root", "/results"],
        ["score", "--results-root", "/results", "--horizon", "3"],
        ["aggregate", "--results-root", "/results"],
        [
            "score",
            "--results-root",
            "/results",
            "--horizon",
            "5",
            "--source-config",
            "/configs/full_05_p1.yaml",
        ],
        [
            "score",
            "--results-root",
            "/results",
            "--horizon",
            "-1",
            "--source-config",
            "/configs/full_05_p1.yaml",
        ],
    ],
)
def test_invalid_cli_is_rejected_before_environment_or_handlers(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(cli, "load_manual_oracle_environment", _poison)
    monkeypatch.setattr(cli, "build_manual_navtrain_manifest", _poison)
    monkeypatch.setattr(cli, "score_manual_navtrain_horizon", _poison)
    monkeypatch.setattr(cli, "aggregate_manual_navtrain_oracle", _poison)

    with pytest.raises(SystemExit):
        cli.main(arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["build-manifest", "--help"],
        ["score", "--help"],
        ["aggregate", "--help"],
    ],
)
def test_help_has_no_environment_filesystem_or_handler_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(cli, "load_manual_oracle_environment", _poison)
    monkeypatch.setattr(cli, "build_manual_navtrain_manifest", _poison)
    monkeypatch.setattr(cli, "score_manual_navtrain_horizon", _poison)
    monkeypatch.setattr(cli, "aggregate_manual_navtrain_oracle", _poison)

    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)

    assert raised.value.code == 0


def test_parser_exposes_exactly_three_commands() -> None:
    parser = cli.build_parser()
    subparser_action = next(action for action in parser._actions if getattr(action, "choices", None))

    assert set(subparser_action.choices) == {"build-manifest", "score", "aggregate"}


def test_help_does_not_import_the_oracle_data_plane() -> None:
    tool_path = Path(cli.__file__).resolve()
    guarded_module = "app.vjepa_cowa_world_model.training.cvoi_manual_navtrain_oracle"
    program = f"""
import builtins
import runpy
import sys

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == {guarded_module!r}:
        raise AssertionError("help imported the Oracle data plane")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.argv = [{str(tool_path)!r}, "--help"]
try:
    runpy.run_path({str(tool_path)!r}, run_name="__main__")
except SystemExit as error:
    if error.code != 0:
        raise
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
