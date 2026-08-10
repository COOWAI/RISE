"""Tests for the strict score-field parsers retained by direct NavTest EPDMS."""

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from app.vjepa_cowa_world_model.training.cvoi_navsim_scores import _csv_components, _csv_number

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "vjepa_cowa_world_model" / "training" / "cvoi_navsim_scores.py"
)
V2_COMPONENTS = frozenset(
    {
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "traffic_light_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "lane_keeping",
        "history_comfort",
        "two_frame_extended_comfort",
    }
)


def _protocol():
    return SimpleNamespace(
        required_components=V2_COMPONENTS,
        nullable_components=frozenset({"two_frame_extended_comfort"}),
    )


def _component_row(**overrides: str) -> dict[str, str]:
    row = {component: "1.0" for component in V2_COMPONENTS}
    row.update(overrides)
    return row


def test_module_retains_only_direct_epdms_score_field_parsing_surface() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    for removed_surface in (
        "CvoiOfficialEffectivePolicy",
        "CvoiNavSimScenarioExecutionBinding",
        "CvoiNavSimScenarioManifestBundle",
        "CvoiNavSimImportedScores",
        "CvoiNavSimRunSummary",
        "CvoiNavSimScenarioScore",
        "CVOI_NAVSIM_PDMS_V1_SCENARIO_SCORE_SCHEMA",
        "V1_PROTOCOL_ID",
        "get_cvoi_navsim_metric_protocol",
        "import_cvoi_navsim_scores",
    ):
        assert removed_surface not in source


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0.0),
        (" 0.75 ", 0.75),
        ("1", 1.0),
        ("-3.5", -3.5),
    ],
)
def test_csv_number_accepts_finite_numeric_text(raw: str, expected: float) -> None:
    assert _csv_number(raw, field="score") == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "   ", "not-a-number"])
def test_csv_number_rejects_blank_or_non_numeric_text(raw: str) -> None:
    with pytest.raises(ValueError, match="blank|numeric"):
        _csv_number(raw, field="score")


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "+inf", "-inf"])
def test_csv_number_rejects_non_finite_values(raw: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        _csv_number(raw, field="score")


@pytest.mark.parametrize("raw", ["-0.01", "1.01"])
def test_csv_number_rejects_values_outside_unit_score_range(raw: str) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _csv_number(raw, field="score", unit_score=True)


def test_csv_components_returns_an_immutable_complete_v2_projection() -> None:
    components = _csv_components(
        _component_row(ego_progress="0.25", two_frame_extended_comfort="0.5"),
        protocol=_protocol(),
        token="navtest-a",
    )

    assert isinstance(components, MappingProxyType)
    assert set(components) == V2_COMPONENTS
    assert components["ego_progress"] == pytest.approx(0.25)
    assert components["two_frame_extended_comfort"] == pytest.approx(0.5)
    with pytest.raises(TypeError):
        components["ego_progress"] = 0.0


@pytest.mark.parametrize("raw", ["", " ", "nan", "NaN"])
def test_csv_components_maps_only_the_declared_nullable_component_to_none(raw: str) -> None:
    components = _csv_components(
        _component_row(two_frame_extended_comfort=raw),
        protocol=_protocol(),
        token="navtest-a",
    )

    assert components["two_frame_extended_comfort"] is None


@pytest.mark.parametrize("raw", ["", "nan", "NaN", "inf", "-inf"])
def test_csv_components_rejects_non_finite_required_components(raw: str) -> None:
    with pytest.raises(ValueError, match="history_comfort.*finite|non-finite component.*history_comfort"):
        _csv_components(
            _component_row(history_comfort=raw),
            protocol=_protocol(),
            token="navtest-a",
        )


@pytest.mark.parametrize("raw", ["inf", "+inf", "-inf"])
def test_csv_components_rejects_infinity_even_for_nullable_components(raw: str) -> None:
    with pytest.raises(ValueError, match="two_frame_extended_comfort.*finite"):
        _csv_components(
            _component_row(two_frame_extended_comfort=raw),
            protocol=_protocol(),
            token="navtest-a",
        )


def test_csv_components_fails_loudly_when_a_required_column_is_missing() -> None:
    row = _component_row()
    del row["history_comfort"]

    with pytest.raises(KeyError, match="history_comfort"):
        _csv_components(row, protocol=_protocol(), token="navtest-a")
