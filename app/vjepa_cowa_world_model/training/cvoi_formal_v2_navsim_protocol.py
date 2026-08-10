"""NavSim constants shared by the manual CVoI training and evaluation chain."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from app.vjepa_cowa_world_model.training.cvoi_navsim_protocols import V2_PROTOCOL_ID, get_cvoi_navsim_metric_protocol

FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS: tuple[int, ...] = (10, 20, 30, 35, 40, 45, 50)
FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS: tuple[int, ...] = tuple(range(5, 81, 5))
FORMAL_V2_NAVSIM_E120_DEFAULT_LAMBDA = 0.005
FORMAL_V2_NAVSIM_E120_LAMBDA_GRID: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.05)
FORMAL_V2_NAVSIM_MAX_AGENTS = 1024
FORMAL_V2_NAVSIM_MAX_HORIZON = 4
FORMAL_V2_NAVSIM_HORIZONS: tuple[int, ...] = tuple(range(FORMAL_V2_NAVSIM_MAX_HORIZON + 1))
FORMAL_V2_NAVSIM_P0_POLICIES: Mapping[str, tuple[float, float, float, float, float]] = MappingProxyType(
    {
        "uniform": (0.2, 0.2, 0.2, 0.2, 0.2),
        "extremes": (0.5, 0.0, 0.0, 0.0, 0.5),
        "short_heavy": (0.225, 0.225, 0.225, 0.225, 0.1),
        "no_full": (0.25, 0.25, 0.25, 0.25, 0.0),
    }
)
FORMAL_V2_NAVSIM_METRIC_PROTOCOL_IDS: tuple[str, ...] = (V2_PROTOCOL_ID,)


def _formal_metric_protocol(
    protocol_id: str,
    *,
    score_family: str,
    devkit_checkout: str,
    devkit_repository_root: str,
    devkit_revision: str,
) -> Mapping[str, str]:
    metric = get_cvoi_navsim_metric_protocol(protocol_id)
    return MappingProxyType(
        {
            "protocol_id": metric.protocol_id,
            "score_family": score_family,
            "aggregate_key": metric.summary_token,
            "authority_script": metric.authority_script.as_posix(),
            "authority_runner": metric.scorer_entrypoint.rsplit("/", 1)[-1],
            "scorer_relative_path": metric.scorer_entrypoint,
            "devkit_checkout": devkit_checkout,
            "devkit_repository_root": devkit_repository_root,
            "devkit_revision": devkit_revision,
        }
    )


_FORMAL_V2_NAVSIM_METRIC_PROTOCOLS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        V2_PROTOCOL_ID: _formal_metric_protocol(
            V2_PROTOCOL_ID,
            score_family="epdms",
            devkit_checkout="/path/to/navsim-devkit",
            devkit_repository_root="/path/to/navsim-devkit",
            devkit_revision="937cefc1b116f930990abea1c54185308a96029f",
        ),
    }
)


def get_formal_v2_navsim_metric_protocol(protocol_id: str) -> dict[str, str]:
    """Return one fresh metric authority or reject an unknown protocol."""

    if type(protocol_id) is not str or protocol_id not in _FORMAL_V2_NAVSIM_METRIC_PROTOCOLS:
        raise ValueError(
            f"unknown Formal-v2 NavSim metric protocol {protocol_id!r}; "
            f"expected one of {FORMAL_V2_NAVSIM_METRIC_PROTOCOL_IDS}"
        )
    return dict(_FORMAL_V2_NAVSIM_METRIC_PROTOCOLS[protocol_id])
