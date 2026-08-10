"""Formal-v2 NavSim counterfactual cohorts and atomic factual pairs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from app.vjepa_cowa_world_model.training import navsim_data


class _IdentityDataset(Dataset):
    def __init__(self, rows: list[tuple[str, int, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def cvoi_pair_key(self, index: int) -> tuple[str, int]:
        scene_name, window_start_pos, _ = self.rows[index]
        match = navsim_data._COUNTERFACTUAL_SCENE_RE.fullmatch(scene_name)
        if match is not None:
            return match.group("base"), int(match.group("start"))
        return scene_name, window_start_pos

    def cvoi_hazard_label(self, index: int) -> tuple[bool, str]:
        _, _, hazard_type = self.rows[index]
        return bool(hazard_type), hazard_type

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_name, window_start_pos, hazard_type = self.rows[index]
        is_counterfactual = "_cf_" in scene_name
        return {
            "scene_name": scene_name,
            "window_start_pos": window_start_pos,
            "cf_annotation_valid": is_counterfactual,
            "cf_is_hazard": is_counterfactual,
            "cf_hazard_type": hazard_type,
            "future_agent_geometry_valid": not is_counterfactual,
            "agent_boxes": np.zeros((1, 256, 7), dtype=np.float32),
            "agent_mask": np.zeros((1, 256), dtype=np.bool_),
            "bev_segmentation": np.zeros((1, 200, 200), dtype=np.uint8),
            "raw_agent_count": None if is_counterfactual else np.zeros(1, dtype=np.int64),
        }


def _tagged(domain: str, rows: list[tuple[str, int, str]]) -> navsim_data.RootTaggedDataset:
    return navsim_data.RootTaggedDataset(
        _IdentityDataset(rows),
        domain=domain,
        dataset_root_name=domain,
        dataset_root_index=0 if domain == "real" else 1,
        future_agent_geometry_valid=domain == "real",
        load_agent_annotations=domain == "real",
        annotation_selection=("all_valid" if domain == "real" else "trajectory_match_and_accident_type_allowlist"),
    )


def _pair_dataset() -> navsim_data.MatchedRealCounterfactualPairDataset:
    real = _tagged(
        "real",
        [
            ("scene-a", 10, ""),
            ("scene-b", 20, ""),
            ("scene-c", 30, ""),
            ("scene-d", 40, ""),
        ],
    )
    counterfactual = _tagged(
        "counterfactual",
        [
            ("scene-a_cf_000010_000019", 0, "自车行为引起"),
            ("scene-b_cf_000020_000029", 0, "非自车行为引起"),
            ("scene-c_cf_000030_000039", 0, "自车行为引起"),
            ("scene-d_cf_000040_000049", 0, "非自车行为引起"),
        ],
    )
    return navsim_data.MatchedRealCounterfactualPairDataset([real, counterfactual])


def test_matched_pair_dataset_uses_source_window_identity_and_never_relabels_non_ego_hazard() -> None:
    dataset = _pair_dataset()

    pair = dataset[1]

    assert pair.pair_key == ("scene-b", 20)
    assert pair.real["dataset_domain"] == "real"
    assert pair.counterfactual["dataset_domain"] == "counterfactual"
    assert pair.real["window_start_pos"] == pair.counterfactual["window_start_pos"] == 20
    assert pair.counterfactual["cf_is_hazard"] is True
    assert pair.counterfactual["cf_hazard_type"] == "非自车行为引起"


def test_distributed_sampler_shuffles_pair_units_without_splitting_them_across_ranks() -> None:
    dataset = _pair_dataset()
    seen: set[tuple[str, int]] = set()

    for rank in range(2):
        sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=True, seed=37, drop_last=True)
        sampler.set_epoch(5)
        rank_keys = []
        for pair_index in sampler:
            pair = dataset[pair_index]
            assert pair.real["base_scene_id"] == pair.counterfactual["base_scene_id"]
            assert pair.real["window_start_pos"] == pair.counterfactual["window_start_pos"]
            rank_keys.append(pair.pair_key)
        assert not seen.intersection(rank_keys)
        seen.update(rank_keys)

    assert seen == {("scene-a", 10), ("scene-b", 20), ("scene-c", 30), ("scene-d", 40)}


def test_matched_pair_dataset_fails_fast_for_missing_or_ambiguous_factual_key() -> None:
    counterfactual = _tagged(
        "counterfactual",
        [("scene-a_cf_000010_000019", 0, "自车行为引起")],
    )
    missing_real = _tagged("real", [("scene-a", 11, "")])
    duplicate_real = _tagged("real", [("scene-a", 10, ""), ("scene-a", 10, "")])

    with pytest.raises(ValueError, match="missing factual.*scene-a.*10"):
        navsim_data.MatchedRealCounterfactualPairDataset([missing_real, counterfactual])
    with pytest.raises(ValueError, match="duplicate factual.*scene-a.*10"):
        navsim_data.MatchedRealCounterfactualPairDataset([duplicate_real, counterfactual])


def test_pair_dataset_rejects_disallowed_or_safe_counterfactual_rows() -> None:
    real = _tagged("real", [("scene-a", 10, "")])
    for hazard_type in ("有事故但与自车无关", ""):
        counterfactual = _tagged(
            "counterfactual",
            [("scene-a_cf_000010_000019", 0, hazard_type)],
        )
        with pytest.raises(ValueError, match="counterfactual.*hazard"):
            navsim_data.MatchedRealCounterfactualPairDataset([real, counterfactual])
