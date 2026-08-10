import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training import cvoi_value
from app.vjepa_cowa_world_model.training.cvoi_manual_lineage import (
    build_cvoi_manual_value_parents,
    resolve_cvoi_manual_value_lineage_by_checkpoint_branch,
)
from app.vjepa_cowa_world_model.training.cvoi_value_training import train_cvoi_value_epoch, validate_cvoi_field_epoch


class TestCVoIValueTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = PrefixDualValueModel(embed_dim=4, hidden_dim=6)
        self.device = torch.device("cpu")

    def test_legacy_protocol_fails_before_reading_training_or_validation_batches(self):
        class ForbiddenIterable:
            def __iter__(self):
                raise AssertionError("legacy protocol must fail before consuming batches")

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, "NavSim-e120.*protocol_version"):
            train_cvoi_value_epoch(
                self.model,
                ForbiddenIterable(),
                optimizer=optimizer,
                phase="field_warmup",
                device=self.device,
                protocol_version="legacy_v1",
            )
        with self.assertRaisesRegex(ValueError, "NavSim-e120.*protocol_version"):
            validate_cvoi_field_epoch(
                self.model,
                real_batches=ForbiddenIterable(),
                counterfactual_batches=None,
                device=self.device,
                cf_field_supervision="none",
                protocol_version="legacy_v1",
            )

    @staticmethod
    def _mixed_field_batch():
        return {
            "z_observed": torch.randn(4, 2, 3, 4),
            "z_future": torch.randn(4, 2, 3, 4),
            "dataset_domains": ["real", "counterfactual", "real", "counterfactual"],
            "real_quality_targets": torch.tensor(
                [
                    [0.2, 0.6],
                    [float("nan"), float("nan")],
                    [0.7, 0.8],
                    [float("nan"), float("nan")],
                ]
            ),
            "real_group_ids": ["scene-a", "", "scene-a", ""],
            "cf_hazard": torch.tensor([False, True, False, True]),
            "cf_hazard_types": ["", "自车行为引起", "", "自车行为引起"],
            "cf_hazard_pair_real_indices": torch.tensor([0, 2]),
            "cf_hazard_pair_counterfactual_indices": torch.tensor([1, 3]),
            "cf_hazard_pair_keys": [("scene-a", 0), ("scene-a", 1)],
            "cf_quality": torch.tensor([float("nan"), 0.9, float("nan"), 0.1]),
        }

    @staticmethod
    def _real_stop_batch():
        return {
            "z_observed": torch.randn(2, 2, 3, 4),
            "z_future": torch.randn(2, 2, 3, 4),
            "dataset_domains": ["real", "real"],
            "stop_quality_targets": torch.tensor([[0.3, 0.5, 0.7], [0.4, 0.6, 0.8]]),
        }

    def test_field_warmup_trains_shared_field_representation_but_freezes_stop_head(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}

        diagnostics = train_cvoi_value_epoch(
            self.model,
            [self._mixed_field_batch()],
            optimizer=optimizer,
            phase="field_warmup",
            device=self.device,
        )

        after = dict(self.model.named_parameters())
        self.assertTrue(
            any(
                not torch.equal(before[name], after[name])
                for name in before
                if name != "stop_head.weight" and name != "stop_head.bias"
            )
        )
        self.assertTrue(torch.equal(before["stop_head.weight"], after["stop_head.weight"]))
        self.assertTrue(torch.equal(before["stop_head.bias"], after["stop_head.bias"]))
        self.assertTrue(
            all(parameter.requires_grad for name, parameter in after.items() if not name.startswith("stop_head."))
        )
        self.assertTrue(
            all(not parameter.requires_grad for name, parameter in after.items() if name.startswith("stop_head."))
        )
        self.assertEqual(diagnostics["num_batches"], 1.0)
        self.assertEqual(diagnostics["sample_count"], 4.0)
        self.assertEqual(diagnostics["real_count"], 2.0)
        self.assertEqual(diagnostics["cf_count"], 2.0)
        self.assertGreater(diagnostics["field_loss"], 0.0)

    def test_field_calibrated_uses_the_same_field_only_freezing_policy(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        train_cvoi_value_epoch(
            self.model,
            [self._real_field_batch()],
            optimizer=optimizer,
            phase="field_calibrated",
            device=self.device,
            field_loss_kwargs={"real_order_weight": 1.0, "real_order_margin": 0.1},
        )

        self.assertTrue(self.model.field_head.weight.requires_grad)
        self.assertTrue(self.model.prefix_gru.weight_ih_l0.requires_grad)
        self.assertFalse(self.model.stop_head.weight.requires_grad)

    def test_bfloat16_frozen_latents_are_cast_to_the_float32_value_model(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        batch = self._mixed_field_batch()
        batch["z_observed"] = batch["z_observed"].to(torch.bfloat16)
        batch["z_future"] = batch["z_future"].to(torch.bfloat16)

        diagnostics = train_cvoi_value_epoch(
            self.model,
            [batch],
            optimizer=optimizer,
            phase="field_warmup",
            device=self.device,
        )

        self.assertEqual(next(self.model.parameters()).dtype, torch.float32)
        self.assertGreater(diagnostics["field_loss"], 0.0)

    def test_stop_calibration_trains_only_stop_head(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}

        diagnostics = train_cvoi_value_epoch(
            self.model,
            [self._real_stop_batch()],
            optimizer=optimizer,
            phase="stop_calibrated",
            device=self.device,
        )

        after = dict(self.model.named_parameters())
        self.assertTrue(
            any(not torch.equal(before[name], after[name]) for name in before if name.startswith("stop_head."))
        )
        self.assertTrue(
            all(torch.equal(before[name], after[name]) for name in before if not name.startswith("stop_head."))
        )
        self.assertTrue(
            all(parameter.requires_grad for name, parameter in after.items() if name.startswith("stop_head."))
        )
        self.assertTrue(
            all(not parameter.requires_grad for name, parameter in after.items() if not name.startswith("stop_head."))
        )
        self.assertEqual(diagnostics["real_count"], 2.0)
        self.assertEqual(diagnostics["sample_count"], 2.0)
        self.assertGreater(diagnostics["stop_loss"], 0.0)

    def test_stop_calibration_rejects_counterfactual_before_forward(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        batch = self._real_stop_batch()
        batch["dataset_domains"] = ["real", "counterfactual"]

        with self.assertRaisesRegex(ValueError, "real-only"):
            train_cvoi_value_epoch(
                self.model,
                [batch],
                optimizer=optimizer,
                phase="stop_calibrated",
                device=self.device,
            )

    def test_field_phase_requires_domain_specific_targets(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        for missing_key in (
            "real_quality_targets",
            "cf_hazard",
            "cf_hazard_types",
            "cf_hazard_pair_real_indices",
            "cf_hazard_pair_counterfactual_indices",
            "cf_hazard_pair_keys",
            "cf_quality",
        ):
            with self.subTest(missing_key=missing_key):
                batch = self._mixed_field_batch()
                del batch[missing_key]
                with self.assertRaisesRegex(ValueError, missing_key):
                    train_cvoi_value_epoch(
                        self.model,
                        [batch],
                        optimizer=optimizer,
                        phase="field_warmup",
                        device=self.device,
                    )

    def test_cf_supervision_modes_never_access_disabled_labels(self):
        class ForbiddenLabelBatch(dict):
            def __init__(self, value, forbidden):
                super().__init__(value)
                self.forbidden = frozenset(forbidden)

            def __contains__(self, key):
                if key in self.forbidden:
                    raise AssertionError(f"disabled label was inspected: {key}")
                return super().__contains__(key)

            def __getitem__(self, key):
                if key in self.forbidden:
                    raise AssertionError(f"disabled label was consumed: {key}")
                return super().__getitem__(key)

        cases = {
            "none": {"cf_hazard", "cf_hazard_types", "cf_quality"},
            "hazard_only": {"cf_quality"},
            "quality_only": {"cf_hazard", "cf_hazard_types"},
            "hazard_quality": set(),
        }
        for mode, forbidden in cases.items():
            with self.subTest(mode=mode):
                model = PrefixDualValueModel(embed_dim=4, hidden_dim=6)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                batch = ForbiddenLabelBatch(self._mixed_field_batch(), forbidden)
                diagnostics = train_cvoi_value_epoch(
                    model,
                    [batch],
                    optimizer=optimizer,
                    phase="field_warmup",
                    device=self.device,
                    cf_field_supervision=mode,
                )
                self.assertEqual(diagnostics["cf_forward_count"], 1.0)
                self.assertEqual(diagnostics["optimizer_steps"], 1.0)
                if mode in {"none", "quality_only"}:
                    self.assertEqual(diagnostics["cf_hazard_pairs"], 0.0)
                if mode in {"none", "hazard_only"}:
                    self.assertEqual(diagnostics["cf_quality_pairs"], 0.0)

    def test_cf_supervision_mode_is_closed_and_cannot_be_overridden_by_weights(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, "cf_field_supervision"):
            train_cvoi_value_epoch(
                self.model,
                [self._mixed_field_batch()],
                optimizer=optimizer,
                phase="field_warmup",
                device=self.device,
                cf_field_supervision="all",
            )
        with self.assertRaisesRegex(ValueError, "owned by cf_field_supervision"):
            train_cvoi_value_epoch(
                self.model,
                [self._mixed_field_batch()],
                optimizer=optimizer,
                phase="field_warmup",
                device=self.device,
                cf_field_supervision="none",
                field_loss_kwargs={"cf_hazard_weight": 1.0},
            )

    def test_exact_value_update_budget_does_not_read_batch_n_plus_one(self):
        class CountingIterator:
            def __init__(self, values):
                self.values = iter(values)
                self.next_calls = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.next_calls += 1
                return next(self.values)

        batches = CountingIterator([self._mixed_field_batch() for _ in range(3)])
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        diagnostics = train_cvoi_value_epoch(
            self.model,
            batches,
            optimizer=optimizer,
            phase="field_warmup",
            device=self.device,
            value_updates_per_epoch=2,
        )

        self.assertEqual(batches.next_calls, 2)
        self.assertEqual(diagnostics["optimizer_steps"], 2.0)
        self.assertEqual(diagnostics["value_updates_per_epoch"], 2.0)

    def test_exact_value_update_budget_fails_when_iterator_is_short(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        with self.assertRaisesRegex(ValueError, "required 2.*provided only 1"):
            train_cvoi_value_epoch(
                self.model,
                [self._mixed_field_batch()],
                optimizer=optimizer,
                phase="field_warmup",
                device=self.device,
                value_updates_per_epoch=2,
            )

    def test_value_schedule_signature_is_stable_and_order_sensitive(self):
        def run(sample_ids):
            model = PrefixDualValueModel(embed_dim=4, hidden_dim=6)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            batch = self._mixed_field_batch()
            batch["stable_sample_ids"] = sample_ids
            return train_cvoi_value_epoch(
                model,
                [batch],
                optimizer=optimizer,
                phase="field_warmup",
                device=self.device,
                value_updates_per_epoch=1,
            )

        first = run(["real:a", "cf:b", "real:c", "cf:d"])
        second = run(["real:a", "cf:b", "real:c", "cf:d"])
        reordered = run(["cf:d", "real:c", "cf:b", "real:a"])

        self.assertEqual(first["eligibility_schedule_signature"], second["eligibility_schedule_signature"])
        self.assertNotEqual(first["eligibility_schedule_signature"], reordered["eligibility_schedule_signature"])
        self.assertEqual(first["cf_batch_count"], 1.0)
        self.assertEqual(first["cf_forward_count"], 1.0)
        self.assertEqual(first["optimizer_step_count"], 1.0)

    def test_requires_explicit_latents_domains_and_stop_quality_targets(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        for missing_key in ("z_observed", "z_future", "dataset_domains", "stop_quality_targets"):
            with self.subTest(missing_key=missing_key):
                batch = self._real_stop_batch()
                del batch[missing_key]
                with self.assertRaisesRegex(ValueError, missing_key):
                    train_cvoi_value_epoch(
                        self.model,
                        [batch],
                        optimizer=optimizer,
                        phase="stop_calibrated",
                        device=self.device,
                    )

    def test_rejects_unknown_phase_and_empty_epoch(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, "phase"):
            train_cvoi_value_epoch(
                self.model,
                [self._mixed_field_batch()],
                optimizer=optimizer,
                phase="joint",
                device=self.device,
            )
        with self.assertRaisesRegex(ValueError, "at least one batch"):
            train_cvoi_value_epoch(
                self.model,
                [],
                optimizer=optimizer,
                phase="field_warmup",
                device=self.device,
            )

    def test_navsim_e120_training_consumes_only_explicit_quality_target_keys(self):
        field_batch = self._real_field_batch()
        field_optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        field_metrics = train_cvoi_value_epoch(
            self.model,
            [field_batch],
            optimizer=field_optimizer,
            phase="field_calibrated",
            device=self.device,
            protocol_version="formal_v2_navsim_e120_h4_v3",
        )
        self.assertGreater(field_metrics["field_loss"], 0.0)

        stop_model = PrefixDualValueModel(embed_dim=4, hidden_dim=6)
        stop_batch = self._real_stop_batch()
        stop_optimizer = torch.optim.SGD(stop_model.parameters(), lr=0.1)
        stop_metrics = train_cvoi_value_epoch(
            stop_model,
            [stop_batch],
            optimizer=stop_optimizer,
            phase="stop_calibrated",
            device=self.device,
            protocol_version="formal_v2_navsim_e120_h4_v3",
        )
        self.assertGreater(stop_metrics["stop_loss"], 0.0)

    def test_navsim_e120_training_rejects_legacy_target_keys_before_forward(self):
        field_batch = self._real_field_batch()
        field_batch["real_geometry_targets"] = field_batch.pop("real_quality_targets")
        stop_batch = self._real_stop_batch()
        stop_batch["stop_targets"] = stop_batch.pop("stop_quality_targets")
        cases = {"field_calibrated": field_batch, "stop_calibrated": stop_batch}
        for phase, batch in cases.items():
            with self.subTest(phase=phase):
                model = PrefixDualValueModel(embed_dim=4, hidden_dim=6)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                forward_calls = []
                hook = model.register_forward_pre_hook(lambda *_: forward_calls.append(True))
                try:
                    with self.assertRaisesRegex(ValueError, "quality_targets"):
                        train_cvoi_value_epoch(
                            model,
                            [batch],
                            optimizer=optimizer,
                            phase=phase,
                            device=self.device,
                            protocol_version="formal_v2_navsim_e120_h4_v3",
                        )
                finally:
                    hook.remove()
                self.assertEqual(forward_calls, [])

    def test_field_validation_is_no_grad_and_does_not_mutate_model(self):
        before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        for parameter in self.model.parameters():
            parameter.grad = torch.randn_like(parameter)
        before_gradients = {name: parameter.grad.detach().clone() for name, parameter in self.model.named_parameters()}
        self.model.train()
        self.model.stop_head.eval()
        before_modes = {name: module.training for name, module in self.model.named_modules()}
        grad_enabled = []

        def record_grad_mode(_module, _inputs):
            grad_enabled.append(torch.is_grad_enabled())

        hook = self.model.register_forward_pre_hook(record_grad_mode)
        try:
            metrics = validate_cvoi_field_epoch(
                self.model,
                real_batches=[self._real_field_batch()],
                counterfactual_batches=[
                    self._navsim_matched_pair_batch(("scene-a", 0), quality=0.8),
                    self._navsim_matched_pair_batch(("scene-b", 0), quality=0.2),
                ],
                device=self.device,
                cf_field_supervision="hazard_quality",
            )
        finally:
            hook.remove()

        self.assertEqual(grad_enabled, [False, False, False])
        self.assertEqual(metrics["real"]["sample_count"], 2)
        self.assertEqual(metrics["counterfactual"]["sample_count"], 2)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(before_gradients[name], parameter.grad), name)
        self.assertEqual(
            {name: module.training for name, module in self.model.named_modules()},
            before_modes,
        )

    @staticmethod
    def _real_field_batch():
        return {
            "z_observed": torch.randn(2, 2, 3, 4),
            "z_future": torch.randn(2, 2, 3, 4),
            "dataset_domains": ["real", "real"],
            "real_quality_targets": torch.tensor([[0.2, 0.6], [0.7, 0.8]]),
            "real_group_ids": ["scene-a", "scene-a"],
        }

    @staticmethod
    def _counterfactual_field_batch(*, hazard: bool, quality: float):
        return {
            "z_observed": torch.randn(1, 2, 3, 4),
            "z_future": torch.randn(1, 2, 3, 4),
            "dataset_domains": ["counterfactual"],
            "cf_hazard": torch.tensor([hazard]),
            "cf_hazard_types": ["自车行为引起" if hazard else ""],
            "cf_quality": torch.tensor([quality]),
        }

    @staticmethod
    def _counterfactual_pair_batch():
        return {
            "z_observed": torch.randn(2, 2, 3, 4),
            "z_future": torch.randn(2, 2, 3, 4),
            "dataset_domains": ["counterfactual", "counterfactual"],
            "cf_hazard": torch.tensor([False, True]),
            "cf_hazard_types": ["", "自车行为引起"],
            "cf_quality": torch.tensor([0.9, 0.1]),
        }

    @staticmethod
    def _navsim_matched_pair_batch(pair_key: tuple[str, int], *, quality: float):
        return {
            "z_observed": torch.randn(2, 2, 3, 4),
            "z_future": torch.randn(2, 2, 3, 4),
            "dataset_domains": ["real", "counterfactual"],
            "real_quality_targets": torch.tensor([[0.2, 0.6], [float("nan"), float("nan")]]),
            "cf_hazard": torch.tensor([False, True]),
            "cf_hazard_types": ["", "自车行为引起"],
            "cf_hazard_pair_real_indices": torch.tensor([0], dtype=torch.long),
            "cf_hazard_pair_counterfactual_indices": torch.tensor([1], dtype=torch.long),
            "cf_hazard_pair_keys": [pair_key],
            "cf_quality": torch.tensor([float("nan"), quality]),
        }

    def test_navsim_h4_field_validation_aggregates_exact_matched_pairs_across_batches(self):
        keys = [("scene-a", 3), ("scene-b", 7)]
        metrics = validate_cvoi_field_epoch(
            self.model,
            real_batches=[self._real_field_batch()],
            counterfactual_batches=[
                self._navsim_matched_pair_batch(keys[0], quality=0.9),
                self._navsim_matched_pair_batch(keys[1], quality=0.1),
            ],
            device=self.device,
            cf_field_supervision="hazard_quality",
            protocol_version="formal_v2_navsim_e120_h4_v3",
        )

        counterfactual = metrics["counterfactual"]
        expected_digest = hashlib.sha256(
            json.dumps(sorted([list(key) for key in keys]), separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).hexdigest()
        self.assertEqual(counterfactual["sample_count"], 2)
        hazard = counterfactual["hazard_ranking"]
        self.assertEqual(hazard["enabled"], True)
        self.assertEqual(hazard["cf_hazard_count"], 2)
        self.assertEqual(hazard["factual_anchor_count"], 2)
        self.assertEqual(hazard["expected_matched_pair_count"], 2)
        self.assertEqual(hazard["valid_matched_pair_count"], 2)
        self.assertEqual(hazard["pair_coverage"], 1.0)
        self.assertEqual(hazard["matched_pair_key_set_sha256"], expected_digest)
        self.assertFalse(
            {"label_count", "safe_label_count", "hazard_label_count", "candidate_pair_count", "valid_pair_count"}
            & set(hazard)
        )
        self.assertEqual(counterfactual["quality_ranking"]["label_count"], 2)
        self.assertEqual(counterfactual["quality_ranking"]["candidate_pair_count"], 1)
        self.assertAlmostEqual(
            counterfactual["field_loss"],
            counterfactual["hazard_ranking"]["ranking_loss"] + counterfactual["quality_ranking"]["ranking_loss"],
        )

    def test_field_validation_never_reads_disabled_counterfactual_labels(self):
        class ForbiddenLabelBatch(dict):
            def __init__(self, value, forbidden):
                super().__init__(value)
                self.forbidden = frozenset(forbidden)

            def __contains__(self, key):
                if key in self.forbidden:
                    raise AssertionError(f"disabled label was inspected: {key}")
                return super().__contains__(key)

            def __getitem__(self, key):
                if key in self.forbidden:
                    raise AssertionError(f"disabled label was consumed: {key}")
                return super().__getitem__(key)

        cases = {
            "hazard_only": (
                self._navsim_matched_pair_batch(("scene-a", 0), quality=0.2),
                {"cf_quality"},
                "quality_ranking",
            ),
            "quality_only": (
                self._counterfactual_pair_batch(),
                {"cf_hazard", "cf_hazard_types"},
                "hazard_ranking",
            ),
        }
        for mode, (raw_batch, forbidden, disabled_metric) in cases.items():
            with self.subTest(mode=mode):
                batch = ForbiddenLabelBatch(raw_batch, forbidden)
                metrics = validate_cvoi_field_epoch(
                    self.model,
                    real_batches=[self._real_field_batch()],
                    counterfactual_batches=[batch],
                    device=self.device,
                    cf_field_supervision=mode,
                )

                counterfactual = metrics["counterfactual"]
                self.assertEqual(counterfactual[disabled_metric], {"enabled": False})

    def test_field_validation_requires_exact_loader_domains_and_required_pair_coverage(self):
        mixed = self._real_field_batch()
        mixed["dataset_domains"] = ["real", "counterfactual"]
        with self.assertRaisesRegex(ValueError, "real validation loader.*only real"):
            validate_cvoi_field_epoch(
                self.model,
                real_batches=[mixed],
                counterfactual_batches=None,
                device=self.device,
                cf_field_supervision="none",
            )

        with self.assertRaisesRegex(ValueError, "matched_real_counterfactual"):
            validate_cvoi_field_epoch(
                self.model,
                real_batches=[self._real_field_batch()],
                counterfactual_batches=[
                    self._counterfactual_field_batch(hazard=False, quality=0.9),
                    self._counterfactual_field_batch(hazard=False, quality=0.1),
                ],
                device=self.device,
                cf_field_supervision="hazard_only",
            )

    def test_strict_real_only_field_validation_never_consumes_a_cf_loader(self):
        class ForbiddenIterable:
            def __iter__(self):
                raise AssertionError("counterfactual loader was consumed")

        metrics = validate_cvoi_field_epoch(
            self.model,
            real_batches=[self._real_field_batch()],
            counterfactual_batches=None,
            device=self.device,
            cf_field_supervision="none",
        )
        self.assertIsNone(metrics["counterfactual"])

        with self.assertRaisesRegex(ValueError, "counterfactual_batches must be None"):
            validate_cvoi_field_epoch(
                self.model,
                real_batches=[self._real_field_batch()],
                counterfactual_batches=ForbiddenIterable(),
                device=self.device,
                cf_field_supervision="none",
            )

    def test_stop_and_navsim_field_calibration_reject_counterfactual_before_model_forward(self):
        cases = {
            "field_calibrated": (self._mixed_field_batch(), "formal_v2_navsim_e120_h4_v3"),
            "stop_calibrated": (
                {**self._real_stop_batch(), "dataset_domains": ["real", "counterfactual"]},
                "formal_v2_navsim_e120_h4_v3",
            ),
        }
        for phase, (batch, protocol_version) in cases.items():
            with self.subTest(phase=phase):
                optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
                calls = {"zero_grad": 0, "forward": 0}
                original_zero_grad = optimizer.zero_grad

                def record_zero_grad(*args, **kwargs):
                    calls["zero_grad"] += 1
                    return original_zero_grad(*args, **kwargs)

                def record_forward(_module, _inputs):
                    calls["forward"] += 1

                optimizer.zero_grad = record_zero_grad
                hook = self.model.register_forward_pre_hook(record_forward)
                try:
                    with self.assertRaisesRegex(ValueError, f"{phase}.*real-only"):
                        train_cvoi_value_epoch(
                            self.model,
                            [batch],
                            optimizer=optimizer,
                            phase=phase,
                            device=self.device,
                            protocol_version=protocol_version,
                        )
                finally:
                    hook.remove()

                self.assertEqual(calls, {"zero_grad": 0, "forward": 0})


class TestCVoINavSimE120DirectValueCheckpoint(unittest.TestCase):
    _FIELDS = {
        "schema",
        "phase",
        "protocol_version",
        "branch_id",
        "epoch",
        "architecture",
        "roles",
        "parents",
        "state_dict",
    }
    _PHASE_BRANCHES = {
        "field_warmup": "field_full",
        "field_calibrated": "calibration_full",
        "stop_calibrated": "stop_full",
    }
    _CHECKPOINT_MATRIX = (
        ("field_warmup", "field_full"),
        ("field_warmup", "field_no_cf"),
        ("field_warmup", "field_hazard_only"),
        ("field_warmup", "field_quality_only"),
        ("field_calibrated", "calibration_full"),
        ("field_calibrated", "calibration_no_cf"),
        ("field_calibrated", "calibration_hazard_only"),
        ("field_calibrated", "calibration_quality_only"),
        ("stop_calibrated", "stop_full"),
        ("stop_calibrated", "stop_no_cf"),
    )

    def setUp(self):
        torch.manual_seed(31)
        self.model = PrefixDualValueModel(embed_dim=4, hidden_dim=6)

    @classmethod
    def _parents(cls, phase, branch_id=None):
        resolved_branch_id = branch_id or cls._PHASE_BRANCHES[phase]
        lineage = resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
            phase=phase,
            branch_id=resolved_branch_id,
        )
        return build_cvoi_manual_value_parents(lineage, phase)

    def _payload(self, phase="field_calibrated", branch_id=None):
        resolved_branch_id = branch_id or self._PHASE_BRANCHES[phase]
        return cvoi_value.build_cvoi_navsim_e120_direct_value_checkpoint(
            self.model,
            phase=phase,
            branch_id=resolved_branch_id,
            epoch=3,
            parents=self._parents(phase, resolved_branch_id),
        )

    def test_direct_checkpoint_branch_matrix_builds_and_validates_exact_lineage_parents(self):
        self.assertEqual(len(self._CHECKPOINT_MATRIX), 10)
        for phase, branch_id in self._CHECKPOINT_MATRIX:
            with self.subTest(phase=phase, branch_id=branch_id):
                lineage = resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
                    phase=phase,
                    branch_id=branch_id,
                )
                parents = build_cvoi_manual_value_parents(lineage, phase)
                payload = cvoi_value.build_cvoi_navsim_e120_direct_value_checkpoint(
                    self.model,
                    phase=phase,
                    branch_id=branch_id,
                    epoch=3,
                    parents=parents,
                )
                self.assertEqual(set(payload), self._FIELDS)
                self.assertEqual(
                    payload["schema"],
                    cvoi_value.CVOI_NAVSIM_E120_VALUE_CHECKPOINT_SCHEMA,
                )
                self.assertEqual(
                    payload["protocol_version"],
                    cvoi_value.CVOI_VALUE_PROTOCOL_FORMAL_V2_NAVSIM_E120,
                )
                self.assertEqual(payload["phase"], phase)
                self.assertEqual(payload["branch_id"], branch_id)
                self.assertEqual(payload["epoch"], 3)
                self.assertEqual(payload["parents"], parents)
                self.assertEqual(set(payload["roles"]), {"value_model"})
                self.assertEqual(set(payload["roles"]["value_model"]), {"keys", "shapes"})
                keys = payload["roles"]["value_model"]["keys"]
                self.assertEqual(keys, sorted(self.model.state_dict()))
                self.assertEqual(
                    payload["roles"]["value_model"]["shapes"],
                    {key: list(self.model.state_dict()[key].shape) for key in keys},
                )
                normalized = cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(
                    payload,
                    required_phase=phase,
                    required_branch_id=branch_id,
                )
                self.assertEqual(set(normalized), self._FIELDS)
                for key, tensor in normalized["state_dict"].items():
                    self.assertTrue(torch.equal(tensor, self.model.state_dict()[key]))
                    self.assertNotEqual(tensor.data_ptr(), payload["state_dict"][key].data_ptr())

    def test_validation_rejects_recursive_proof_keys(self):
        for proof_key in (
            "receipt",
            "audit_signature",
            "data_provenance",
            "source_commit",
            "state_sha256",
        ):
            with self.subTest(proof_key=proof_key):
                payload = self._payload()
                payload["parents"]["field"][proof_key] = "forbidden"
                with self.assertRaisesRegex(ValueError, "forbidden proof key"):
                    cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(payload)

    def test_validation_rejects_phase_branch_epoch_and_required_identity_drift(self):
        cases = (
            ("phase", "joint"),
            ("branch_id", "field_full"),
            ("epoch", 0),
            ("epoch", True),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                payload = self._payload()
                payload[key] = value
                with self.assertRaisesRegex(ValueError, key if key != "branch_id" else "branch"):
                    cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(payload)

        payload = self._payload()
        with self.assertRaisesRegex(ValueError, "required phase"):
            cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(
                payload,
                required_phase="field_warmup",
            )
        with self.assertRaisesRegex(ValueError, "required branch"):
            cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(
                payload,
                required_branch_id="field_full",
            )

    def test_validation_rejects_role_and_role_metadata_drift(self):
        mutations = []

        missing_role = self._payload()
        missing_role["roles"] = {}
        mutations.append(("roles", missing_role))

        extra_role = self._payload()
        extra_role["roles"]["planner"] = copy.deepcopy(extra_role["roles"]["value_model"])
        mutations.append(("roles", extra_role))

        missing_role_field = self._payload()
        del missing_role_field["roles"]["value_model"]["shapes"]
        mutations.append(("fields", missing_role_field))

        extra_role_field = self._payload()
        extra_role_field["roles"]["value_model"]["dtype"] = "float32"
        mutations.append(("fields", extra_role_field))

        missing_key = self._payload()
        missing_key["roles"]["value_model"]["keys"].pop()
        mutations.append(("keys", missing_key))

        duplicate_key = self._payload()
        duplicate_key["roles"]["value_model"]["keys"].append(duplicate_key["roles"]["value_model"]["keys"][-1])
        mutations.append(("sorted and unique", duplicate_key))

        extra_shape = self._payload()
        extra_shape["roles"]["value_model"]["shapes"]["extra"] = [1]
        mutations.append(("shapes keys", extra_shape))

        for error, payload in mutations:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(payload)

    def test_validation_rejects_state_key_tensor_and_shape_drift(self):
        missing_state = self._payload()
        missing_state["state_dict"].pop(next(iter(missing_state["state_dict"])))

        extra_state = self._payload()
        extra_state["state_dict"]["extra"] = torch.ones(1)

        non_tensor_state = self._payload()
        non_tensor_state["state_dict"]["frame_norm.weight"] = [1.0]

        declared_shape_drift = self._payload()
        declared_shape_drift["roles"]["value_model"]["shapes"]["frame_norm.weight"] = [5]

        tensor_shape_drift = self._payload()
        tensor_shape_drift["state_dict"]["frame_norm.weight"] = torch.ones(5)

        for error, payload in (
            ("state keys", missing_state),
            ("state keys", extra_state),
            ("string keys to tensors", non_tensor_state),
            ("state shapes", declared_shape_drift),
            ("state shapes", tensor_shape_drift),
        ):
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(payload)

    def test_validation_rejects_self_consistent_state_that_does_not_match_declared_architecture(self):
        payload = self._payload()
        payload["state_dict"] = {"fake.weight": torch.ones(2, 3)}
        payload["roles"] = {
            "value_model": {
                "keys": ["fake.weight"],
                "shapes": {"fake.weight": [2, 3]},
            }
        }

        with self.assertRaisesRegex(ValueError, "declared architecture state keys"):
            cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(payload)

    def test_build_does_not_consume_global_cpu_rng(self):
        before = torch.random.get_rng_state().clone()

        self._payload()

        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    def test_validate_does_not_consume_global_cpu_rng(self):
        payload = self._payload()
        before = torch.random.get_rng_state().clone()

        cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(payload)

        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    def test_calibration_metadata_read_does_not_consume_global_cpu_rng(self):
        payload = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.pt"
            torch.save(payload, path)
            before = torch.random.get_rng_state().clone()

            cvoi_value.read_cvoi_navsim_e120_direct_calibration_checkpoint_metadata(
                path,
                required_branch_id="calibration_full",
            )

            self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    def test_validation_rejects_wrong_phase_specific_parents(self):
        cases = []
        for phase in self._PHASE_BRANCHES:
            payload = self._payload(phase)
            payload["parents"]["unexpected"] = {}
            cases.append(payload)

        wrong_identity = self._payload("stop_calibrated")
        wrong_identity["parents"]["guided_planner"]["branch_id"] = "p0_uniform"
        cases.append(wrong_identity)

        for payload in cases:
            with self.subTest(phase=payload["phase"]):
                with self.assertRaisesRegex(ValueError, "parent"):
                    cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(payload)

    def test_direct_parent_and_branch_mutations_fail_before_target_model_state_load(self):
        cases = []

        calibration_no_cf = self._payload("field_calibrated", "calibration_no_cf")
        calibration_no_cf["parents"]["field"]["branch_id"] = "field_full"
        cases.append(("calibration_no_cf parent field_full", calibration_no_cf, "parent"))

        stop_no_cf_calibration = self._payload("stop_calibrated", "stop_no_cf")
        stop_no_cf_calibration["parents"]["calibration"]["branch_id"] = "calibration_full"
        cases.append(("stop_no_cf parent calibration_full", stop_no_cf_calibration, "parent"))

        stop_no_cf_guided = self._payload("stop_calibrated", "stop_no_cf")
        stop_no_cf_guided["parents"]["guided_planner"]["branch_id"] = "p1_full"
        cases.append(("stop_no_cf parent p1_full", stop_no_cf_guided, "parent"))

        for branch_id in ("stop_hazard_only", "stop_quality_only"):
            unsupported_stop = self._payload("stop_calibrated", "stop_full")
            unsupported_stop["branch_id"] = branch_id
            cases.append((branch_id, unsupported_stop, "Stop stage"))

        for name, payload, error in cases:
            with self.subTest(name=name):
                target = PrefixDualValueModel(embed_dim=4, hidden_dim=6)
                before = {key: tensor.detach().clone() for key, tensor in target.state_dict().items()}
                load_calls = []
                original_load_state_dict = target.load_state_dict

                def record_load_state_dict(*args, **kwargs):
                    load_calls.append((args, kwargs))
                    return original_load_state_dict(*args, **kwargs)

                target.load_state_dict = record_load_state_dict
                with self.assertRaisesRegex(ValueError, error):
                    cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(
                        payload,
                        target_model=target,
                    )
                self.assertEqual(load_calls, [])
                for key, tensor in target.state_dict().items():
                    self.assertTrue(torch.equal(tensor, before[key]))

    def test_target_model_is_strictly_loaded_after_architecture_key_and_shape_checks(self):
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(1.0)
        payload = self._payload()
        target = PrefixDualValueModel(embed_dim=4, hidden_dim=6)

        cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(
            payload,
            target_model=target,
        )

        for key, tensor in target.state_dict().items():
            self.assertTrue(torch.equal(tensor, payload["state_dict"][key]))

        architecture_drift = PrefixDualValueModel(embed_dim=5, hidden_dim=6)
        with self.assertRaisesRegex(ValueError, "architecture mismatch"):
            cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(
                payload,
                target_model=architecture_drift,
            )

        key_drift = PrefixDualValueModel(embed_dim=4, hidden_dim=6)
        key_drift.register_buffer("extra_state", torch.ones(1))
        with self.assertRaisesRegex(ValueError, "target model state keys"):
            cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(
                payload,
                target_model=key_drift,
            )

        shape_drift = PrefixDualValueModel(embed_dim=4, hidden_dim=6)
        shape_drift.field_head = torch.nn.Linear(6, 2)
        with self.assertRaisesRegex(ValueError, "target model state shapes"):
            cvoi_value.validate_cvoi_navsim_e120_direct_value_checkpoint(
                payload,
                target_model=shape_drift,
            )

    def test_reader_loads_regular_file_and_strictly_restores_target_model(self):
        payload = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.pt"
            torch.save(payload, path)
            target = PrefixDualValueModel(embed_dim=4, hidden_dim=6)

            loaded = cvoi_value.read_cvoi_navsim_e120_direct_value_checkpoint(
                path,
                required_phase="field_calibrated",
                required_branch_id="calibration_full",
                target_model=target,
            )

        self.assertEqual(set(loaded), self._FIELDS)
        for key, tensor in target.state_dict().items():
            self.assertTrue(torch.equal(tensor, payload["state_dict"][key]))

    def test_reader_loads_direct_value_checkpoint_with_weights_only_enabled(self):
        payload = self._payload()
        map_location = torch.device("cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.pt"
            path.write_bytes(b"direct Value checkpoint sentinel")

            def load_payload(
                artifact,
                *,
                map_location: torch.device,
                weights_only: bool,
            ):
                self.assertEqual(artifact, path)
                self.assertEqual(map_location, torch.device("cpu"))
                self.assertIs(weights_only, True)
                return payload

            with mock.patch.object(cvoi_value.torch, "load", side_effect=load_payload):
                loaded = cvoi_value.read_cvoi_navsim_e120_direct_value_checkpoint(
                    path,
                    required_phase="field_calibrated",
                    required_branch_id="calibration_full",
                    map_location=map_location,
                )

        self.assertEqual(loaded["branch_id"], "calibration_full")

    def test_reader_rejects_damaged_relative_symlink_and_directory_paths(self):
        payload = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.pt"
            torch.save(payload, valid)
            damaged = root / "damaged.pt"
            damaged.write_bytes(b"not a torch checkpoint")
            link = root / "link.pt"
            link.symlink_to(valid)

            with self.assertRaisesRegex(ValueError, "absolute"):
                cvoi_value.read_cvoi_navsim_e120_direct_value_checkpoint(
                    Path("relative.pt"),
                    required_phase="field_calibrated",
                    required_branch_id="calibration_full",
                )
            with self.assertRaises(Exception):
                cvoi_value.read_cvoi_navsim_e120_direct_value_checkpoint(
                    damaged,
                    required_phase="field_calibrated",
                    required_branch_id="calibration_full",
                )
            with self.assertRaisesRegex(ValueError, "non-symlink regular file"):
                cvoi_value.read_cvoi_navsim_e120_direct_value_checkpoint(
                    link,
                    required_phase="field_calibrated",
                    required_branch_id="calibration_full",
                )
            with self.assertRaisesRegex(ValueError, "non-symlink regular file"):
                cvoi_value.read_cvoi_navsim_e120_direct_value_checkpoint(
                    root,
                    required_phase="field_calibrated",
                    required_branch_id="calibration_full",
                )

    def test_calibration_metadata_reader_fully_validates_and_returns_exact_callback_contract(self):
        payload = self._payload()
        expected = {
            key: copy.deepcopy(payload[key])
            for key in (
                "schema",
                "phase",
                "protocol_version",
                "branch_id",
                "epoch",
                "roles",
                "parents",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.pt"
            torch.save(payload, path)
            metadata = cvoi_value.read_cvoi_navsim_e120_direct_calibration_checkpoint_metadata(
                path,
                required_branch_id="calibration_full",
            )
            self.assertEqual(metadata, expected)

            poisoned = self._payload()
            poisoned["state_dict"]["frame_norm.weight"] = torch.ones(5)
            torch.save(poisoned, path)
            with self.assertRaisesRegex(ValueError, "state shapes"):
                cvoi_value.read_cvoi_navsim_e120_direct_calibration_checkpoint_metadata(
                    path,
                    required_branch_id="calibration_full",
                )

    def test_direct_calibration_metadata_reader_requires_exact_branch(self):
        payload = self._payload("field_calibrated", "calibration_no_cf")
        expected = {
            key: copy.deepcopy(payload[key])
            for key in (
                "schema",
                "phase",
                "protocol_version",
                "branch_id",
                "epoch",
                "roles",
                "parents",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_no_cf.pt"
            torch.save(payload, path)

            metadata = cvoi_value.read_cvoi_navsim_e120_direct_calibration_checkpoint_metadata(
                path,
                required_branch_id="calibration_no_cf",
            )
            self.assertEqual(metadata, expected)

            with self.assertRaisesRegex(ValueError, "required branch"):
                cvoi_value.read_cvoi_navsim_e120_direct_calibration_checkpoint_metadata(
                    path,
                    required_branch_id="calibration_full",
                )
            with self.assertRaises(TypeError):
                cvoi_value.read_cvoi_navsim_e120_direct_calibration_checkpoint_metadata(path)


if __name__ == "__main__":
    unittest.main()
