import json
import tempfile
import unittest
from pathlib import Path

import torch

from app.vjepa_cowa_world_model.training.budget_control import load_budget_controller_from_checkpoint
from app.vjepa_cowa_world_model.training.budget_controller_training import (
    interpolate_budget_utilities,
    load_budget_oracle_examples,
    load_budget_oracle_sweeps,
    train_budget_controller_from_config,
    train_budget_controller_from_oracle,
    train_budget_controller_grpo_from_oracle,
)
from app.vjepa_cowa_world_model.training.config import parse_training_config


class TestBudgetControllerTraining(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_loads_oracle_sweep_and_selects_best_budget_per_scene(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            oracle_path = Path(tmpdir) / "oracle.jsonl"
            self._write_jsonl(
                oracle_path,
                [
                    {"scene_id": "a", "budget": 0.0, "utility": 0.1, "pooled_latent": [1.0, 0.0]},
                    {"scene_id": "a", "budget": 1.0, "utility": 0.9, "pooled_latent": [1.0, 0.0]},
                    {"scene_id": "b", "budget": 0.0, "utility": 0.8, "pooled_latent": [0.0, 1.0]},
                    {"scene_id": "b", "budget": 1.0, "utility": 0.2, "pooled_latent": [0.0, 1.0]},
                ],
            )

            examples = load_budget_oracle_examples(oracle_path, feature_dim=0, lambda_compute=0.05)

        self.assertEqual(examples.scene_ids, ["a", "b"])
        self.assertTrue(torch.allclose(examples.pooled_latent, torch.tensor([[1.0, 0.0], [0.0, 1.0]])))
        self.assertIsNone(examples.cheap_features)
        self.assertTrue(torch.allclose(examples.target_budget, torch.tensor([1.0, 0.0])))

    def test_loads_oracle_sweeps_grouped_by_scene(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            oracle_path = Path(tmpdir) / "oracle.jsonl"
            self._write_jsonl(
                oracle_path,
                [
                    {"scene_id": "b", "budget": 0.0, "utility": 0.2, "pooled_latent": [0.0, 1.0]},
                    {"scene_id": "b", "budget": 0.5, "utility": 0.5, "pooled_latent": [0.0, 1.0]},
                    {"scene_id": "a", "budget": 0.0, "utility": 0.1, "pooled_latent": [1.0, 0.0]},
                    {"scene_id": "a", "budget": 1.0, "utility": 0.9, "pooled_latent": [1.0, 0.0]},
                ],
            )

            sweeps = load_budget_oracle_sweeps(oracle_path, feature_dim=0, lambda_compute=0.05)

        self.assertEqual(sweeps.scene_ids, ["a", "b"])
        self.assertTrue(torch.allclose(sweeps.pooled_latent, torch.tensor([[1.0, 0.0], [0.0, 1.0]])))
        self.assertTrue(torch.allclose(sweeps.budget_grid, torch.tensor([[0.0, 1.0], [0.0, 0.5]])))
        self.assertTrue(torch.allclose(sweeps.utility_grid, torch.tensor([[0.1, 0.9], [0.2, 0.5]])))
        self.assertTrue(torch.allclose(sweeps.target_budget, torch.tensor([1.0, 0.5])))

    def test_load_oracle_sweeps_rejects_single_budget_scene(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            oracle_path = Path(tmpdir) / "oracle.jsonl"
            self._write_jsonl(
                oracle_path,
                [
                    {"scene_id": "a", "budget": 0.0, "utility": 0.1, "pooled_latent": [1.0, 0.0]},
                ],
            )

            with self.assertRaisesRegex(ValueError, "at least 2 budget"):
                load_budget_oracle_sweeps(oracle_path, feature_dim=0, lambda_compute=0.05)

    def test_interpolates_budget_utilities_linearly(self):
        budget_grid = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=torch.float32)
        utility_grid = torch.tensor([[0.0, 1.0, 0.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
        sampled_budget = torch.tensor([[0.25, 0.75], [0.25, 0.75]], dtype=torch.float32)

        reward = interpolate_budget_utilities(budget_grid, utility_grid, sampled_budget)

        expected = torch.tensor([[0.5, 0.5], [1.5, 2.5]], dtype=torch.float32)
        self.assertTrue(torch.allclose(reward, expected))

    def test_trains_controller_from_oracle_and_saves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            oracle_path = tmp / "oracle.jsonl"
            checkpoint_path = tmp / "budget_controller.pt"
            rows = []
            for idx in range(12):
                rows.append(
                    {
                        "scene_id": f"low_{idx}",
                        "target_budget": 0.1,
                        "pooled_latent": [0.0, 1.0],
                    }
                )
                rows.append(
                    {
                        "scene_id": f"high_{idx}",
                        "target_budget": 0.9,
                        "pooled_latent": [1.0, 0.0],
                    }
                )
            self._write_jsonl(oracle_path, rows)
            cfg = parse_training_config(
                {
                    "folder": str(tmp),
                    "data": {"batch_size": 8},
                    "optimization": {
                        "epochs": 6,
                        "lr": 0.02,
                        "start_lr": 0.02,
                        "weight_decay": 0.0,
                        "grad_clip_norm": 1.0,
                    },
                    "budget_controller": {
                        "enabled": True,
                        "mode": "oracle_distillation",
                        "oracle_path": str(oracle_path),
                        "output_checkpoint": str(checkpoint_path),
                        "hidden_dim": 16,
                        "feature_dim": 0,
                        "bc_mse_weight": 1.0,
                    },
                }
            )

            metrics = train_budget_controller_from_oracle(cfg, device=torch.device("cpu"))

            self.assertTrue(checkpoint_path.exists())
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["latent_dim"], 2)
            self.assertEqual(payload["feature_dim"], 0)
            self.assertIn("controller", payload)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["final_loss"])))
            self.assertLess(metrics["final_loss"], metrics["initial_loss"])

    def test_grpo_finetune_from_oracle_moves_budget_toward_high_utility(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            oracle_path = tmp / "oracle.jsonl"
            bc_checkpoint = tmp / "budget_controller_bc.pt"
            grpo_checkpoint = tmp / "budget_controller_grpo.pt"
            rows = []
            for idx in range(16):
                latent = [1.0, 0.0] if idx % 2 == 0 else [0.0, 1.0]
                rows.append({"scene_id": f"scene_{idx}", "budget": 0.0, "utility": 0.0, "pooled_latent": latent})
                rows.append({"scene_id": f"scene_{idx}", "budget": 1.0, "utility": 1.0, "pooled_latent": latent})
            self._write_jsonl(oracle_path, rows)
            bc_cfg = parse_training_config(
                {
                    "folder": str(tmp),
                    "data": {"batch_size": 8},
                    "optimization": {
                        "epochs": 3,
                        "lr": 0.02,
                        "start_lr": 0.02,
                        "weight_decay": 0.0,
                        "grad_clip_norm": 1.0,
                    },
                    "budget_controller": {
                        "enabled": True,
                        "mode": "oracle_distillation",
                        "oracle_path": str(oracle_path),
                        "output_checkpoint": str(bc_checkpoint),
                        "hidden_dim": 16,
                        "feature_dim": 0,
                        "min_concentration": 0.5,
                        "bc_mse_weight": 1.0,
                    },
                }
            )
            train_budget_controller_from_oracle(bc_cfg, device=torch.device("cpu"))
            before = load_budget_controller_from_checkpoint(bc_checkpoint, device=torch.device("cpu"))
            pooled = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            with torch.no_grad():
                budget_before = before.sample_budget(pooled, deterministic=True).budget.mean().item()

            grpo_cfg = parse_training_config(
                {
                    "folder": str(tmp),
                    "data": {"batch_size": 8},
                    "optimization": {
                        "epochs": 20,
                        "lr": 0.01,
                        "start_lr": 0.01,
                        "weight_decay": 0.0,
                        "grad_clip_norm": 1.0,
                    },
                    "budget_controller": {
                        "enabled": True,
                        "mode": "grpo",
                        "oracle_path": str(oracle_path),
                        "controller_checkpoint": str(bc_checkpoint),
                        "output_checkpoint": str(grpo_checkpoint),
                        "hidden_dim": 16,
                        "feature_dim": 0,
                        "min_concentration": 0.5,
                        "grpo_num_samples_per_scene": 4,
                        "grpo_bc_weight": 0.0,
                    },
                }
            )

            metrics = train_budget_controller_grpo_from_oracle(grpo_cfg, device=torch.device("cpu"))
            after = load_budget_controller_from_checkpoint(grpo_checkpoint, device=torch.device("cpu"))
            with torch.no_grad():
                budget_after = after.sample_budget(pooled, deterministic=True).budget.mean().item()

            self.assertTrue(grpo_checkpoint.exists())
            self.assertGreater(metrics["final_expected_utility"], metrics["initial_expected_utility"])
            self.assertGreater(budget_after, budget_before)

    def test_bc_then_grpo_dispatcher_saves_final_grpo_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            oracle_path = tmp / "oracle.jsonl"
            checkpoint_path = tmp / "budget_controller.pt"
            rows = []
            for idx in range(8):
                rows.append({"scene_id": f"scene_{idx}", "budget": 0.0, "utility": 0.0, "pooled_latent": [1.0, 0.0]})
                rows.append({"scene_id": f"scene_{idx}", "budget": 1.0, "utility": 1.0, "pooled_latent": [1.0, 0.0]})
            self._write_jsonl(oracle_path, rows)
            cfg = parse_training_config(
                {
                    "folder": str(tmp),
                    "data": {"batch_size": 4},
                    "optimization": {
                        "epochs": 2,
                        "lr": 0.01,
                        "start_lr": 0.01,
                        "weight_decay": 0.0,
                        "grad_clip_norm": 1.0,
                    },
                    "budget_controller": {
                        "enabled": True,
                        "mode": "bc_then_grpo",
                        "oracle_path": str(oracle_path),
                        "output_checkpoint": str(checkpoint_path),
                        "hidden_dim": 8,
                        "feature_dim": 0,
                        "min_concentration": 0.5,
                        "grpo_num_samples_per_scene": 2,
                    },
                }
            )

            metrics = train_budget_controller_from_config(cfg, device=torch.device("cpu"))
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        self.assertEqual(payload["mode"], "grpo")
        self.assertIn("bc_final_loss", metrics)
        self.assertIn("grpo_final_loss", metrics)


if __name__ == "__main__":
    unittest.main()
