import unittest

from app.vjepa_cowa_world_model.training.config import parse_training_config


class TestLeWMTrainingConfig(unittest.TestCase):
    def test_lewm_rejects_normalized_representations(self):
        with self.assertRaisesRegex(ValueError, "loss.normalize_reps=False"):
            parse_training_config(
                {
                    "lewm": {"enabled": True},
                    "loss": {"normalize_reps": True},
                }
            )

    def test_lewm_method_marker_rejects_normalized_representations(self):
        with self.assertRaisesRegex(ValueError, "loss.normalize_reps=False"):
            parse_training_config(
                {
                    "method": "lewm",
                    "loss": {"normalize_reps": True},
                }
            )

    def test_lewm_accepts_disabled_representation_normalization(self):
        config = parse_training_config(
            {
                "lewm": {"enabled": True},
                "loss": {"normalize_reps": False},
            }
        )

        self.assertTrue(config.world_model.enabled)
        self.assertFalse(config.loss.normalize_reps)
