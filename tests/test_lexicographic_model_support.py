import copy
import unittest
from pathlib import Path

import torch
import yaml

from train.config_validation import validate_config
from train.lexicographic.config import validate_lexicographic_requirements


REPO_ROOT = Path(__file__).resolve().parents[1]
LEX_CFG = {
    "enabled": True,
    "start_epoch": 0,
    "projection_mode": "coarse_first",
    "projection_rule": "orthogonalize_all",
    "eps": 1.0e-12,
    "log_metrics": True,
}


class LexicographicModelSupportTests(unittest.TestCase):
    def _load(self, relative_path):
        with (REPO_ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _with_lex(self, relative_path):
        cfg = copy.deepcopy(self._load(relative_path))
        cfg["train"]["lexicographic"] = copy.deepcopy(LEX_CFG)
        return cfg

    def test_native_supported_models_pass_static_validation(self):
        hcast = self._with_lex("configs/hcast/hcast_lex_cifar100.yaml")
        validate_config(hcast)

        hiercos = self._with_lex("configs/hiercos/hiercos_cifar100.yaml")
        hiercos["model"]["loss"] = "level_softmax_ce_reg"
        validate_config(hiercos)

        capsnet = self._with_lex("configs/capsnet/capsnet_cifar100.yaml")
        capsnet["model"]["loss"]["weight_mode"] = "none"
        validate_config(capsnet)
        capsnet["model"]["loss"]["weight_mode"] = "dynamic"
        validate_config(capsnet)

        hrn = self._with_lex("configs/hrn/hrn_cifar100.yaml")
        hrn["model"]["loss"] = "level_marginal"
        validate_config(hrn)

    def test_native_hrn_lex_requires_level_marginal_mode(self):
        hrn = self._with_lex("configs/hrn/hrn_cifar100.yaml")
        with self.assertRaisesRegex(ValueError, "model.loss=level_marginal"):
            validate_config(hrn)

    def test_lhdnn_lex_is_rejected_with_or_without_plugin(self):
        lhdnn = self._with_lex("configs/lhdnn/lhdnn_cifar100.yaml")
        with self.assertRaisesRegex(ValueError, "not supported for LH-DNN"):
            validate_config(lhdnn)

        lhdnn_with_plugin = copy.deepcopy(lhdnn)
        lhdnn_with_plugin["orthonormal_plugin"] = {
            "enabled": True,
            "loss": "level_softmax_ce_reg",
            "weight_mode": "equal",
            "alpha": 0.05,
            "transform_mode": "final_only",
            "fixed_frame_mode": "identity",
            "fixed_frame_per_level": True,
            "transform_lr_scale": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "not supported for LH-DNN"):
            validate_config(lhdnn_with_plugin)

    def test_runtime_validation_matches_static_model_policy(self):
        level_losses = [torch.ones((), requires_grad=True) for _ in range(3)]

        class Config:
            def __init__(self, model, plugin=None):
                self.model = model
                self.orthonormal_plugin = plugin or {"enabled": False}

        validate_lexicographic_requirements(
            Config({"name": "ht_capsnet", "loss": {"weight_mode": "none"}}),
            level_losses,
        )
        validate_lexicographic_requirements(
            Config({"name": "hrn", "loss": "level_marginal"}),
            level_losses,
        )
        with self.assertRaisesRegex(ValueError, "level_marginal"):
            validate_lexicographic_requirements(
                Config({"name": "hrn", "loss": "native"}),
                level_losses,
            )
        with self.assertRaisesRegex(ValueError, "not supported for LH-DNN"):
            validate_lexicographic_requirements(
                Config(
                    {"name": "lhdnn"},
                    {"enabled": True, "loss": "level_softmax_ce_reg"},
                ),
                level_losses,
            )


if __name__ == "__main__":
    unittest.main()
