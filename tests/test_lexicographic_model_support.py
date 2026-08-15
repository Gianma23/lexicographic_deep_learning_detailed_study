import copy
import unittest
from pathlib import Path

import torch
import yaml

from train.config_validation import validate_config
from train.lexicographic.config import validate_lexicographic_requirements
from train.runtime.checkpointing import _validate_resume_config_or_raise


REPO_ROOT = Path(__file__).resolve().parents[1]
LEX_CFG = {
    "enabled": True,
    "projection_mode": "coarse_first",
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
        hcast = self._with_lex("configs/hcast/hcast_cifar100.yaml")
        hcast["model"]["loss"]["globalkl"] = False
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

    def test_removed_projection_rule_is_rejected_by_static_validation(self):
        hcast = self._with_lex("configs/hcast/hcast_cifar100.yaml")
        hcast["model"]["loss"]["globalkl"] = False
        hcast["train"]["lexicographic"]["projection_rule"] = "orthogonalize_all"

        with self.assertRaisesRegex(ValueError, "projection_rule"):
            validate_config(hcast)

    def test_lhdnn_lex_is_rejected(self):
        lhdnn = self._with_lex("configs/lhdnn/lhdnn_cifar100.yaml")
        with self.assertRaisesRegex(ValueError, "not supported for LH-DNN"):
            validate_config(lhdnn)

    def test_runtime_validation_matches_static_model_policy(self):
        level_losses = [torch.ones((), requires_grad=True) for _ in range(3)]

        class Config:
            def __init__(self, model):
                self.model = model

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
                Config({"name": "lhdnn"}),
                level_losses,
            )


class LegacyProjectionRuleResumeTests(unittest.TestCase):
    def _config(self):
        return {
            "model": {"name": "hcast"},
            "train": {
                "lexicographic": {
                    "enabled": True,
                    "projection_mode": "coarse_first",
                }
            },
        }

    def test_checkpoint_orthogonalize_all_rule_is_compatible(self):
        checkpoint_cfg = self._config()
        checkpoint_cfg["train"]["lexicographic"]["projection_rule"] = "orthogonalize_all"

        _validate_resume_config_or_raise(
            checkpoint_cfg_resolved=checkpoint_cfg,
            current_cfg_resolved=self._config(),
            resume_path="legacy.pt",
        )

    def test_other_checkpoint_projection_rules_remain_incompatible(self):
        for legacy_rule in ("legacy_conditional", "unknown_rule"):
            with self.subTest(projection_rule=legacy_rule):
                checkpoint_cfg = self._config()
                checkpoint_cfg["train"]["lexicographic"]["projection_rule"] = legacy_rule

                with self.assertRaisesRegex(ValueError, "projection_rule"):
                    _validate_resume_config_or_raise(
                        checkpoint_cfg_resolved=checkpoint_cfg,
                        current_cfg_resolved=self._config(),
                        resume_path="legacy.pt",
                    )

    def test_current_side_projection_rule_is_not_tombstoned(self):
        current_cfg = self._config()
        current_cfg["train"]["lexicographic"]["projection_rule"] = "orthogonalize_all"

        with self.assertRaisesRegex(ValueError, "projection_rule"):
            _validate_resume_config_or_raise(
                checkpoint_cfg_resolved=self._config(),
                current_cfg_resolved=current_cfg,
                resume_path="legacy.pt",
            )


if __name__ == "__main__":
    unittest.main()
