import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from omegaconf import OmegaConf

from models import compute_loss as compute_model_loss
from models.common.subspace_supervision import (
    SUBSPACE_SCORES_KEY,
    SUBSPACE_TARGET_PROFILES_KEY,
    build_sqrt_path_target_profiles,
    compute_subspace_supervision_loss,
    subspace_norms,
)
from models.hiercos.losses import compute_loss as compute_hiercos_loss
from models.hiercos.model import HierCosModel
from models.hiercos.topology import build_topology
from train.config_validation import validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = [2, 3, 5]
TAXONOMY = {
    "parent_of": {
        1: {0: 0, 1: 0, 2: 1},
        2: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2},
    }
}


def _supervision_cfg(enabled: bool = True, model_name: str = "synthetic"):
    return SimpleNamespace(
        model={"name": model_name},
        train={
            "subspace_supervision": {
                "enabled": enabled,
                "target_mode": "sqrt_path_weights",
                "loss": "normalized_mse",
                "eps": 1e-12,
            }
        },
    )


def _toy_profiles():
    topology = build_topology(NUM_CLASSES, TAXONOMY, owner="test")
    profiles = build_sqrt_path_target_profiles(
        level_node_ids=topology["level_node_ids"],
        level_subspace_masks=topology["level_subspace_masks"],
        leaf_to_level_local=topology["leaf_to_level_local"],
        path_weights=topology["node_prob_weights"],
    )
    return topology, profiles


class SubspaceTargetProfileTests(unittest.TestCase):
    def test_dense_profiles_encode_each_shared_ancestry_case(self):
        topology, profiles = _toy_profiles()
        weights = topology["node_prob_weights"].double()

        # Leaf 0 and leaf 1 share coarse and middle nodes; leaves 2/3 share
        # only the coarse node; leaf 4 is in a disjoint coarse branch.
        self.assertAlmostEqual(float(profiles[2][0, 0]), 1.0, places=6)
        self.assertAlmostEqual(
            float(profiles[2][0, 1]),
            float(torch.sqrt(weights[0] + weights[1])),
            places=6,
        )
        self.assertAlmostEqual(
            float(profiles[2][0, 2]),
            float(torch.sqrt(weights[0])),
            places=6,
        )
        self.assertAlmostEqual(float(profiles[2][0, 4]), 0.0, places=7)

        self.assertAlmostEqual(float(profiles[1][0, 0]), 1.0, places=6)
        self.assertAlmostEqual(
            float(profiles[1][0, 1]),
            float(torch.sqrt(weights[0])),
            places=6,
        )
        self.assertAlmostEqual(float(profiles[1][0, 2]), 0.0, places=7)

    def test_true_class_is_the_target_argmax_at_every_level(self):
        topology, profiles = _toy_profiles()
        expected = topology["leaf_to_level_local"]
        for level, profile in enumerate(profiles):
            self.assertTrue(torch.equal(profile.argmax(dim=1), expected[:, level]))


class SharedSubspaceLossTests(unittest.TestCase):
    def test_normalized_mse_is_scale_invariant_for_synthetic_model(self):
        topology, profiles = _toy_profiles()
        leaf_targets = torch.tensor([0, 1, 4], dtype=torch.long)
        hard_targets = topology["leaf_to_level_local"].index_select(0, leaf_targets)
        selected = [profile.index_select(0, leaf_targets) for profile in profiles]

        output_a = {
            SUBSPACE_SCORES_KEY: [score.clone().requires_grad_() for score in selected],
            SUBSPACE_TARGET_PROFILES_KEY: profiles,
        }
        output_b = {
            SUBSPACE_SCORES_KEY: [
                score * torch.tensor([[0.25], [2.0], [11.0]])
                for score in selected
            ],
            SUBSPACE_TARGET_PROFILES_KEY: profiles,
        }
        loss_a, _ = compute_model_loss(_supervision_cfg(), output_a, hard_targets)
        loss_b, _ = compute_model_loss(_supervision_cfg(), output_b, hard_targets)
        torch.testing.assert_close(loss_a, loss_b, rtol=1e-6, atol=1e-7)
        self.assertLess(float(loss_a.item()), 1e-12)

    def test_level_errors_are_summed_over_classes_then_averaged_equally(self):
        lookups = [torch.eye(2) for _ in range(3)]
        output = {
            SUBSPACE_SCORES_KEY: [
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[0.0, 1.0]]),
                torch.tensor([[1.0, 1.0]]),
            ],
            SUBSPACE_TARGET_PROFILES_KEY: lookups,
        }
        targets = torch.tensor([[0, 0, 0]], dtype=torch.long)
        loss, metrics, aux = compute_subspace_supervision_loss(
            output,
            targets,
            _supervision_cfg(),
            return_aux=True,
        )
        expected_raw = [0.0, 2.0, 2.0 - 2.0**0.5]
        expected_total = sum(expected_raw) / 3.0
        self.assertAlmostEqual(float(loss.item()), expected_total, places=6)
        self.assertAlmostEqual(metrics["subspace_profile_mse"], expected_total, places=6)
        for level, expected in enumerate(expected_raw):
            self.assertAlmostEqual(
                metrics[f"subspace_profile_mse_level_{level}"],
                expected,
                places=6,
            )
            self.assertAlmostEqual(
                float(aux["level_losses"][level].item()),
                expected / 3.0,
                places=6,
            )
        self.assertIn("subspace_score_l2", metrics)

    def test_zero_subspace_has_finite_zero_gradient(self):
        coordinates = torch.zeros((2, 4), requires_grad=True)
        masks = [
            torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool),
            torch.ones((1, 4), dtype=torch.bool),
        ]
        scores = subspace_norms(coordinates, masks)
        torch.stack([score.sum() for score in scores]).sum().backward()
        self.assertTrue(bool(torch.isfinite(coordinates.grad).all()))
        self.assertTrue(torch.equal(coordinates.grad, torch.zeros_like(coordinates)))

    def test_shared_helper_preserves_exact_forward_values(self):
        coordinates = torch.tensor([[1.0, -2.0, 3.0, 0.0]], requires_grad=True)
        masks = [torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.bool)]
        actual = subspace_norms(coordinates, masks)[0]
        expected = torch.sqrt(coordinates.pow(2) @ masks[0].float().transpose(0, 1))
        self.assertTrue(torch.equal(actual, expected))

    def test_shared_helper_keeps_float32_accumulation_under_autocast(self):
        coordinates = torch.tensor([[1.0, -2.0, 3.0, 0.0]], requires_grad=True)
        masks = [torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.bool)]
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = subspace_norms(coordinates, masks)[0]
        self.assertEqual(actual.dtype, torch.float32)
        expected = torch.sqrt(coordinates.pow(2) @ masks[0].float().transpose(0, 1))
        self.assertTrue(torch.equal(actual, expected))
        actual.sum().backward()
        self.assertTrue(bool(torch.isfinite(coordinates.grad).all()))

    def test_missing_or_malformed_contract_fails_explicitly(self):
        targets = torch.tensor([[0, 0, 0]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, SUBSPACE_SCORES_KEY):
            compute_model_loss(_supervision_cfg(), {}, targets)
        malformed = {
            SUBSPACE_SCORES_KEY: [torch.ones(1, 2)],
            SUBSPACE_TARGET_PROFILES_KEY: [torch.ones(2, 3)],
        }
        with self.assertRaisesRegex(ValueError, "width mismatch"):
            compute_model_loss(
                _supervision_cfg(),
                malformed,
                torch.tensor([[0]], dtype=torch.long),
            )

    def test_soft_targets_are_rejected_at_runtime(self):
        _, profiles = _toy_profiles()
        output = {
            SUBSPACE_SCORES_KEY: [profile[:1].clone() for profile in profiles],
            SUBSPACE_TARGET_PROFILES_KEY: profiles,
        }
        targets = {
            "hard_targets": torch.tensor([[0, 0, 0]], dtype=torch.long),
            "soft_targets_per_level": [torch.ones(1, width) for width in NUM_CLASSES],
        }
        with self.assertRaisesRegex(ValueError, "soft targets"):
            compute_model_loss(_supervision_cfg(), output, targets)


class HierCosSubspaceSupervisionTests(unittest.TestCase):
    @staticmethod
    def _build_model() -> HierCosModel:
        torch.manual_seed(17)
        return HierCosModel(
            num_classes_per_level=NUM_CLASSES,
            taxonomy=TAXONOMY,
            variant="haframe_wide_resnet",
            transform_mode="full",
            pretrained=False,
            fixed_frame_mode="orthonormal_random",
            fixed_frame_per_level=False,
            wide_depth=10,
            wide_widen_factor=1,
            wide_drop_rate=0.0,
        )

    def test_hiercos_contract_aliases_evaluated_scores_and_uses_nonpersistent_targets(self):
        model = self._build_model().eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 32, 32))
        self.assertIs(output[SUBSPACE_SCORES_KEY], output["logits_per_level"])
        self.assertEqual(len(output[SUBSPACE_TARGET_PROFILES_KEY]), len(NUM_CLASSES))
        self.assertFalse(
            any(key.startswith("subspace_target_profile_") for key in model.state_dict())
        )

    def test_disabled_override_preserves_native_loss(self):
        model = self._build_model().eval()
        targets = torch.tensor([[0, 0, 0], [1, 2, 4]], dtype=torch.long)
        with torch.no_grad():
            output = model(torch.randn(2, 3, 32, 32))
        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "hiercos",
                    "loss": "kl_reg",
                    "weight_mode": "kl_leaf",
                    "alpha": 0.05,
                },
                "train": {"subspace_supervision": {"enabled": False}},
            }
        )
        expected_loss, expected_metrics = compute_hiercos_loss(output, targets, cfg)
        actual_loss, actual_metrics = compute_model_loss(cfg, output, targets)
        self.assertTrue(torch.equal(expected_loss, actual_loss))
        self.assertEqual(expected_metrics, actual_metrics)

    def test_loss_gradients_reach_transform_and_backbone(self):
        model = self._build_model().train()
        targets = torch.tensor([[0, 0, 0], [1, 2, 4]], dtype=torch.long)
        output = model(torch.randn(2, 3, 32, 32))
        loss, metrics = compute_model_loss(
            _supervision_cfg(model_name="hiercos"),
            output,
            targets,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

        for module in (model.f_theta, model.backbone):
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(any(bool((gradient != 0).any()) for gradient in gradients))
            self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))


class SubspaceSupervisionConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (REPO_ROOT / "configs/hiercos/hiercos_cifar100.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            cls.base = yaml.safe_load(handle)

    def _enabled(self):
        cfg = copy.deepcopy(self.base)
        cfg["model"]["alpha"] = 0.0
        cfg["model"]["projection"] = {"enabled": False}
        cfg["train"]["subspace_supervision"] = {
            "enabled": True,
            "target_mode": "sqrt_path_weights",
            "loss": "normalized_mse",
            "eps": 1e-12,
        }
        return cfg

    def test_valid_enabled_configuration(self):
        validate_config(self._enabled())

    def test_rejects_mixup_cutmix_lexicographic_hcc_and_projection(self):
        mutations = {
            "smoothing": lambda cfg: cfg["train"].update(smoothing=0.1),
            "mixup": lambda cfg: cfg["dataset"]["transforms"].update(mixup=0.2),
            "cutmix": lambda cfg: cfg["dataset"]["transforms"].update(cutmix=1.0),
            "cutmix_minmax": lambda cfg: cfg["dataset"]["transforms"].update(
                cutmix_minmax=[0.2, 0.8]
            ),
            "lexicographic": lambda cfg: cfg["train"].update(
                lexicographic={"enabled": True}
            ),
            "hcc.enabled": lambda cfg: cfg.update(hcc={"enabled": True, "eps": 1e-12}),
            "LH projection": lambda cfg: cfg["model"].update(
                projection={"enabled": True},
                loss="level_softmax_ce_reg",
                fixed_frame_mode="identity",
            ),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message):
                cfg = self._enabled()
                mutate(cfg)
                with self.assertRaisesRegex(ValueError, message):
                    validate_config(cfg)

    def test_rejects_unknown_target_and_loss_modes(self):
        for key in ("target_mode", "loss"):
            with self.subTest(key=key):
                cfg = self._enabled()
                cfg["train"]["subspace_supervision"][key] = "unsupported"
                with self.assertRaisesRegex(ValueError, f"subspace_supervision.{key}"):
                    validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
