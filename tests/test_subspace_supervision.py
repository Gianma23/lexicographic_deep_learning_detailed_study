import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from omegaconf import OmegaConf

from models import compute_loss as compute_model_loss
from models.common.subspace_supervision import (
    SUBSPACE_COORDINATES_KEY,
    SUBSPACE_PATH_OVERLAP_KEY,
    SUBSPACE_SCORES_KEY,
    compute_subspace_supervision_loss,
    subspace_norms,
    subspace_target_profile,
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


def _supervision_cfg(
    enabled: bool = True,
    model_name: str = "synthetic",
    weight_mode: str = "kl_leaf",
    **overrides,
):
    block = {"enabled": enabled}
    block.update(overrides)
    return SimpleNamespace(
        model={"name": model_name, "weight_mode": weight_mode},
        train={"subspace_supervision": block},
    )


def _topology(num_classes=None, taxonomy=None):
    return build_topology(num_classes or NUM_CLASSES, taxonomy=taxonomy or TAXONOMY)


def _profiles(topology):
    depth = len(topology["level_path_overlap"])
    return [
        subspace_target_profile(overlap, depth)
        for overlap in topology["level_path_overlap"]
    ]


class SubspaceTargetProfileTests(unittest.TestCase):
    def test_profile_is_shared_path_fraction_of_the_lowest_common_ancestor(self):
        topology = _topology()
        profiles = _profiles(topology)
        depth = len(NUM_CLASSES)

        # Leaf 0 has path (0, 0, 0); leaf 1 is its sibling, leaf 2 a cousin under
        # the same coarse ancestor, leaf 4 shares nothing with it.
        leaf_profile = profiles[2][0].double()
        self.assertAlmostEqual(float(leaf_profile[0]), 1.0, places=6)
        self.assertAlmostEqual(
            float(leaf_profile[1]), (2.0 / depth) ** 0.5, places=6
        )
        self.assertAlmostEqual(
            float(leaf_profile[2]), (1.0 / depth) ** 0.5, places=6
        )
        self.assertAlmostEqual(float(leaf_profile[4]), 0.0, places=6)
        # Monotone in taxonomic proximity, and never a one-hot.
        self.assertGreater(float(leaf_profile[1]), float(leaf_profile[2]))
        self.assertGreater(float(leaf_profile[2]), float(leaf_profile[4]))

    def test_true_class_is_the_strict_argmax_at_every_level(self):
        topology = _topology()
        profiles = _profiles(topology)
        paths = topology["leaf_to_level_local"]
        for leaf in range(int(paths.size(0))):
            for level in range(len(NUM_CLASSES)):
                row = profiles[level][leaf]
                self.assertEqual(int(row.argmax()), int(paths[leaf, level]))
                self.assertEqual(int((row == row.max()).sum()), 1)

    def test_profile_matches_the_masked_norm_of_the_intended_geometry(self):
        topology = _topology()
        profiles = _profiles(topology)
        energy = torch.full((len(NUM_CLASSES),), 1.0 / len(NUM_CLASSES), dtype=torch.float64)
        coordinates = torch.zeros(
            (int(topology["leaf_to_level_local"].size(0)), topology["total_nodes"]),
            dtype=torch.float64,
        )
        for leaf in range(int(coordinates.size(0))):
            for level in range(len(NUM_CLASSES)):
                node = int(
                    topology["level_node_ids"][level][
                        int(topology["leaf_to_level_local"][leaf, level])
                    ]
                )
                coordinates[leaf, node] = energy[level].sqrt()
        expected = subspace_norms(coordinates, topology["level_subspace_masks"])
        for level, profile in enumerate(profiles):
            torch.testing.assert_close(
                profile.double(), expected[level], rtol=1e-6, atol=1e-6
            )

    def test_rejects_invalid_depth(self):
        topology = _topology()
        for depth in (0, -1, 2.5, True):
            with self.subTest(depth=depth):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    subspace_target_profile(topology["level_path_overlap"][2], depth)

    def test_rejects_overlap_counts_above_the_depth(self):
        topology = _topology()
        overlap = topology["level_path_overlap"][2].clone()
        overlap[0, 0] = len(NUM_CLASSES) + 1
        with self.assertRaisesRegex(ValueError, "must lie in"):
            subspace_target_profile(overlap, len(NUM_CLASSES))

    def test_profile_depends_on_the_taxonomy_alone(self):
        """The target carries no level weighting: it is fixed by how much of the
        ground-truth path a subspace contains, so a sibling's target is always
        sqrt((L - 1) / L) whatever `model.weight_mode` resolves to."""
        topology = _topology()
        depth = len(NUM_CLASSES)
        profile = subspace_target_profile(topology["level_path_overlap"][2], depth)
        self.assertAlmostEqual(float(profile[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(
            float(profile[0, 1]), ((depth - 1) / depth) ** 0.5, places=6
        )

    def test_path_overlap_counts_shared_path_levels(self):
        topology = _topology()
        # Leaf 0 path is (0, 0, 0). At the finest level: itself sees all three
        # levels, its sibling two, a cousin one, an unrelated leaf none.
        overlap = topology["level_path_overlap"][2][0]
        self.assertEqual([int(v) for v in overlap], [3, 2, 1, 1, 0])


class SharedSubspaceLossTests(unittest.TestCase):
    def _loss_inputs(self, scores=None):
        topology = _topology()
        if scores is None:
            scores = [
                torch.tensor([[2.0, 0.0], [0.0, 3.0]], requires_grad=True),
                torch.tensor([[0.0, 1.0, 2.0], [3.0, 1.0, 0.0]], requires_grad=True),
                torch.tensor(
                    [[0.0, 1.0, 4.0, 2.0, 0.0], [1.0, 0.0, 2.0, 0.0, 5.0]],
                    requires_grad=True,
                ),
            ]
        output = {
            SUBSPACE_SCORES_KEY: scores,
            SUBSPACE_PATH_OVERLAP_KEY: topology["level_path_overlap"],
            SUBSPACE_COORDINATES_KEY: torch.full((2, topology["total_nodes"]), 0.5),
            "hiercos_level_node_ids": topology["level_node_ids"],
            "node_prob_weights": topology["node_prob_weights"],
        }
        return topology, output

    def test_soft_target_matches_the_tempered_induced_profile(self):
        topology, output = self._loss_inputs()
        hard_targets = torch.tensor([[0, 2, 4], [1, 0, 0]], dtype=torch.long)
        # Targets must be a consistent path, so rebuild them from the topology.
        paths = topology["leaf_to_level_local"]
        hard_targets = torch.stack([paths[4], paths[0]]).long()
        tau = 0.4

        loss, metrics, aux = compute_subspace_supervision_loss(
            output,
            hard_targets,
            _supervision_cfg(tau=tau, weight_mode="equal"),
            return_aux=True,
        )

        expected_raw = []
        # The target is weight-free, so it is the same profile the loss builds.
        profiles = _profiles(topology)
        for level, scores in enumerate(output[SUBSPACE_SCORES_KEY]):
            targets = profiles[level].index_select(0, hard_targets[:, -1])
            scale = output[SUBSPACE_COORDINATES_KEY].norm(dim=1, keepdim=True)
            q = torch.softmax(targets / tau, dim=1)
            log_p = torch.log_softmax(scores / scale / tau, dim=1)
            expected_raw.append(-(q * log_p).sum(dim=1).mean())
            # The supervised distribution is genuinely soft.
            self.assertLess(float(q.max()), 0.999)

        expected_total = torch.stack(expected_raw).mean()
        torch.testing.assert_close(loss, expected_total)
        self.assertAlmostEqual(
            metrics["subspace_soft_cross_entropy"], float(expected_total.item()), places=6
        )
        for level, raw_loss in enumerate(expected_raw):
            self.assertAlmostEqual(
                metrics[f"subspace_soft_cross_entropy_level_{level}"],
                float(raw_loss.item()),
                places=6,
            )
            # `subspace_soft_cross_entropy_losses` are the raw per-level terms;
            # `level_losses` are their weighted contributions, which sum to the
            # total -- the same contract the native Hier-COS losses expose.
            torch.testing.assert_close(
                aux["subspace_soft_cross_entropy_losses"][level], raw_loss
            )
            torch.testing.assert_close(
                aux["level_losses"][level], raw_loss / len(NUM_CLASSES)
            )
        torch.testing.assert_close(torch.stack(aux["level_losses"]).sum(), loss)

    def test_intended_geometry_reaches_the_attainable_floor(self):
        topology = _topology()
        profiles = _profiles(topology)
        paths = topology["leaf_to_level_local"]
        # Feed back exactly the profile the intended node geometry induces.
        hard_targets = torch.stack([paths[0], paths[3]]).long()
        scores = [
            profile.index_select(0, hard_targets[:, -1]).clone()
            for profile in profiles
        ]
        output = {
            SUBSPACE_SCORES_KEY: scores,
            SUBSPACE_PATH_OVERLAP_KEY: topology["level_path_overlap"],
            SUBSPACE_COORDINATES_KEY: torch.ones(2, topology["total_nodes"])
            / (topology["total_nodes"] ** 0.5),
            "hiercos_level_node_ids": topology["level_node_ids"],
            "node_prob_weights": topology["node_prob_weights"],
        }
        _loss, metrics = compute_subspace_supervision_loss(
            output, hard_targets, _supervision_cfg(tau=0.25)
        )
        self.assertAlmostEqual(metrics["subspace_target_kl"], 0.0, places=6)
        for level in range(len(NUM_CLASSES)):
            self.assertAlmostEqual(
                metrics[f"subspace_target_kl_level_{level}"], 0.0, places=6
            )

    def test_loss_is_invariant_to_a_uniform_rescaling_of_the_coordinates(self):
        topology, output = self._loss_inputs()
        hard_targets = torch.stack(
            [topology["leaf_to_level_local"][2], topology["leaf_to_level_local"][1]]
        ).long()
        cfg = _supervision_cfg(tau=0.3)
        base, _ = compute_subspace_supervision_loss(output, hard_targets, cfg)
        # Scaling the coordinates scales every subspace norm by the same factor,
        # which is the one degree of freedom the score map is homogeneous in.
        scaled_output = dict(output)
        scaled_output[SUBSPACE_SCORES_KEY] = [
            17.0 * scores for scores in output[SUBSPACE_SCORES_KEY]
        ]
        scaled_output[SUBSPACE_COORDINATES_KEY] = 17.0 * output[SUBSPACE_COORDINATES_KEY]
        scaled, _ = compute_subspace_supervision_loss(scaled_output, hard_targets, cfg)
        torch.testing.assert_close(base, scaled)

    def test_per_level_rescaling_is_not_invariant(self):
        """A shared scale is what identifies the node energies: scaling one
        level alone must change the loss, or that level's energy is free."""
        topology, output = self._loss_inputs()
        hard_targets = torch.stack(
            [topology["leaf_to_level_local"][2], topology["leaf_to_level_local"][1]]
        ).long()
        cfg = _supervision_cfg(tau=0.3)
        base, _ = compute_subspace_supervision_loss(output, hard_targets, cfg)
        skewed = dict(output)
        skewed[SUBSPACE_SCORES_KEY] = [
            3.0 * s if level == 0 else s
            for level, s in enumerate(output[SUBSPACE_SCORES_KEY])
        ]
        other, _ = compute_subspace_supervision_loss(skewed, hard_targets, cfg)
        self.assertNotAlmostEqual(
            float(base.detach()), float(other.detach()), places=4
        )

    def test_missing_coordinates_fail_explicitly(self):
        _topology_unused, output = self._loss_inputs()
        hard_targets = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long)
        del output[SUBSPACE_COORDINATES_KEY]
        with self.assertRaisesRegex(ValueError, SUBSPACE_COORDINATES_KEY):
            compute_subspace_supervision_loss(output, hard_targets, _supervision_cfg())

    def test_loss_weights_follow_model_weight_mode(self):
        from models.hiercos.losses import resolve_level_weights

        topology, output = self._loss_inputs()
        hard_targets = torch.stack(
            [topology["leaf_to_level_local"][0], topology["leaf_to_level_local"][4]]
        ).long()
        for mode in ("kl_leaf", "equal", "kl_coarse", "marginal_branching"):
            with self.subTest(mode=mode):
                cfg = _supervision_cfg(weight_mode=mode)
                _loss, metrics = compute_subspace_supervision_loss(
                    output, hard_targets, cfg
                )
                expected = resolve_level_weights(
                    output=output,
                    cfg=cfg,
                    level_node_ids=topology["level_node_ids"],
                    num_levels=3,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                expected = expected / expected.sum()
                for level, weight in enumerate(expected):
                    self.assertAlmostEqual(
                        metrics[f"loss_weight_level_{level}"],
                        float(weight),
                        places=6,
                    )
        # The modes are genuinely distinct, so the arm is not silently uniform.
        _loss, kl_metrics = compute_subspace_supervision_loss(
            output, hard_targets, _supervision_cfg(weight_mode="kl_leaf")
        )
        self.assertNotAlmostEqual(
            kl_metrics["loss_weight_level_0"], 1.0 / 3.0, places=3
        )

    def test_total_is_the_weighted_sum_of_the_level_cross_entropies(self):
        """The weights are the coefficients of the scalarisation, exactly as in
        the native Hier-COS softmax losses. They sum to one, so `equal` reduces
        to the plain mean and the loss scale is unchanged."""
        topology, output = self._loss_inputs()
        hard_targets = torch.stack(
            [topology["leaf_to_level_local"][0], topology["leaf_to_level_local"][4]]
        ).long()
        loss, metrics = compute_subspace_supervision_loss(
            output, hard_targets, _supervision_cfg(weight_mode="kl_leaf")
        )
        levels = range(len(NUM_CLASSES))
        weights = [metrics[f"loss_weight_level_{level}"] for level in levels]
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
        for level in levels:
            self.assertAlmostEqual(
                metrics[f"loss_level_{level}"],
                weights[level] * metrics[f"subspace_soft_cross_entropy_level_{level}"],
                places=6,
            )
        self.assertAlmostEqual(
            float(loss.detach()),
            sum(metrics[f"loss_level_{level}"] for level in levels),
            places=6,
        )
        # Under `equal` the scalarisation is the uniform mean it replaced.
        _loss, equal_metrics = compute_subspace_supervision_loss(
            output, hard_targets, _supervision_cfg(weight_mode="equal")
        )
        per_level = [
            equal_metrics[f"subspace_soft_cross_entropy_level_{level}"]
            for level in levels
        ]
        self.assertAlmostEqual(
            equal_metrics["total"], sum(per_level) / len(per_level), places=6
        )

    def test_weight_mode_leaves_the_target_geometry_untouched(self):
        """The per-level divergences measure distance to the target. They must be
        identical across weight modes, because the target is weight-free."""
        topology, output = self._loss_inputs()
        hard_targets = torch.stack(
            [topology["leaf_to_level_local"][0], topology["leaf_to_level_local"][4]]
        ).long()
        _loss, kl_metrics = compute_subspace_supervision_loss(
            output, hard_targets, _supervision_cfg(weight_mode="kl_leaf")
        )
        _loss, coarse_metrics = compute_subspace_supervision_loss(
            output, hard_targets, _supervision_cfg(weight_mode="kl_coarse")
        )
        for level in range(len(NUM_CLASSES)):
            self.assertAlmostEqual(
                kl_metrics[f"subspace_target_kl_level_{level}"],
                coarse_metrics[f"subspace_target_kl_level_{level}"],
                places=6,
            )

    def test_divergence_vanishes_at_the_intended_geometry(self):
        topology, output = self._loss_inputs()
        hard_targets = torch.stack(
            [topology["leaf_to_level_local"][0], topology["leaf_to_level_local"][3]]
        ).long()
        # Feed the induced profile back in as the scores: the divergence is the
        # part of the loss the optimizer can remove, so it must be zero there --
        # and, since the target is weight-free, zero under every weight mode.
        scores = [
            profile.index_select(0, hard_targets[:, -1]).clone()
            for profile in _profiles(topology)
        ]
        # Unit-norm coordinates so the shared scale is 1 and the scores are
        # exactly the induced profile.
        matched = dict(
            output,
            **{
                SUBSPACE_SCORES_KEY: scores,
                SUBSPACE_COORDINATES_KEY: torch.eye(
                    2, topology["total_nodes"]
                ),
            },
        )
        for mode in ("kl_leaf", "kl_coarse", "equal"):
            with self.subTest(mode=mode):
                _loss, metrics = compute_subspace_supervision_loss(
                    matched, hard_targets, _supervision_cfg(weight_mode=mode)
                )
                self.assertAlmostEqual(metrics["subspace_target_kl"], 0.0, places=6)

    def test_missing_path_overlap_fails_explicitly(self):
        _topology_unused, output = self._loss_inputs()
        hard_targets = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long)
        del output[SUBSPACE_PATH_OVERLAP_KEY]
        with self.assertRaisesRegex(ValueError, SUBSPACE_PATH_OVERLAP_KEY):
            compute_subspace_supervision_loss(output, hard_targets, _supervision_cfg())

    def test_inconsistent_targets_fail_explicitly(self):
        _topology_unused, output = self._loss_inputs()
        # Leaf 4 does not sit under coarse class 0, so the path is inconsistent.
        hard_targets = torch.tensor([[0, 0, 4], [0, 0, 4]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "inconsistent with hard targets"):
            compute_subspace_supervision_loss(output, hard_targets, _supervision_cfg())

    def test_rejects_invalid_tau(self):
        _topology_unused, output = self._loss_inputs()
        hard_targets = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "tau"):
            compute_subspace_supervision_loss(
                output, hard_targets, _supervision_cfg(tau=0.0)
            )

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
        malformed = {SUBSPACE_SCORES_KEY: [torch.ones(2)]}
        with self.assertRaisesRegex(ValueError, "shape"):
            compute_model_loss(_supervision_cfg(), malformed, torch.tensor([[0]]))

    def test_soft_targets_are_rejected_at_runtime(self):
        output = {SUBSPACE_SCORES_KEY: [torch.ones(1, width) for width in NUM_CLASSES]}
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

    def test_hiercos_contract_aliases_evaluated_scores(self):
        model = self._build_model().eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 32, 32))
        self.assertIs(output[SUBSPACE_SCORES_KEY], output["logits_per_level"])

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

    def test_enabled_subspace_loss_ignores_native_loss_and_alpha(self):
        model = self._build_model().eval()
        targets = torch.tensor([[0, 0, 0], [1, 2, 4]], dtype=torch.long)
        with torch.no_grad():
            output = model(torch.randn(2, 3, 32, 32))

        cfg_a = _supervision_cfg(model_name="hiercos")
        cfg_a.model.update(loss="kl_reg", weight_mode="kl_leaf", alpha=0.0)
        cfg_b = _supervision_cfg(model_name="hiercos")
        cfg_b.model.update(loss="level_softmax_ce_reg", weight_mode="kl_leaf", alpha=99.0)

        loss_a, metrics_a = compute_model_loss(cfg_a, output, targets)
        loss_b, metrics_b = compute_model_loss(cfg_b, output, targets)
        torch.testing.assert_close(loss_a, loss_b)
        self.assertEqual(metrics_a, metrics_b)

    def test_enabled_subspace_loss_honours_model_weight_mode(self):
        """`weight_mode` is deliberately NOT ignored: it sets the per-level
        coefficients of the scalarisation, the same role it has in the native
        Hier-COS softmax losses, so the arm matches its baseline."""
        model = self._build_model().eval()
        targets = torch.tensor([[0, 0, 0], [1, 2, 4]], dtype=torch.long)
        with torch.no_grad():
            output = model(torch.randn(2, 3, 32, 32))

        cfg_a = _supervision_cfg(model_name="hiercos")
        cfg_a.model.update(loss="kl_reg", weight_mode="kl_leaf", alpha=0.0)
        cfg_b = _supervision_cfg(model_name="hiercos")
        cfg_b.model.update(loss="kl_reg", weight_mode="equal", alpha=0.0)

        loss_a, metrics_a = compute_model_loss(cfg_a, output, targets)
        loss_b, metrics_b = compute_model_loss(cfg_b, output, targets)
        self.assertNotAlmostEqual(float(loss_a), float(loss_b), places=4)
        self.assertAlmostEqual(
            metrics_b["loss_weight_level_0"], 1.0 / 3.0, places=6
        )
        self.assertNotAlmostEqual(
            metrics_a["loss_weight_level_0"], 1.0 / 3.0, places=3
        )


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
        cfg["train"]["subspace_supervision"] = {"enabled": True}
        return cfg

    def test_valid_enabled_configuration(self):
        validate_config(self._enabled())

    def test_accepts_tau_settings(self):
        for block in (
            {"enabled": True, "tau": 0.15},
            {"enabled": True, "tau": 0.6},
            {"enabled": True, "eps": 1e-10},
        ):
            with self.subTest(block=block):
                cfg = self._enabled()
                cfg["train"]["subspace_supervision"] = block
                validate_config(cfg)

    def test_rejects_invalid_tau(self):
        for key, value in (("tau", 0.0), ("tau", -1.0)):
            with self.subTest(key=key, value=value):
                cfg = self._enabled()
                cfg["train"]["subspace_supervision"][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    validate_config(cfg)

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

    def test_rejects_removed_subspace_options(self):
        removed_options = {
            "loss": "cross_entropy",
            "target_mode": "sqrt_path_weights",
            "temperature": 1.0,
            "level_weights": "equal",
        }
        for key, value in removed_options.items():
            with self.subTest(key=key):
                cfg = self._enabled()
                cfg["train"]["subspace_supervision"][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    validate_config(cfg)

if __name__ == "__main__":
    unittest.main()
