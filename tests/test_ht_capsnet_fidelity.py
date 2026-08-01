import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from models.ht_capsnet.losses import _margin_loss, compute_loss
from models.ht_capsnet.model import (
    HTCapsNet,
    KerasMultiHeadAttention,
    _build_backbone,
    _ConvBackbone,
)
from models.ht_capsnet.routing import (
    hierarchical_agreement,
    squash,
    taxonomy_guided_routing_weights,
)
from train.mixup import Mixup
from train.runtime.optimization import HTCapsNetExponentialScheduler


def _taxonomy():
    return {
        "parent_of": {
            1: {0: 0, 1: 0, 2: 1},
            2: {0: 0, 1: 1, 2: 1, 3: 2},
        }
    }


def _loss_cfg():
    return SimpleNamespace(
        model=SimpleNamespace(
            loss={
                "margin_m_pos": 0.9,
                "margin_m_neg": 0.1,
                "lambda_downweight": 0.5,
                "weight_mode": "dynamic",
                "dynamic_weight": 0.0,
            }
        )
    )


class KerasAttentionTests(unittest.TestCase):
    def test_sdpa_matches_explicit_scaled_dot_product_reference(self):
        torch.manual_seed(4)
        layer = KerasMultiHeadAttention(
            query_dim=6,
            context_dim=5,
            num_heads=2,
            key_dim=3,
            dropout=0.0,
        ).double()
        layer.eval()
        query = torch.randn(2, 4, 6, dtype=torch.float64, requires_grad=True)
        context = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)

        actual = layer(query, context, context)
        q = layer._split_heads(layer.query_projection(query))
        k = layer._split_heads(layer.key_projection(context))
        v = layer._split_heads(layer.value_projection(context))
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(3.0), dim=-1)
        reference = torch.matmul(weights, v).transpose(1, 2).contiguous().view(2, 4, 6)
        reference = layer.output_projection(reference)

        torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
        actual.square().mean().backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(context.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in layer.parameters()))

    def test_projection_width_is_independent_of_capsule_dimension(self):
        layer = KerasMultiHeadAttention(16, 32, num_heads=16, key_dim=32)
        self.assertEqual(layer.query_projection.out_features, 512)
        self.assertEqual(layer.key_projection.out_features, 512)
        self.assertEqual(layer.output_projection.out_features, 16)


class RoutingAndLossTests(unittest.TestCase):
    def test_squash_and_margin_loss_match_source_equations(self):
        capsules = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
        squared_norm = capsules.square().sum(dim=-1, keepdim=True)
        expected = squared_norm / (1.0 + squared_norm)
        expected = expected * capsules / torch.sqrt(squared_norm + 1e-8)
        torch.testing.assert_close(squash(capsules), expected)

        scores = torch.tensor([[0.8, 0.3, 0.05]])
        target = torch.tensor([0])
        expected_margin = (0.9 - 0.8) ** 2 + 0.5 * (0.3 - 0.1) ** 2
        self.assertAlmostEqual(
            float(_margin_loss(scores, target, 0.9, 0.1, 0.5)),
            expected_margin,
            places=7,
        )

    def test_taxonomy_mask_repeats_source_pattern_including_remainder(self):
        raw = torch.tensor([[[1.0, -1.0], [0.5, -0.5], [-0.25, 0.25]]])
        taxonomy = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        actual = taxonomy_guided_routing_weights(
            raw,
            level=1,
            taxonomy_matrix=taxonomy,
            taxonomy_temperature=0.5,
            mask_threshold_high=0.99,
            mask_threshold_low=0.1,
            mask_temperature=0.5,
            mask_center=0.5,
        )
        soft = 0.89 * torch.sigmoid(0.5 * (taxonomy - 0.5)) + 0.1
        extended = torch.stack([soft[0], soft[1], soft[0]], dim=0).unsqueeze(0)
        expected = torch.softmax(raw * extended * 0.5, dim=-1)
        torch.testing.assert_close(actual, expected)

        # When there are fewer input capsules than parents, upstream repeats
        # the first taxonomy row for the entire remainder.
        short_actual = taxonomy_guided_routing_weights(
            raw[:, :1], level=1, taxonomy_matrix=taxonomy
        )
        short_soft = 0.8 * torch.sigmoid(0.5 * (taxonomy - 0.5)) + 0.1
        short_expected = torch.softmax(raw[:, :1] * short_soft[:1].unsqueeze(0) * 0.5, dim=-1)
        torch.testing.assert_close(short_actual, short_expected)

    def test_hierarchical_agreement_matches_explicit_gate(self):
        votes = torch.tensor([[[[1.0, 2.0], [2.0, 1.0]]]])
        previous = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        transform = torch.eye(2)
        gate = torch.tensor([[[0.5, 0.25], [0.75, 0.5]]])
        actual = hierarchical_agreement(votes, previous, transform, gate)
        agreement = torch.matmul(votes.reshape(1, 2, 2), previous.transpose(1, 2))
        agreement = agreement.reshape(1, 1, 2, 2)
        expected = votes * torch.sigmoid((agreement * gate.unsqueeze(1)).sum(dim=-1, keepdim=True))
        torch.testing.assert_close(actual, expected)

    def test_unit_weight_mode_exposes_three_raw_margin_objectives(self):
        cfg = _loss_cfg()
        cfg.model.loss["weight_mode"] = "none"
        logits = [
            torch.randn(2, count, requires_grad=True)
            for count in (2, 3, 4)
        ]
        targets = torch.tensor([[0, 0, 0], [1, 2, 3]])

        total, metrics, aux = compute_loss(
            {"logits_per_level": logits},
            targets,
            cfg,
            return_aux=True,
        )
        self.assertEqual(len(aux["level_losses"]), 3)
        self.assertTrue(all(loss.ndim == 0 and loss.requires_grad for loss in aux["level_losses"]))
        torch.testing.assert_close(total, torch.stack(aux["level_losses"]).sum())
        for level in range(3):
            self.assertEqual(metrics[f"loss_weight_level_{level}"], 1.0)


class HTCapsNetStateTests(unittest.TestCase):
    def _model(self):
        return HTCapsNet(
            num_classes_per_level=[2, 3, 4],
            taxonomy=_taxonomy(),
            primary_dim=8,
            secondary_dims=[8, 4, 2],
            routing_iters=2,
            num_blocks=1,
            initial_filters=8,
            backbone_name="custom",
            backbone_weights=None,
            attn_heads=2,
            attn_key_dim=3,
            input_size=32,
        )

    def test_primary_capsules_follow_nhwc_flatten_order(self):
        model = self._model()
        feature = torch.arange(8 * 16 * 16, dtype=torch.float32).view(1, 8, 16, 16)
        actual = model._build_primary_caps(feature)
        expected_flat = feature.permute(0, 2, 3, 1).contiguous().view(1, -1, 8)
        torch.testing.assert_close(actual, squash(expected_flat, dim=-1))

    def test_custom_backbone_matches_keras_activation_bn_order_and_constants(self):
        backbone = _ConvBackbone(num_blocks=1, initial_filters=8)
        self.assertIsInstance(backbone.net[0], torch.nn.Conv2d)
        self.assertIsInstance(backbone.net[1], torch.nn.ReLU)
        self.assertIsInstance(backbone.net[2], torch.nn.BatchNorm2d)
        self.assertEqual(backbone.net[2].eps, 1e-3)
        self.assertEqual(backbone.net[2].momentum, 0.01)

    def test_requested_efficientnet_weights_never_fall_back(self):
        with patch("timm.create_model", side_effect=OSError("weights unavailable")):
            with self.assertRaisesRegex(RuntimeError, "never falls back"):
                _build_backbone("efficientnet_b7", "imagenet", 4, 64, 2)

    def test_inference_probabilities_are_normalized(self):
        model = self._model().eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 32, 32))
        for probabilities in output["effective_probs_per_level"]:
            torch.testing.assert_close(
                probabilities.sum(dim=-1),
                torch.ones(probabilities.size(0)),
            )

    def test_dynamic_weights_apply_to_next_batch_and_round_trip(self):
        model = self._model()
        initial = model.level_loss_weights.detach().clone()
        logits = [
            torch.tensor([[2.0, 0.0], [2.0, 0.0]], requires_grad=True),
            torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]], requires_grad=True),
            torch.tensor([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 2.0, 0.0]], requires_grad=True),
        ]
        output = {"logits_per_level": logits, "level_loss_weights": initial.clone()}
        targets = torch.zeros((2, 3), dtype=torch.long)

        _, metrics, aux = compute_loss(output, targets, _loss_cfg(), return_aux=True)
        for level, value in enumerate(initial):
            self.assertAlmostEqual(metrics[f"loss_weight_level_{level}"], float(value), places=7)
        torch.testing.assert_close(model.level_loss_weights, initial)

        model.post_optimizer_step(aux)
        self.assertFalse(torch.equal(model.level_loss_weights, initial))
        saved = model.state_dict()
        restored = self._model()
        restored.load_state_dict(saved)
        torch.testing.assert_close(restored.level_loss_weights, model.level_loss_weights)

        before_eval = restored.level_loss_weights.detach().clone()
        compute_loss(
            {"logits_per_level": logits, "level_loss_weights": before_eval.clone()},
            targets,
            _loss_cfg(),
            return_aux=False,
        )
        torch.testing.assert_close(restored.level_loss_weights, before_eval)


class MixupAndSchedulerTests(unittest.TestCase):
    def test_random_element_mixup_uses_one_partner_map_for_images_and_targets(self):
        mixup = Mixup(
            mixup_alpha=0.2,
            mode="elem",
            pairing="random",
            label_smoothing=0.0,
            num_classes=[4],
        )
        mixup._params_per_elem = lambda batch_size: (
            np.full(batch_size, 0.25, dtype=np.float32),
            np.zeros(batch_size, dtype=bool),
        )
        torch.manual_seed(7)
        partners = torch.randperm(4)
        torch.manual_seed(7)
        images = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1)
        original = images.clone()
        labels = torch.arange(4)
        mixed_images, mixed_targets = mixup(images, [labels])
        expected_images = original * 0.25 + original[partners] * 0.75
        expected_targets = torch.nn.functional.one_hot(labels, 4).float() * 0.25
        expected_targets += torch.nn.functional.one_hot(labels[partners], 4).float() * 0.75
        torch.testing.assert_close(mixed_images, expected_images)
        torch.testing.assert_close(mixed_targets, expected_targets)

    def test_exact_epoch_indexed_exponential_schedule(self):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.Adam([parameter], lr=0.001)
        scheduler = HTCapsNetExponentialScheduler(
            optimizer,
            initial_lr=0.001,
            start_epoch=10,
            decay_rate=0.95,
        )
        scheduler.step(10)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.001, places=12)
        scheduler.step(11)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.00095, places=12)


if __name__ == "__main__":
    unittest.main()
