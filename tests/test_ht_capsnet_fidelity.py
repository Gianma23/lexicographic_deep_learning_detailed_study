import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as torchvision_transforms

from datasets.cifar100 import CIFAR100Dataset
from datasets.loaders import _collate_fn
from datasets.transforms import build_batch_normalizer, build_transforms
from models.ht_capsnet.losses import (
    _dynamic_level_weights,
    _initial_level_weights,
    _margin_loss,
    compute_loss,
)
from models.ht_capsnet.model import (
    HTCapsNet,
    KerasMultiHeadAttention,
    _build_backbone,
    _ConvBackbone,
    _keras_glorot_uniform_limit,
)
from models.ht_capsnet.routing import (
    hierarchical_agreement,
    squash,
    taxonomy_guided_routing_weights,
)
from train.mixup import Mixup
from train.runtime.optimization import HTCapsNetExponentialScheduler, KerasAdam
from train.runtime.selection import selection_components, selection_key


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

    def test_rank_three_keras_glorot_fans_set_attention_scale(self):
        torch.manual_seed(5)
        layer = KerasMultiHeadAttention(64, 32, num_heads=16, key_dim=32)
        expected_limits = {
            "query_projection": _keras_glorot_uniform_limit((64, 16, 32)),
            "key_projection": _keras_glorot_uniform_limit((32, 16, 32)),
            "value_projection": _keras_glorot_uniform_limit((32, 16, 32)),
            "output_projection": _keras_glorot_uniform_limit((16, 32, 64)),
        }
        for name, limit in expected_limits.items():
            weights = getattr(layer, name).weight.detach()
            self.assertLessEqual(float(weights.abs().max()), limit)
            self.assertAlmostEqual(
                float(weights.std(unbiased=False)),
                limit / math.sqrt(3.0),
                delta=limit * 0.02,
            )

        legacy = KerasMultiHeadAttention(
            64,
            32,
            num_heads=16,
            key_dim=32,
            initializer="pytorch_xavier",
        )
        self.assertGreater(
            float(legacy.query_projection.weight.detach().std()),
            float(layer.query_projection.weight.detach().std()) * 2.0,
        )


class RoutingAndLossTests(unittest.TestCase):
    def test_squash_and_margin_loss_match_source_equations(self):
        capsules = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
        squared_norm = capsules.square().sum(dim=-1, keepdim=True)
        expected = squared_norm / (1.0 + squared_norm)
        expected = expected * capsules / torch.sqrt(squared_norm + 1e-7)
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

    def test_parent_lengths_are_softmaxed_before_source_taxonomy_mask(self):
        raw = torch.full((1, 2, 3), 2.0)
        taxonomy = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        previous = torch.zeros(1, 2, 4)
        previous[:, 0, 0] = 8.0
        previous[:, 1, 0] = 2.0

        actual = taxonomy_guided_routing_weights(
            raw,
            level=1,
            taxonomy_matrix=taxonomy,
            prev_predictions=previous,
            mask_threshold_high=0.99,
        )
        activations = torch.softmax(torch.tensor([[8.0, 2.0]]), dim=-1)
        weighted_taxonomy = taxonomy.unsqueeze(0) * activations.unsqueeze(-1)
        soft_mask = 0.89 * torch.sigmoid(
            0.5 * (weighted_taxonomy - 0.5)
        ) + 0.1
        expected = torch.softmax(raw * soft_mask * 0.5, dim=-1)
        torch.testing.assert_close(actual, expected)

    def test_dynamic_weights_match_released_callback_parentheses(self):
        logits = [
            torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
            torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            torch.tensor([[0.0, 2.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]),
        ]
        targets = [torch.zeros(2, dtype=torch.long) for _ in logits]
        initial = _initial_level_weights([2, 3, 4])

        actual, _, _ = _dynamic_level_weights(
            logits,
            targets,
            decay=0.0,
        )

        source_terms = [
            1.0 - initial[0],
            1.0 - 0.5 * initial[1],
            1.0,
        ]
        torch.testing.assert_close(
            torch.tensor(actual),
            torch.tensor([value / sum(source_terms) for value in source_terms]),
        )

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
    def _model(self, **overrides):
        kwargs = dict(
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
        kwargs.update(overrides)
        return HTCapsNet(**kwargs)

    def test_primary_capsules_follow_nhwc_flatten_order(self):
        model = self._model()
        feature = torch.arange(8 * 16 * 16, dtype=torch.float32).view(1, 8, 16, 16)
        actual = model._build_primary_caps(feature)
        expected_flat = feature.permute(0, 2, 3, 1).contiguous().view(1, -1, 8)
        torch.testing.assert_close(actual, squash(expected_flat, dim=-1))

    def test_later_primary_capsules_reshape_the_source_squashed_tensor(self):
        feature = torch.linspace(-2.0, 3.0, 8 * 16 * 16).view(1, 8, 16, 16)
        model = self._model()
        primary = model._build_primary_caps(feature)
        later_level = model._reshape_primary_for_level(
            primary,
            target_dim=4,
        )
        torch.testing.assert_close(later_level, primary.reshape(1, -1, 4))

        independently_squashed = squash(
            feature.permute(0, 2, 3, 1).contiguous().view(1, -1, 4),
            dim=-1,
        )
        self.assertGreater(
            float((later_level - independently_squashed).abs().max()),
            1e-3,
        )

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

    def test_efficientnet_uses_explicit_keras_weight_conversion_and_training_defaults(self):
        with patch("timm.create_model", return_value=torch.nn.Identity()) as create_model:
            _build_backbone("efficientnet_b7", None, 4, 64, 2)
        create_model.assert_called_once_with(
            "tf_efficientnet_b7.aa_in1k",
            pretrained=False,
            num_classes=0,
            global_pool="",
            bn_momentum=0.01,
            drop_path_rate=0.2,
        )

    def test_inference_probabilities_are_normalized(self):
        model = self._model().eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 32, 32))
        for probabilities, margin_scores in zip(
            output["effective_probs_per_level"],
            output["margin_scores_per_level"],
        ):
            torch.testing.assert_close(
                probabilities.sum(dim=-1),
                torch.ones(probabilities.size(0)),
            )
            torch.testing.assert_close(margin_scores, probabilities)

    def test_training_margin_scores_are_raw_capsule_lengths(self):
        model = self._model().train()
        output = model(torch.randn(2, 3, 32, 32))
        for raw, margin_scores in zip(
            output["logits_per_level"],
            output["margin_scores_per_level"],
        ):
            self.assertIs(raw, margin_scores)

    def test_released_source_port_completes_a_finite_backward_pass(self):
        model = self._model().train()
        cfg = _loss_cfg()
        targets = torch.tensor([[0, 0, 0], [1, 2, 3]])

        output = model(torch.randn(2, 3, 32, 32))
        loss, _ = compute_loss(output, targets, cfg)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

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

    def test_dynamic_weights_read_epoch_running_accuracy_not_batch_accuracy(self):
        # Upstream reads `acc_i` from the Keras `logs` dict inside
        # `on_train_batch_end`, which carries the metric accumulated since the
        # start of the epoch. Two batches with different accuracies must produce
        # the weights implied by their pooled accuracy, not by the second batch.
        model = self._model()
        cfg = _loss_cfg()
        targets = torch.zeros((2, 3), dtype=torch.long)
        all_correct = [
            torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
            torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            torch.tensor([[2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]),
        ]
        all_wrong = [torch.roll(level, shifts=1, dims=-1) for level in all_correct]

        def step(logits):
            output = {
                "logits_per_level": logits,
                "level_loss_weights": model.level_loss_weights.detach().clone(),
                "lw_running_correct": model.lw_running_correct.detach().clone(),
                "lw_running_count": model.lw_running_count.detach().clone(),
            }
            _, _, aux = compute_loss(output, targets, cfg, return_aux=True)
            model.post_optimizer_step(aux)

        model.set_epoch(0)
        step(all_correct)
        torch.testing.assert_close(model.lw_running_correct, torch.tensor([2.0, 2.0, 2.0]))
        torch.testing.assert_close(model.lw_running_count, torch.tensor(2.0))

        step(all_wrong)
        # Pooled accuracy is 2/4 = 0.5 at every level, not the 0.0 of batch two.
        torch.testing.assert_close(model.lw_running_correct, torch.tensor([2.0, 2.0, 2.0]))
        torch.testing.assert_close(model.lw_running_count, torch.tensor(4.0))
        pooled = model.level_loss_weights.detach().clone()

        initial = _initial_level_weights([2, 3, 4])
        taus = [1.0 - 0.5 * float(initial[level]) for level in range(3)]
        expected = torch.tensor([tau / sum(taus) for tau in taus])
        torch.testing.assert_close(pooled, expected)

        # A new epoch clears the accumulator, so the next batch stands alone.
        model.set_epoch(1)
        torch.testing.assert_close(model.lw_running_count, torch.tensor(0.0))
        step(all_wrong)
        self.assertFalse(torch.equal(model.level_loss_weights, pooled))

    def test_keras_efficientnet_stem_applies_rescaling_and_normalization(self):
        # Keras `EfficientNetB7` embeds `Rescaling(1/255)` then `Normalization`,
        # whose ImageNet weights carry mean [0.485, 0.456, 0.406] and variance
        # [0.229, 0.224, 0.225]; the layer divides by sqrt(variance).
        model = self._model()
        model.backbone_preprocessing = "keras"
        images = torch.randn(2, 3, 32, 32)

        actual = model._prepare_backbone_input(images)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        variance = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        expected = ((images / 255.0) - mean) / torch.sqrt(variance)
        torch.testing.assert_close(actual, expected)

    def test_custom_backbone_leaves_inputs_unrescaled(self):
        model = self._model()
        images = torch.randn(2, 3, 32, 32)
        torch.testing.assert_close(model._prepare_backbone_input(images), images)

    def test_timm_preprocessing_is_an_explicit_non_source_ablation(self):
        model = self._model()
        model.backbone_preprocessing = "timm"
        images = torch.rand(2, 3, 32, 32)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        torch.testing.assert_close(
            model._prepare_backbone_input(images),
            (images - mean) / std,
        )

class ScopedNormalizationTests(unittest.TestCase):

    @staticmethod
    def _cfg(scope):
        return SimpleNamespace(
            dataset={
                "image_size": 8,
                "transforms": {
                    "fixed_resize_only": True,
                    "fixed_resize_interpolation": "bilinear",
                    "use_timm": False,
                    "normalization": "standardscaler",
                    "normalization_scope": scope,
                },
            }
        )

    def test_batch_scope_moves_the_statistic_out_of_the_per_sample_transform(self):
        image_pipeline = build_transforms(self._cfg("image"), "train")
        batch_pipeline = build_transforms(self._cfg("batch"), "train")
        self.assertEqual(len(image_pipeline.transforms), len(batch_pipeline.transforms) + 1)
        self.assertIsNone(build_batch_normalizer(self._cfg("image")))
        self.assertIsNotNone(build_batch_normalizer(self._cfg("batch")))

    def test_collate_reduces_over_the_batch_and_matches_the_source_reduction(self):
        torch.manual_seed(3)
        # Distinct per-image brightness and contrast, which per-image scoping erases.
        images = [torch.randn(3, 4, 4) * (i + 1) + i for i in range(5)]
        batch = [(image, torch.zeros(3, dtype=torch.long), {}) for image in images]

        stacked = torch.stack(images, dim=0)
        expected = (stacked - stacked.mean()) / stacked.std(unbiased=False)

        collated, _, _ = _collate_fn(
            batch,
            batch_normalizer=build_batch_normalizer(self._cfg("batch")),
        )
        torch.testing.assert_close(collated, expected)
        self.assertAlmostEqual(float(collated.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(collated.std(unbiased=False)), 1.0, places=5)
        # Per-image brightness/contrast survives, unlike under per-image scoping.
        per_image_mean = collated.mean(dim=(1, 2, 3))
        self.assertGreater(float(per_image_mean.max() - per_image_mean.min()), 0.5)

    def test_image_scope_leaves_the_collate_untouched(self):
        torch.manual_seed(3)
        images = [torch.randn(3, 4, 4) * (i + 1) + i for i in range(5)]
        batch = [(image, torch.zeros(3, dtype=torch.long), {}) for image in images]
        collated, _, _ = _collate_fn(batch)
        torch.testing.assert_close(collated, torch.stack(images, dim=0))

    def test_dataset_scope_uses_one_fixed_scalar_for_the_complete_split(self):
        cfg = self._cfg("dataset")
        pipeline = build_transforms(
            cfg,
            "train",
            dataset_statistics={"mean": 0.25, "std": 0.5},
        )
        image = Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8), mode="RGB")
        actual = pipeline(image)
        expected = torch.full_like(actual, ((128.0 / 255.0) - 0.25) / 0.5)
        torch.testing.assert_close(actual, expected)
        self.assertIsNone(build_batch_normalizer(cfg))

    def test_cifar_split_statistics_reduce_only_selected_samples(self):
        dataset = object.__new__(CIFAR100Dataset)
        dataset._cifar_images = np.stack(
            [
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.full((2, 2, 3), 128, dtype=np.uint8),
                np.full((2, 2, 3), 255, dtype=np.uint8),
            ],
            axis=0,
        )
        dataset.samples = [{"image": 0}, {"image": 2}]
        statistics = dataset.scalar_normalization_statistics("standardscaler")
        self.assertAlmostEqual(statistics["mean"], 0.5, places=12)
        self.assertAlmostEqual(statistics["std"], 0.5, places=12)


class FixedResizeTests(unittest.TestCase):
    @staticmethod
    def _cfg(antialias=None, crop_bottom=None, intermediate_size=None):
        transforms_cfg = {
            "fixed_resize_only": True,
            "fixed_resize_interpolation": "bilinear",
            "use_timm": False,
            "normalization": "none",
        }
        if antialias is not None:
            transforms_cfg["fixed_resize_antialias"] = antialias
        if crop_bottom is not None:
            transforms_cfg["manual"] = {"crop_bottom_pixels": crop_bottom}
        if intermediate_size is not None:
            transforms_cfg["fixed_resize_intermediate_size"] = intermediate_size
        return SimpleNamespace(dataset={"image_size": 16, "transforms": transforms_cfg})

    def test_antialias_false_converts_before_resampling(self):
        pipeline = build_transforms(self._cfg(antialias=False), "train").transforms
        self.assertIsInstance(pipeline[0], torchvision_transforms.ToTensor)
        self.assertIsInstance(pipeline[1], torchvision_transforms.Resize)
        self.assertFalse(pipeline[1].antialias)

    def test_antialias_default_keeps_the_pil_resample_order(self):
        pipeline = build_transforms(self._cfg(), "train").transforms
        self.assertIsInstance(pipeline[0], torchvision_transforms.Resize)
        self.assertIsInstance(pipeline[1], torchvision_transforms.ToTensor)

    def test_bottom_crop_runs_before_the_resize_on_both_branches(self):
        image = Image.new("RGB", (40, 60))
        for antialias in (True, False):
            pipeline = build_transforms(self._cfg(antialias, crop_bottom=20), "train")
            self.assertEqual(pipeline.transforms[0].pixels, 20)
            self.assertEqual(pipeline.transforms[0](image).size, (40, 40))
            self.assertEqual(tuple(pipeline(image).shape), (3, 16, 16))

    def test_bottom_crop_is_absent_when_unset(self):
        pipeline = build_transforms(self._cfg(antialias=False), "train").transforms
        self.assertFalse(any(hasattr(op, "pixels") for op in pipeline))

    def test_antialias_choice_changes_the_downsampled_image(self):
        torch.manual_seed(11)
        image = Image.fromarray(
            (torch.rand(96, 96, 3).numpy() * 255).astype(np.uint8), mode="RGB"
        )
        aliased = build_transforms(self._cfg(antialias=False), "train")(image)
        filtered = build_transforms(self._cfg(antialias=True), "train")(image)
        self.assertGreater(float((aliased - filtered).abs().mean()), 1e-3)

    def test_source_path_loader_runs_512_then_final_tensor_resize(self):
        pipeline = build_transforms(
            self._cfg(antialias=False, intermediate_size=512),
            "train",
        ).transforms
        self.assertIsInstance(pipeline[0], torchvision_transforms.ToTensor)
        self.assertEqual(tuple(pipeline[1].size), (512, 512))
        self.assertEqual(tuple(pipeline[2].size), (16, 16))
        self.assertFalse(pipeline[1].antialias)
        self.assertFalse(pipeline[2].antialias)


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

    def test_keras_adam_matches_epsilon_hat_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
        gradient = torch.tensor([0.25, -0.5], dtype=torch.float64)
        parameter.grad = gradient.clone()
        optimizer = KerasAdam([parameter], lr=0.001, eps=1e-7)

        beta1, beta2 = 0.9, 0.999
        moment = (1.0 - beta1) * gradient
        variance = (1.0 - beta2) * gradient.square()
        lr_t = 0.001 * math.sqrt(1.0 - beta2) / (1.0 - beta1)
        expected = torch.tensor([1.0, -2.0], dtype=torch.float64)
        expected -= lr_t * moment / (variance.sqrt() + 1e-7)

        optimizer.step()
        torch.testing.assert_close(parameter, expected, rtol=1e-12, atol=1e-12)
        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = KerasAdam([restored_parameter], lr=0.001, eps=1e-7)
        restored.load_state_dict(optimizer.state_dict())
        self.assertEqual(restored.param_groups[0]["step"], 1)


class CheckpointSelectionTests(unittest.TestCase):
    def test_deepest_accuracy_ignores_hierarchical_metric_ranking(self):
        metrics = {
            "acc_level_independent_0": 0.8,
            "acc_level_independent_1": 0.6,
            "acc_level_independent_2": 0.4,
            "fpa_independent": 0.9,
            "tice_independent": 0.1,
            "weighted_ap_independent": 0.8,
        }
        self.assertEqual(
            selection_key(
                metrics,
                mode="independent",
                strategy="deepest_accuracy",
            ),
            (0.4, float("-inf"), float("-inf")),
        )
        components = selection_components(
            metrics,
            mode="independent",
            strategy="deepest_accuracy",
        )
        self.assertEqual(
            components["primary_name"],
            "acc_level_independent_2",
        )
        self.assertEqual(
            selection_key(metrics, mode="independent"),
            (0.9, -0.1, 0.8),
        )


if __name__ == "__main__":
    unittest.main()
