import copy
import unittest
from types import SimpleNamespace

import torch

from models.hiercos.model import HierCosModel
from models.orthonormal_plugin.losses import compute_loss
from train.evaluation import evaluate_batch


NUM_CLASSES = [2, 3, 4]
TAXONOMY = {
    "parent_of": {
        1: {0: 0, 1: 0, 2: 1},
        2: {0: 0, 1: 1, 2: 1, 3: 2},
    }
}
TARGETS = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)


def _loss_cfg(loss_mode: str):
    return SimpleNamespace(
        model={
            "name": "hiercos",
            "loss": loss_mode,
            "weight_mode": "kl_leaf",
            "alpha": 0.05,
        },
        orthonormal_plugin={"enabled": False},
    )


class HierCosHccTests(unittest.TestCase):
    @staticmethod
    def _build_model(hcc_cfg=None, fixed_frame_per_level: bool = False) -> HierCosModel:
        # Reset before every construction so HCC configuration is the only
        # difference between otherwise identical tiny Hier-COS models.
        torch.manual_seed(123)
        model = HierCosModel(
            num_classes_per_level=NUM_CLASSES,
            taxonomy=TAXONOMY,
            variant="haframe_wide_resnet",
            transform_mode="full",
            pretrained=False,
            fixed_frame_mode="orthonormal_random",
            fixed_frame_per_level=fixed_frame_per_level,
            wide_depth=10,
            wide_widen_factor=1,
            wide_drop_rate=0.0,
            hcc_cfg=hcc_cfg,
            train_epochs=10,
        )
        model.eval()
        return model

    @staticmethod
    def _images() -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, steps=2 * 3 * 32 * 32).reshape(2, 3, 32, 32)

    def test_no_hcc_disabled_and_preactivation_paths_are_bitwise_identical(self):
        for fixed_frame_per_level in (False, True):
            with self.subTest(fixed_frame_per_level=fixed_frame_per_level):
                baseline = self._build_model(fixed_frame_per_level=fixed_frame_per_level)
                disabled = self._build_model(
                    {"enabled": False},
                    fixed_frame_per_level=fixed_frame_per_level,
                )
                preactivation = self._build_model(
                    {
                        "enabled": True,
                        "temperature": 10.0,
                        "eps": 1e-12,
                        "alpha_schedule": "step",
                        "alpha_start_epoch": 5,
                    },
                    fixed_frame_per_level=fixed_frame_per_level,
                )
                preactivation.set_epoch(0)

                self.assertEqual(list(baseline.state_dict()), list(disabled.state_dict()))
                self.assertEqual(list(baseline.state_dict()), list(preactivation.state_dict()))
                for key, value in baseline.state_dict().items():
                    self.assertTrue(torch.equal(value, disabled.state_dict()[key]), key)
                    self.assertTrue(torch.equal(value, preactivation.state_dict()[key]), key)

                images = self._images()
                with torch.no_grad():
                    outputs = [model(images) for model in (baseline, disabled, preactivation)]

                reference = outputs[0]
                for candidate in outputs[1:]:
                    self.assertTrue(torch.equal(reference["node_logits"], candidate["node_logits"]))
                    self.assertTrue(torch.equal(reference["leaf_logits"], candidate["leaf_logits"]))
                    for expected, actual in zip(reference["logits_per_level"], candidate["logits_per_level"]):
                        self.assertTrue(torch.equal(expected, actual))
                    self.assertIsNone(candidate["effective_logits_per_level"])
                    self.assertIsNone(candidate["effective_node_logits_per_level"])

                for loss_mode in ("kl_reg", "global_softmax_ce_reg", "level_softmax_ce_reg"):
                    reference_loss, reference_metrics = compute_loss(
                        reference,
                        TARGETS,
                        _loss_cfg(loss_mode),
                    )
                    for candidate in outputs[1:]:
                        candidate_loss, candidate_metrics = compute_loss(
                            candidate,
                            TARGETS,
                            _loss_cfg(loss_mode),
                        )
                        self.assertTrue(torch.equal(reference_loss, candidate_loss), loss_mode)
                        self.assertEqual(reference_metrics, candidate_metrics)

    def test_active_hcc_drives_loss_scores_evaluation_and_gradients(self):
        model = self._build_model(
            {
                "enabled": True,
                "temperature": 1.0,
                "eps": 1e-12,
                "alpha_schedule": "step",
                "alpha_start_epoch": 0,
            }
        )
        output = model(self._images())

        effective_nodes = output["effective_node_logits_per_level"]
        effective_scores = output["effective_logits_per_level"]
        self.assertIsInstance(effective_nodes, list)
        self.assertIsInstance(effective_scores, list)
        effective_node_logits = torch.cat(effective_nodes, dim=1)
        recomputed_scores = model._level_subspace_scores(effective_node_logits)
        for expected, actual in zip(recomputed_scores, effective_scores):
            self.assertTrue(torch.equal(expected, actual))

        projector = model.hcc.projector
        residual_12 = effective_nodes[1] @ projector.m12.transpose(0, 1) - effective_nodes[0]
        residual_23 = effective_nodes[2] @ projector.m23.transpose(0, 1) - effective_nodes[1]
        self.assertLess(float(residual_12.abs().max().item()), 1e-5)
        self.assertLess(float(residual_23.abs().max().item()), 1e-5)

        # Each existing loss must be numerically identical to applying that
        # same formula to the effective node tensor as an ordinary raw input.
        reference_output = copy.copy(output)
        reference_output["node_logits"] = effective_node_logits
        reference_output["orthonormal_plugin_node_logits"] = effective_node_logits
        reference_output["node_logits_per_level"] = effective_nodes
        reference_output["orthonormal_plugin_node_logits_per_level"] = effective_nodes
        reference_output["effective_node_logits_per_level"] = None
        for loss_mode in ("kl_reg", "global_softmax_ce_reg", "level_softmax_ce_reg"):
            actual_loss, actual_metrics = compute_loss(
                output,
                TARGETS,
                _loss_cfg(loss_mode),
            )
            expected_loss, expected_metrics = compute_loss(
                reference_output,
                TARGETS,
                _loss_cfg(loss_mode),
            )
            self.assertTrue(torch.equal(expected_loss, actual_loss), loss_mode)
            self.assertEqual(expected_metrics, actual_metrics)

        # Evaluation must behave as if the recomputed effective scores were
        # the model's ordinary output scores.
        evaluation_reference = copy.copy(output)
        evaluation_reference["logits_per_level"] = effective_scores
        evaluation_reference["effective_logits_per_level"] = None
        actual_metrics = evaluate_batch(output, TARGETS, TAXONOMY)
        expected_metrics = evaluate_batch(evaluation_reference, TARGETS, TAXONOMY)
        primary_metric_keys = [
            key
            for key in expected_metrics
            if key.startswith(("acc_level_", "weighted_ap_", "fpa_", "ahd_", "tice_"))
        ]
        for key in primary_metric_keys:
            self.assertEqual(expected_metrics[key], actual_metrics[key], key)

        model.zero_grad(set_to_none=True)
        loss, _metrics = compute_loss(output, TARGETS, _loss_cfg("kl_reg"))
        loss.backward()
        backbone_grads = [
            parameter.grad
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(grad is not None and bool((grad != 0).any()) for grad in backbone_grads))


if __name__ == "__main__":
    unittest.main()
