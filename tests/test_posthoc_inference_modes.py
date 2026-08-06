import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from evaluation.evaluate_checkpoints import (
    _prepare_test_config,
    _resolve_inference_modes,
    canonical_inference_rule,
    legacy_mode_name,
    native_inference_rule,
)
from evaluation.posthoc_inference import PosthocInferenceRule
from models.hiercos.model import HierCosModel
from models.orthonormal_plugin.config import INPUT_KEY


NUM_CLASSES = [2, 3, 4]
TAXONOMY = {
    "parent_of": {
        1: {0: 0, 1: 0, 2: 1},
        2: {0: 0, 1: 1, 2: 1, 3: 2},
    }
}


def _rule(mode: str, model_name: str = "hcast") -> PosthocInferenceRule:
    return PosthocInferenceRule(
        rule=mode,
        num_classes_per_level=NUM_CLASSES,
        taxonomy=TAXONOMY,
        model_name=model_name,
    )


def _classifier_output() -> dict:
    torch.manual_seed(7)
    scores = [torch.randn(5, classes) for classes in NUM_CLASSES]
    return {"logits_per_level": scores, INPUT_KEY: scores}


class NodeScoreReadoutTests(unittest.TestCase):
    def test_classifier_logits_are_read_signed_and_unchanged(self):
        # `node_score` must reproduce what H-CAST/HRN do: rank by the node's own
        # signed value. Taking magnitudes would promote confidently rejected
        # classes.
        output = _classifier_output()
        scores = _rule("node_score").scores_from_output(output)
        for expected, actual in zip(output["logits_per_level"], scores):
            self.assertTrue(torch.equal(expected, actual))

    def test_hiercos_coordinates_are_read_as_magnitudes(self):
        # Hier-COS trains `log_softmax(|node_logits|)`, so the sign of a frame
        # coordinate carries no class evidence.
        torch.manual_seed(11)
        node_logits = torch.randn(5, sum(NUM_CLASSES))
        scores = _rule("node_score", model_name="hiercos").scores_from_output(
            {"node_logits": node_logits}
        )
        expected = torch.split(node_logits.abs(), NUM_CLASSES, dim=1)
        for reference, actual in zip(expected, scores):
            self.assertTrue(torch.equal(reference, actual))
        self.assertTrue(all(float(score.min()) >= 0.0 for score in scores))


class HccTransformTests(unittest.TestCase):
    def test_projected_scores_satisfy_the_hierarchy_constraints(self):
        rule = _rule("hcc_node_score")
        projected = rule.scores_from_output(_classifier_output())

        residual_12 = projected[1] @ rule.projector.m12.transpose(0, 1) - projected[0]
        residual_23 = projected[2] @ rule.projector.m23.transpose(0, 1) - projected[1]
        self.assertLess(float(residual_12.abs().max().item()), 1e-5)
        self.assertLess(float(residual_23.abs().max().item()), 1e-5)

    def test_projection_shifts_each_sibling_group_by_one_constant(self):
        # `z_hat = z - coeff @ M` subtracts one value per parent, broadcast to
        # that parent's children. With a signed readout the sibling ranking is
        # preserved, so top-down decoding cannot move.
        output = _classifier_output()
        projected = _rule("hcc_node_score").scores_from_output(output)

        for level in (1, 2):
            shift = projected[level] - output["logits_per_level"][level]
            siblings_by_parent: dict = {}
            for child, parent in TAXONOMY["parent_of"][level].items():
                siblings_by_parent.setdefault(parent, []).append(child)
            for parent, children in siblings_by_parent.items():
                group = shift[:, children]
                spread = float((group - group[:, :1]).abs().max().item())
                self.assertLess(spread, 1e-5, (level, parent))

    def test_coarse_level_is_anchored_for_the_node_score_readout(self):
        output = _classifier_output()
        projected = _rule("hcc_node_score").scores_from_output(output)
        self.assertTrue(torch.equal(projected[0], output["logits_per_level"][0]))
        self.assertFalse(torch.equal(projected[1], output["logits_per_level"][1]))

    def test_hcc_requires_three_levels(self):
        two_level_taxonomy = {"parent_of": {1: {0: 0, 1: 0, 2: 1}}}
        # The untransformed readouts stay available at other depths.
        PosthocInferenceRule("node_score", [2, 3], two_level_taxonomy, "hcast")
        PosthocInferenceRule("subspace_norm", [2, 3], two_level_taxonomy, "hcast")
        with self.assertRaises(ValueError):
            PosthocInferenceRule("hcc_node_score", [2, 3], two_level_taxonomy, "hcast")


class HierCosNativeEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _build_model(hcc_cfg=None) -> HierCosModel:
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
            fixed_frame_per_level=False,
            wide_depth=10,
            wide_widen_factor=1,
            wide_drop_rate=0.0,
            hcc_cfg=hcc_cfg,
        )
        model.eval()
        return model

    @staticmethod
    def _images() -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, steps=2 * 3 * 32 * 32).reshape(2, 3, 32, 32)

    def test_subspace_norm_reproduces_native_hiercos_inference(self):
        model = self._build_model()
        with torch.no_grad():
            output = model(self._images())

        scores = _rule("subspace_norm", model_name="hiercos").scores_from_output(output)
        for expected, actual in zip(output["logits_per_level"], scores):
            self.assertTrue(torch.equal(expected, actual))

    def test_hcc_subspace_norm_reproduces_the_trained_hcc_forward_pass(self):
        baseline = self._build_model()
        with_hcc = self._build_model({"enabled": True, "eps": 1e-12})
        with torch.no_grad():
            baseline_output = baseline(self._images())
            hcc_output = with_hcc(self._images())

        scores = _rule("hcc_subspace_norm", model_name="hiercos").scores_from_output(
            baseline_output
        )
        self.assertIsNotNone(hcc_output["effective_logits_per_level"])
        for expected, actual in zip(hcc_output["effective_logits_per_level"], scores):
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

class SiblingShiftInvarianceTests(unittest.TestCase):
    """The per-parent shift preserves sibling order only for a signed readout.

    Coarse `[4, 0]` and middle `[1, -2, 0]` make the level-1 correction
    `-(-1 - 4)/2 = +2.5` on both children of coarse class 0, sending
    `[1, -2] -> [3.5, 0.5]`. Signed, the ranking is unchanged. Under `abs` it
    flips, because `|1| < |-2|` but `|3.5| > |0.5|`.
    """

    COARSE = [4.0, 0.0]
    MIDDLE = [1.0, -2.0, 0.0]
    FINE = [0.0, 0.0, 0.0, 0.0]
    SIBLINGS = [0, 1]

    def _classifier_scores(self, mode: str) -> torch.Tensor:
        levels = [
            torch.tensor([self.COARSE]),
            torch.tensor([self.MIDDLE]),
            torch.tensor([self.FINE]),
        ]
        return _rule(mode).scores_from_output({"logits_per_level": levels})[1]

    def _hiercos_scores(self, mode: str) -> torch.Tensor:
        node_logits = torch.tensor([self.COARSE + self.MIDDLE + self.FINE])
        return _rule(mode, model_name="hiercos").scores_from_output(
            {"node_logits": node_logits}
        )[1]

    def test_signed_readout_keeps_the_sibling_ranking(self):
        plain = self._classifier_scores("node_score")[0, self.SIBLINGS]
        projected = self._classifier_scores("hcc_node_score")[0, self.SIBLINGS]
        self.assertFalse(torch.equal(plain, projected))
        self.assertEqual(int(plain.argmax()), int(projected.argmax()))

    def test_magnitude_readout_can_reorder_siblings(self):
        plain = self._hiercos_scores("node_score")[0, self.SIBLINGS]
        projected = self._hiercos_scores("hcc_node_score")[0, self.SIBLINGS]
        self.assertEqual(int(plain.argmax()), 1)
        self.assertEqual(int(projected.argmax()), 0)


class TestConfigAdjustmentTests(unittest.TestCase):
    """A missing manifest silently changes the label space, so it must be flagged."""

    @staticmethod
    def _cfg(test_annotation, drop_last_eval=False):
        return OmegaConf.create(
            {
                "dataset": {
                    "root": "/nonexistent-root",
                    "annotations": {"train": "train.json", "test": test_annotation},
                },
                "dataloader": {"drop_last_eval": drop_last_eval},
            }
        )

    def test_missing_test_manifest_is_recorded_as_a_fallback(self):
        cfg = self._cfg("test.json")
        adjustments = _prepare_test_config(cfg)
        fields = [adjustment["field"] for adjustment in adjustments]
        self.assertIn("dataset.annotations.test", fields)
        self.assertNotIn("test", cfg.dataset.annotations)

    def test_present_manifest_is_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "test.json"
            manifest.write_text("{}", encoding="utf-8")
            cfg = self._cfg(str(manifest), drop_last_eval=True)
            adjustments = _prepare_test_config(cfg)
            fields = [adjustment["field"] for adjustment in adjustments]
            self.assertNotIn("dataset.annotations.test", fields)
            self.assertIn("dataloader.drop_last_eval", fields)
            self.assertEqual(cfg.dataset.annotations.test, str(manifest))


class InferenceModeResolutionTests(unittest.TestCase):
    def test_native_rule_per_model(self):
        self.assertEqual(native_inference_rule("hcast", False), "node_score")
        self.assertEqual(native_inference_rule("hrn", False), "node_score")
        self.assertEqual(native_inference_rule("hiercos", False), "subspace_norm")
        self.assertEqual(native_inference_rule("hcast", True), "hcc_node_score")
        self.assertEqual(native_inference_rule("hiercos", True), "hcc_subspace_norm")

    def test_all_and_both_lead_with_the_native_rule(self):
        self.assertEqual(
            _resolve_inference_modes("all", "hcast"),
            ("node_score", "subspace_norm", "hcc_node_score", "hcc_subspace_norm"),
        )
        self.assertEqual(
            _resolve_inference_modes("all", "hiercos"),
            ("subspace_norm", "node_score", "hcc_node_score", "hcc_subspace_norm"),
        )
        self.assertEqual(
            _resolve_inference_modes("both", "hcast"), ("node_score", "subspace_norm")
        )
        self.assertEqual(
            _resolve_inference_modes("both", "hiercos"), ("subspace_norm", "node_score")
        )
        self.assertEqual(
            _resolve_inference_modes("all", "hcast", True)[0], "hcc_node_score"
        )

    def test_legacy_names_map_onto_the_grid(self):
        self.assertEqual(canonical_inference_rule("normal", "hcast"), "node_score")
        self.assertEqual(canonical_inference_rule("normal", "hiercos"), "subspace_norm")
        self.assertEqual(canonical_inference_rule("hiercos", "hcast"), "subspace_norm")
        self.assertEqual(canonical_inference_rule("node_softmax", "hiercos"), "node_score")
        self.assertEqual(canonical_inference_rule("hcc", "hcast"), "hcc_node_score")
        self.assertEqual(canonical_inference_rule("hcc", "hiercos"), "hcc_subspace_norm")
        self.assertEqual(_resolve_inference_modes("normal", "hiercos"), ("subspace_norm",))

    def test_old_yaml_rows_map_back_onto_the_grid(self):
        # `normal` named whatever the checkpoint did natively, so it follows the
        # model rather than a fixed readout.
        self.assertEqual(legacy_mode_name("node_score", "hcast"), "normal")
        self.assertEqual(legacy_mode_name("subspace_norm", "hcast"), "hiercos")
        self.assertEqual(legacy_mode_name("hcc_node_score", "hcast"), "hcc")
        self.assertIsNone(legacy_mode_name("hcc_subspace_norm", "hcast"))

        self.assertEqual(legacy_mode_name("subspace_norm", "hiercos"), "normal")
        self.assertEqual(legacy_mode_name("node_score", "hiercos"), "node_softmax")
        self.assertEqual(legacy_mode_name("hcc_subspace_norm", "hiercos"), "hcc")
        self.assertIsNone(legacy_mode_name("hcc_node_score", "hiercos"))

        # An HCC-trained run's `normal` row was its HCC-corrected forward pass.
        self.assertEqual(legacy_mode_name("hcc_node_score", "hcast", True), "normal")
        self.assertEqual(legacy_mode_name("subspace_norm", "hcast", True), "hiercos")
        self.assertIsNone(legacy_mode_name("node_score", "hcast", True))

    def test_every_cell_is_valid_for_every_model(self):
        for model_name in ("hcast", "hrn", "hiercos", "lhdnn", "ht_capsnet"):
            for mode in _resolve_inference_modes("all", model_name):
                _rule(mode, model_name=model_name)


if __name__ == "__main__":
    unittest.main()
