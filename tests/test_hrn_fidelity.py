import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.hrn.losses import _build_state_space, compute_loss
from models.hrn.model import HRNModel, _build_resnet50_backbone
from train.runtime.optimization import HierCosCosineScheduler


class HRNFidelityTests(unittest.TestCase):
    def test_requested_imagenet_weights_never_fall_back(self):
        with patch("torchvision.models.resnet50", side_effect=OSError("weights unavailable")):
            with self.assertRaisesRegex(RuntimeError, "requires ImageNet-pretrained"):
                _build_resnet50_backbone(pretrained=True)

    def test_state_space_and_full_label_loss(self):
        parent_of = {
            1: {0: 0, 1: 0, 2: 1},
            2: {0: 0, 1: 1, 2: 1, 3: 2},
        }
        state_space = _build_state_space(
            [2, 3, 4], parent_of, torch.device("cpu"), torch.float32
        )
        self.assertEqual(tuple(state_space.shape), (10, 9))

        tree_logits = [torch.randn(2, count) for count in (2, 3, 4)]
        output = {
            "logits_per_level": [tree_logits[0], tree_logits[1], torch.randn(2, 4)],
            "tree_scores_per_level": [torch.sigmoid(value) for value in tree_logits],
            "species_ce_logits": torch.randn(2, 4),
        }
        targets = torch.tensor([[0, 0, 0], [1, 2, 3]])
        loss, metrics = compute_loss(
            output,
            targets,
            SimpleNamespace(),
            taxonomy={"parent_of": parent_of},
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("hier_loss", metrics)
        self.assertIn("ce_loss_leaf", metrics)

        with self.assertRaisesRegex(ValueError, "full leaf-labeled"):
            compute_loss(
                output,
                {"hard_targets": targets, "observed_level": torch.ones(2, dtype=torch.long)},
                SimpleNamespace(),
                taxonomy={"parent_of": parent_of},
            )

    def test_parameter_groups_and_level_specific_effective_scores(self):
        try:
            model = HRNModel([2, 3, 4], pretrained=False)
        except Exception as exc:
            self.skipTest(f"torchvision ResNet-50 unavailable: {exc}")

        groups = model.parameter_groups(base_lr=0.002, trunk_lr_scale=0.1)
        self.assertEqual(len(groups), 11)
        self.assertTrue(all(group["lr"] == 0.002 for group in groups[:10]))
        self.assertEqual(groups[-1]["lr"], 0.0002)

        model.features = torch.nn.Identity()
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(2, 2048, 2, 2))
        torch.testing.assert_close(
            output["effective_probs_per_level"][0],
            torch.sigmoid(output["logits_per_level"][0]),
        )
        torch.testing.assert_close(
            output["effective_probs_per_level"][1],
            torch.sigmoid(output["logits_per_level"][1]),
        )
        torch.testing.assert_close(
            output["effective_probs_per_level"][2].sum(dim=-1),
            torch.ones(2),
        )

    def test_cosine_schedule_contract(self):
        head = torch.nn.Parameter(torch.zeros(()))
        trunk = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.SGD(
            [{"params": [head], "lr": 0.002}, {"params": [trunk], "lr": 0.0002}],
            lr=0.002,
        )
        scheduler = HierCosCosineScheduler(optimizer, num_epochs=200, base_lr=0.002)
        scheduler.step(100)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.001, places=12)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 0.0001, places=12)


if __name__ == "__main__":
    unittest.main()
