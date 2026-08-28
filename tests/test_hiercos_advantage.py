import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.hiercos.losses import compute_loss
from models.hiercos.model import HierCosModel


class HierCosAdvantageTest(unittest.TestCase):
    @staticmethod
    def _advantage_model() -> HierCosModel:
        model = HierCosModel.__new__(HierCosModel)
        nn.Module.__init__(model)
        model.depth = 3
        model.parent_index_buffer_names = [None, "parent_index_level_1", "parent_index_level_2"]
        model.register_buffer(
            "parent_index_level_1",
            torch.tensor([0, 0, 1], dtype=torch.long),
            persistent=False,
        )
        model.register_buffer(
            "parent_index_level_2",
            torch.tensor([0, 1, 1, 2], dtype=torch.long),
            persistent=False,
        )
        return model

    def test_projected_heads_return_native_signed_coordinates(self):
        model = self._advantage_model()
        model.projection_eps = 1.0e-6
        model.projection_heads = nn.ModuleList(
            [nn.Linear(5, width) for width in (2, 3, 4)]
        )
        expected = [
            torch.tensor([-2.0, 1.0]),
            torch.tensor([-0.5, 4.0, -3.0]),
            torch.tensor([-1.0, 2.0, -0.25, 5.0]),
        ]
        with torch.no_grad():
            for head, bias in zip(model.projection_heads, expected):
                head.weight.zero_()
                head.bias.copy_(bias)

        branch_logits = model._projected_branch_logits_per_level(
            z=torch.zeros((1, 5)),
            rho_prime=torch.ones((1, 5)),
        )

        for actual, bias in zip(branch_logits, expected):
            torch.testing.assert_close(actual, bias.unsqueeze(0))

    def test_advantage_is_recursive_post_absolute_and_detached(self):
        model = self._advantage_model()
        node_logits = [
            torch.tensor([[-2.0, 1.0]], requires_grad=True),
            torch.tensor([[-0.5, 4.0, -3.0]], requires_grad=True),
            torch.tensor([[-1.0, 2.0, -0.25, 5.0]], requires_grad=True),
        ]

        scores = model._advantage_scores_per_level(node_logits)

        torch.testing.assert_close(scores[0], torch.tensor([[2.0, 1.0]]))
        torch.testing.assert_close(scores[1], torch.tensor([[2.5, 6.0, 4.0]]))
        torch.testing.assert_close(scores[2], torch.tensor([[3.5, 8.0, 6.25, 9.0]]))

        scores[-1].sum().backward()
        self.assertIsNone(node_logits[0].grad)
        self.assertIsNone(node_logits[1].grad)
        torch.testing.assert_close(
            node_logits[2].grad,
            torch.tensor([[-1.0, 1.0, -1.0, 1.0]]),
        )

    def test_loss_uses_advantage_for_ce_but_native_scores_for_regularization(self):
        node_logits_per_level = [
            torch.tensor([[2.0, -1.0]]),
            torch.tensor([[0.5, -4.0, 3.0]]),
            torch.tensor([[1.0, -2.0, 0.25, -5.0]]),
        ]
        node_logits = torch.cat(node_logits_per_level, dim=1)
        level_node_ids = [
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([2, 3, 4], dtype=torch.long),
            torch.tensor([5, 6, 7, 8], dtype=torch.long),
        ]
        common_output = {
            "node_logits": node_logits,
            "node_logits_per_level": node_logits_per_level,
            "effective_node_logits_per_level": None,
            "hiercos_level_node_ids": level_node_ids,
            "leaf_to_level_local": torch.tensor(
                [[0, 0, 0], [0, 1, 1], [1, 1, 2], [1, 2, 3]],
                dtype=torch.long,
            ),
            "node_prob_weights": torch.ones(3),
        }
        targets = torch.tensor([[0, 1, 1]], dtype=torch.long)
        cfg = SimpleNamespace(
            model={"loss": "level_softmax_ce_reg", "weight_mode": "equal", "alpha": 0.1}
        )

        favorable = [
            torch.tensor([[5.0, 0.0]]),
            torch.tensor([[0.0, 5.0, 0.0]]),
            torch.tensor([[0.0, 5.0, 0.0, 0.0]]),
        ]
        unfavorable = [
            torch.tensor([[0.0, 5.0]]),
            torch.tensor([[5.0, 0.0, 0.0]]),
            torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
        ]

        _, favorable_metrics = compute_loss(
            {
                **common_output,
                "logits_per_level": favorable,
                "advantage_scores_per_level": favorable,
            },
            targets,
            cfg,
        )
        _, unfavorable_metrics = compute_loss(
            {
                **common_output,
                "logits_per_level": unfavorable,
                "advantage_scores_per_level": unfavorable,
            },
            targets,
            cfg,
        )

        self.assertLess(favorable_metrics["ce"], unfavorable_metrics["ce"])
        self.assertEqual(favorable_metrics["reg"], unfavorable_metrics["reg"])

    def test_full_model_exposes_advantage_as_the_deployed_score_stream(self):
        torch.manual_seed(7)
        taxonomy = {
            "parent_of": {
                1: {0: 0, 1: 0, 2: 1},
                2: {0: 0, 1: 1, 2: 1, 3: 2},
            }
        }
        model = HierCosModel(
            num_classes_per_level=[2, 3, 4],
            taxonomy=taxonomy,
            variant="haframe_wide_resnet",
            pretrained=False,
            fixed_frame_mode="identity",
            projection_cfg={
                "enabled": True,
                "advantage_enabled": True,
                "feature_dim": 10,
                "eps": 1.0e-6,
            },
            wide_depth=10,
            wide_widen_factor=1,
        )
        model.train()

        output = model(torch.randn((2, 3, 32, 32)))
        advantage_scores = output["advantage_scores_per_level"]
        self.assertIs(output["logits_per_level"], advantage_scores)
        self.assertIsNot(output["subspace_scores_per_level"], advantage_scores)

        native_scores = [level_logits.abs() for level_logits in output["node_logits_per_level"]]
        torch.testing.assert_close(advantage_scores[0], native_scores[0])
        for level in (1, 2):
            expected = native_scores[level] + model._parent_baseline(
                level,
                advantage_scores[level - 1].detach(),
            )
            torch.testing.assert_close(advantage_scores[level], expected)

        cfg = SimpleNamespace(
            model={"loss": "level_softmax_ce_reg", "weight_mode": "equal", "alpha": 0.1}
        )
        targets = torch.tensor([[0, 1, 1], [1, 2, 3]], dtype=torch.long)
        loss, _ = compute_loss(output, targets, cfg)
        loss.backward()
        self.assertIsNotNone(model.projection_heads[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
