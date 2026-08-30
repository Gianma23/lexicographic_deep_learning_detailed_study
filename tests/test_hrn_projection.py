import unittest

import torch
import torch.nn as nn

from models.hrn.model import HRNModel


class HRNProjectionTest(unittest.TestCase):
    @staticmethod
    def _model() -> HRNModel:
        taxonomy = {
            "parent_of": {
                1: {0: 0, 1: 0, 2: 1},
                2: {0: 0, 1: 1, 2: 1, 3: 2},
            }
        }
        return HRNModel(
            num_classes_per_level=[2, 3, 4],
            backbone="wide_resnet",
            pretrained=False,
            wide_depth=10,
            wide_widen_factor=1,
            branch_hidden_dim=16,
            embedding_dim=8,
            projection_cfg={"enabled": True, "eps": 1.0e-6},
            taxonomy=taxonomy,
        )

    def test_projection_uses_one_direct_linear_head_per_output(self):
        model = self._model()

        self.assertFalse(hasattr(model, "conv_block1"))
        self.assertFalse(hasattr(model, "fc1"))
        for head, classes in zip(
            (model.classifier_1, model.classifier_2, model.classifier_3),
            model.num_classes_per_level,
        ):
            self.assertIsInstance(head, nn.Linear)
            self.assertEqual(head.in_features, model.num_ftrs)
            self.assertEqual(head.out_features, classes)
        self.assertEqual(model.classifier_3_1.in_features, model.num_ftrs)
        self.assertEqual(model.classifier_3_1.out_features, model.num_classes_per_level[2])

    def test_projected_forward_preserves_level_shapes(self):
        torch.manual_seed(7)
        model = self._model().eval()

        with torch.no_grad():
            output = model(torch.randn((2, 3, 32, 32)))

        self.assertEqual(
            [tuple(logits.shape) for logits in output["logits_per_level"]],
            [(2, 2), (2, 3), (2, 4)],
        )
        self.assertEqual(tuple(output["tree_logits_per_level"][2].shape), (2, 4))
        self.assertEqual(tuple(output["species_ce_logits"].shape), (2, 4))
        self.assertEqual(
            [tuple(features.shape) for features in output["embeddings_per_level"]],
            [(2, model.num_ftrs)] * 3,
        )

    def test_projection_parameter_groups_cover_each_parameter_once(self):
        model = self._model()
        grouped = [
            parameter
            for group in model.parameter_groups(base_lr=0.1)
            for parameter in group["params"]
        ]

        self.assertEqual(len(grouped), len({id(parameter) for parameter in grouped}))
        self.assertEqual(
            {id(parameter) for parameter in grouped},
            {id(parameter) for parameter in model.parameters()},
        )

    def test_fine_gradient_is_filtered_against_direct_coarse_and_middle_heads(self):
        torch.manual_seed(11)
        model = self._model()
        model.projection_eps = 1.0e-8
        with torch.no_grad():
            model.shared_linear.weight.copy_(torch.eye(model.num_ftrs))
            model.shared_linear.bias.zero_()

        pooled = (torch.rand((2, model.num_ftrs)) + 1.0).requires_grad_()
        output = model._projected_head_logits(pooled)
        fine_gradient = torch.autograd.grad(
            output["species_ce_logits"].sum(),
            pooled,
        )[0]
        protected_rows = torch.cat(
            [model.classifier_1.weight, model.classifier_2.weight],
            dim=0,
        ).detach()
        protected_component = torch.matmul(fine_gradient, protected_rows.T)

        torch.testing.assert_close(
            protected_component,
            torch.zeros_like(protected_component),
            atol=1.0e-5,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
