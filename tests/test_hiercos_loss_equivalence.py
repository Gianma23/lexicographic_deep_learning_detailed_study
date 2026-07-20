from __future__ import annotations

import unittest

import torch
from omegaconf import OmegaConf

from models.orthonormal_plugin.losses import compute_loss


class HierCosLossEquivalenceTest(unittest.TestCase):
    @staticmethod
    def _output(node_logits: torch.Tensor, node_prob_weights: torch.Tensor):
        batch_size = int(node_logits.size(0))
        return {
            "logits_per_level": [
                node_logits.new_zeros((batch_size, 2)),
                node_logits.new_zeros((batch_size, 4)),
            ],
            "node_logits": node_logits,
            "hiercos_level_node_ids": [
                torch.tensor([0, 1], device=node_logits.device),
                torch.tensor([2, 3, 4, 5], device=node_logits.device),
            ],
            "leaf_to_level_local": torch.tensor(
                [
                    [0, 0],
                    [0, 1],
                    [1, 2],
                    [1, 3],
                ],
                device=node_logits.device,
            ),
            "node_prob_weights": node_prob_weights,
        }

    @staticmethod
    def _config(loss: str):
        return OmegaConf.create(
            {
                "model": {
                    "loss": loss,
                    "weight_mode": "kl_leaf",
                    "alpha": 0.1,
                }
            }
        )

    def test_global_cross_entropy_matches_kl_gradients_for_same_path_targets(self) -> None:
        depth_weights = torch.arange(2, 0, -1, dtype=torch.float64)
        node_prob_weights = torch.exp(1.0 / depth_weights)
        node_prob_weights = node_prob_weights.div(node_prob_weights.norm()).pow(2)
        targets = torch.tensor([[0, 0], [1, 3]], dtype=torch.long)
        initial_logits = torch.tensor(
            [
                [0.4, -0.7, 1.2, -0.3, 0.8, -1.1],
                [-0.2, 0.9, -0.6, 1.4, -1.0, 0.5],
            ],
            dtype=torch.float64,
        )

        ce_logits = initial_logits.clone().requires_grad_(True)
        ce_loss, _ = compute_loss(
            output=self._output(ce_logits, node_prob_weights),
            targets=targets,
            cfg=self._config("global_softmax_ce_reg"),
        )
        ce_grad = torch.autograd.grad(ce_loss, ce_logits)[0]

        kl_logits = initial_logits.clone().requires_grad_(True)
        kl_loss, _ = compute_loss(
            output=self._output(kl_logits, node_prob_weights),
            targets=targets,
            cfg=self._config("kl_reg"),
        )
        kl_grad = torch.autograd.grad(kl_loss, kl_logits)[0]

        target_entropy = -(node_prob_weights * node_prob_weights.log()).sum()
        torch.testing.assert_close(ce_grad, kl_grad, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(ce_loss - kl_loss, target_entropy, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
