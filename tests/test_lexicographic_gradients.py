import unittest
from unittest import mock

import torch

from train.lexicographic import gradients as gradient_ops


def _legacy_grad_norm(grads, include_mask):
    norm_sq = None
    for grad, include in zip(grads, include_mask):
        if not include or grad is None:
            continue
        grad_fp = grad.detach().float()
        value = torch.sum(grad_fp * grad_fp)
        norm_sq = value if norm_sq is None else norm_sq + value
    if norm_sq is None:
        return 0.0
    return float(torch.sqrt(norm_sq.clamp_min(0.0)).item())


def _legacy_grad_cosine(grads_a, grads_b, include_mask, eps):
    dot = None
    norm_a_sq = None
    norm_b_sq = None
    for grad_a, grad_b, include in zip(grads_a, grads_b, include_mask):
        if not include or grad_a is None or grad_b is None:
            continue
        grad_a_fp = grad_a.detach().float()
        grad_b_fp = grad_b.detach().float()
        dot_term = torch.sum(grad_a_fp * grad_b_fp)
        norm_a_term = torch.sum(grad_a_fp * grad_a_fp)
        norm_b_term = torch.sum(grad_b_fp * grad_b_fp)
        dot = dot_term if dot is None else dot + dot_term
        norm_a_sq = norm_a_term if norm_a_sq is None else norm_a_sq + norm_a_term
        norm_b_sq = norm_b_term if norm_b_sq is None else norm_b_sq + norm_b_term
    if dot is None or norm_a_sq is None or norm_b_sq is None:
        return 0.0
    denom = torch.sqrt(norm_a_sq.clamp_min(0.0)) * torch.sqrt(
        norm_b_sq.clamp_min(0.0)
    )
    cosine = dot / (denom + float(eps))
    return float(cosine.clamp(min=-1.0, max=1.0).item())


class BatchedGradientMetricTests(unittest.TestCase):
    def test_batched_reductions_match_previous_formulas(self):
        grads_a = (
            torch.tensor([1.0, -2.0, 3.0]),
            torch.tensor([[0.5, -1.5], [2.5, 4.0]]),
            None,
            torch.tensor([-3.0, 0.25]),
        )
        grads_b = (
            torch.tensor([-0.5, 1.0, 2.0]),
            torch.tensor([[2.0, 0.5], [-1.0, 3.0]]),
            torch.tensor([7.0]),
            torch.tensor([1.5, -2.0]),
        )
        mask_all = [True, True, True, True]
        mask_edges = [True, False, False, True]
        mask_empty = [False, False, True, False]
        eps = 1.0e-12

        actual = gradient_ops._batched_grad_metrics(
            norm_specs=[
                ("norm_all", grads_a, mask_all),
                ("norm_edges", grads_a, mask_edges),
                ("norm_empty", grads_a, mask_empty),
            ],
            cosine_specs=[
                ("cos_all", grads_a, grads_b, mask_all, eps),
                ("cos_edges", grads_a, grads_b, mask_edges, eps),
                ("cos_empty", grads_a, grads_b, mask_empty, eps),
            ],
        )
        expected = {
            "norm_all": _legacy_grad_norm(grads_a, mask_all),
            "norm_edges": _legacy_grad_norm(grads_a, mask_edges),
            "norm_empty": _legacy_grad_norm(grads_a, mask_empty),
            "cos_all": _legacy_grad_cosine(grads_a, grads_b, mask_all, eps),
            "cos_edges": _legacy_grad_cosine(grads_a, grads_b, mask_edges, eps),
            "cos_empty": _legacy_grad_cosine(grads_a, grads_b, mask_empty, eps),
        }

        self.assertEqual(actual, expected)


class LexicographicProjectionReuseTests(unittest.TestCase):
    def setUp(self):
        self.params = [
            torch.nn.Parameter(torch.zeros(2)),
            torch.nn.Parameter(torch.zeros(2)),
            torch.nn.Parameter(torch.zeros(2)),
        ]
        self.named_params = [(f"param_{idx}", param) for idx, param in enumerate(self.params)]
        self.level_grad_map = {
            "coarse": (
                torch.tensor([1.0, 2.0]),
                torch.tensor([-2.0, 1.0]),
                torch.tensor([0.5, -0.25]),
            ),
            "mid": (
                torch.tensor([-1.0, 0.5]),
                torch.tensor([3.0, -4.0]),
                None,
            ),
            "fine": (
                torch.tensor([0.25, -3.0]),
                None,
                None,
            ),
        }
        self.level_losses = [torch.ones((), requires_grad=True) for _ in range(3)]

    def test_prepare_builds_projected_gradients_once_with_metrics(self):
        trunk_masks = gradient_ops._resolve_trunk_masks(
            self.level_grad_map["coarse"],
            self.level_grad_map["mid"],
            self.level_grad_map["fine"],
        )
        for projection_mode in ("coarse_first", "fine_first"):
            for projection_rule in ("orthogonalize_all", "conflict_only"):
                with self.subTest(mode=projection_mode, rule=projection_rule):
                    expected_grads, expected_metrics = gradient_ops._build_lexicographic_grads(
                        coarse_grads=self.level_grad_map["coarse"],
                        mid_grads=self.level_grad_map["mid"],
                        fine_grads=self.level_grad_map["fine"],
                        trunk_masks=trunk_masks,
                        projection_mode=projection_mode,
                        projection_rule=projection_rule,
                        include_metrics=True,
                    )

                    with mock.patch.object(
                        gradient_ops,
                        "_build_lexicographic_grads",
                        wraps=gradient_ops._build_lexicographic_grads,
                    ) as build_mock:
                        state, metrics = gradient_ops.prepare_lexicographic_update(
                            trainable_named_params=self.named_params,
                            level_losses=self.level_losses,
                            precomputed_level_grad_map=self.level_grad_map,
                            projection_mode=projection_mode,
                            projection_rule=projection_rule,
                            include_metrics=True,
                        )

                    self.assertEqual(build_mock.call_count, 1)
                    self.assertIsNotNone(state)
                    self.assertEqual(metrics, expected_metrics)
                    for pack_name, expected_pack in expected_grads.items():
                        actual_pack = state.projected_grads[pack_name]
                        for actual_grad, expected_grad in zip(actual_pack, expected_pack):
                            if expected_grad is None:
                                self.assertIsNone(actual_grad)
                            else:
                                self.assertTrue(torch.equal(actual_grad, expected_grad))


if __name__ == "__main__":
    unittest.main()
