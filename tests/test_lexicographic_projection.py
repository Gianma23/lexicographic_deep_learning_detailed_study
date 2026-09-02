import unittest

import torch

from train.lexicographic.gradients import _build_lexicographic_grads, _resolve_trunk_masks


def _dot(grads_a, grads_b, mask):
    total = 0.0
    for grad_a, grad_b, include in zip(grads_a, grads_b, mask):
        if not include or grad_a is None or grad_b is None:
            continue
        total += float(torch.sum(grad_a.float() * grad_b.float()).item())
    return total


class LexicographicProjectionTest(unittest.TestCase):
    """The projected step must descend every higher-priority objective.

    `_build_lexicographic_grads` removes the target's component along each
    higher-priority gradient in turn. Projecting off their resultant instead
    would only make the target orthogonal to the sum, which lets
    `<total, g_priority>` fall below `||g_priority||^2` -- and go negative when
    the lower-priority gradients are much larger, as on HRN CIFAR-100.
    """

    NUM_PARAMS = 6

    def _level_grads(self, seed: int, scales=(1.0, 1.5, 6.0)):
        """Three level gradients over one all-levels block, fine much larger."""
        generator = torch.Generator().manual_seed(seed)
        levels = []
        for scale in scales:
            levels.append(
                tuple(
                    torch.randn(4, 5, generator=generator) * scale
                    for _ in range(self.NUM_PARAMS)
                )
            )
        return levels

    def _run(self, seed: int, projection_mode: str):
        coarse, mid, fine = self._level_grads(seed)
        masks = _resolve_trunk_masks(coarse, mid, fine, blocks=("p123",))
        pack, metrics = _build_lexicographic_grads(
            coarse,
            mid,
            fine,
            masks,
            projection_mode=projection_mode,
            eps=1e-12,
            include_metrics=True,
            blocks=("p123",),
        )
        return (coarse, mid, fine), masks["p123"], pack, metrics

    def test_all_parameters_land_in_the_all_levels_block(self):
        coarse, mid, fine = self._level_grads(0)
        masks = _resolve_trunk_masks(coarse, mid, fine, blocks=("p123", "p23", "p3"))
        self.assertTrue(all(masks["p123"]))
        self.assertFalse(any(masks["p23"]))
        self.assertFalse(any(masks["p3"]))

    def test_step_descends_the_priority_objective_exactly(self):
        for seed in range(25):
            for projection_mode, priority_index in (("coarse_first", 0), ("fine_first", 2)):
                with self.subTest(seed=seed, projection_mode=projection_mode):
                    levels, mask, pack, _ = self._run(seed, projection_mode)
                    priority = levels[priority_index]
                    descent = _dot(pack["total"], priority, mask)
                    self.assertAlmostEqual(
                        descent,
                        _dot(priority, priority, mask),
                        delta=1e-3,
                        msg="the step must descend the priority objective at the rate "
                        "of its own gradient",
                    )
                    self.assertGreater(descent, 0.0)

    def test_second_objective_keeps_its_own_descent(self):
        """The middle level is orthogonalised, so it too keeps `||g||^2`."""
        for seed in range(25):
            with self.subTest(seed=seed):
                levels, mask, pack, _ = self._run(seed, "coarse_first")
                mid_projected = pack["mid_projected"]
                self.assertAlmostEqual(
                    _dot(pack["total"], mid_projected, mask),
                    _dot(mid_projected, mid_projected, mask),
                    delta=1e-3,
                )

    def test_last_target_is_orthogonal_to_each_higher_gradient(self):
        for seed in range(25):
            with self.subTest(seed=seed):
                levels, mask, pack, _ = self._run(seed, "coarse_first")
                coarse_projected = pack["coarse"]
                mid_projected = pack["mid_projected"]
                fine_projected = pack["fine_projected"]
                scale = _dot(fine_projected, fine_projected, mask) ** 0.5
                for reference in (coarse_projected, mid_projected):
                    reference_scale = _dot(reference, reference, mask) ** 0.5
                    self.assertLess(
                        abs(_dot(fine_projected, reference, mask)),
                        1e-4 * scale * reference_scale,
                        msg="orthogonality to the resultant alone is not enough",
                    )

    def test_projection_diagnostics_cover_every_reference(self):
        _levels, _mask, _pack, metrics = self._run(0, "coarse_first")
        for key in (
            "post_projection_applied_p123_mid_coarse",
            "post_projection_applied_p123_fine_coarse",
            "post_projection_applied_p123_fine_mid",
            "post_projection_applied_p123_fine_higher",
            "post_projection_applied_t1_fine_higher",
        ):
            self.assertIn(key, metrics)
        # Each per-reference cosine is driven to zero, not only the resultant one.
        for key in (
            "post_cos_p123_mid_coarse",
            "post_cos_p123_fine_coarse",
            "post_cos_p123_fine_mid",
            "post_cos_p123_fine_higher",
            "post_cos_t1_fine_proj_coarse",
            "post_cos_t1_fine_proj_mid_proj",
        ):
            self.assertIn(key, metrics)
            self.assertAlmostEqual(metrics[key], 0.0, delta=1e-4)

    def test_two_level_block_is_unchanged(self):
        """A block with two active levels needs a single projection step."""
        generator = torch.Generator().manual_seed(3)
        empty = tuple(None for _ in range(self.NUM_PARAMS))
        mid = tuple(torch.randn(4, 5, generator=generator) for _ in range(self.NUM_PARAMS))
        fine = tuple(
            torch.randn(4, 5, generator=generator) * 6.0 for _ in range(self.NUM_PARAMS)
        )
        masks = _resolve_trunk_masks(empty, mid, fine, blocks=("p23",))
        self.assertTrue(all(masks["p23"]))
        pack, metrics = _build_lexicographic_grads(
            empty, mid, fine, masks, "coarse_first", eps=1e-12,
            include_metrics=True, blocks=("p23",),
        )
        mask = masks["p23"]
        self.assertAlmostEqual(
            _dot(pack["total"], mid, mask), _dot(mid, mid, mask), delta=1e-3
        )
        self.assertIn("post_projection_applied_p23_fine_mid", metrics)
        self.assertNotIn("post_projection_applied_p23_fine_higher", metrics)


if __name__ == "__main__":
    unittest.main()
