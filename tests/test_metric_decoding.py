"""Guards for the cached taxonomy tensors and the decode-once metric path.

These optimizations must not change a single metric value, so the tests compare
against straightforward reimplementations rather than stored numbers.
"""
import unittest

import torch

from train.evaluation import _mean_gt_rank_within_parent, evaluate_batch
from train.metrics import (
    _child_parent_tensors,
    _hierarchical_argmax_preds,
    average_hierarchical_distance,
    consistency_rate,
    decoded_preds,
    full_path_accuracy,
    per_level_top1,
    tice_score,
    weighted_average_precision,
)


NUM_CLASSES = [3, 5, 9]
TAXONOMY = {
    "parent_of": {
        1: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2},
        2: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4},
    }
}
HEADLINE = ("acc_level_independent_", "acc_level_topdown_", "weighted_ap_", "fpa_", "ahd_", "tice_")


def _scores(batch=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, n, generator=generator) for n in NUM_CLASSES]


def _targets(batch=6, seed=1):
    generator = torch.Generator().manual_seed(seed)
    fine = torch.randint(0, NUM_CLASSES[2], (batch,), generator=generator)
    middle = torch.tensor([TAXONOMY["parent_of"][2][int(c)] for c in fine])
    coarse = torch.tensor([TAXONOMY["parent_of"][1][int(c)] for c in middle])
    return torch.stack([coarse, middle, fine], dim=1)


def _naive_allowed(mapping, num_parents, num_children):
    allowed = torch.zeros((num_parents, num_children), dtype=torch.bool)
    for child, parent in mapping.items():
        if 0 <= child < num_children and 0 <= parent < num_parents:
            allowed[parent, child] = True
    return allowed


class CachedTaxonomyTensorTests(unittest.TestCase):
    def test_matches_element_wise_construction(self):
        for level, num_parents in ((1, NUM_CLASSES[0]), (2, NUM_CLASSES[1])):
            mapping = TAXONOMY["parent_of"][level]
            num_children = NUM_CLASSES[level]
            allowed, lookup = _child_parent_tensors(
                mapping, num_parents, num_children, torch.device("cpu")
            )
            self.assertTrue(torch.equal(allowed, _naive_allowed(mapping, num_parents, num_children)))
            for child, parent in mapping.items():
                self.assertEqual(int(lookup[child]), parent)

    def test_cache_returns_equal_tensors_on_repeat_calls(self):
        mapping = TAXONOMY["parent_of"][2]
        first = _child_parent_tensors(mapping, NUM_CLASSES[1], NUM_CLASSES[2], torch.device("cpu"))
        second = _child_parent_tensors(mapping, NUM_CLASSES[1], NUM_CLASSES[2], torch.device("cpu"))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_out_of_range_entries_are_handled_like_before(self):
        # A child pointing outside the parent range must never count as valid,
        # and a childless parent must fall back to an unmasked argmax.
        mapping = {0: 0, 1: 99}
        allowed, lookup = _child_parent_tensors(mapping, 2, 2, torch.device("cpu"))
        self.assertTrue(torch.equal(allowed, torch.tensor([[True, False], [False, False]])))
        self.assertEqual(int(lookup[0]), 0)
        self.assertEqual(int(lookup[1]), 99)

        taxonomy = {"parent_of": {1: mapping}}
        scores = [torch.tensor([[0.1, 0.9]]), torch.tensor([[0.7, 0.3]])]
        preds = _hierarchical_argmax_preds(scores, taxonomy)
        # Parent 1 has no children, so the fallback keeps the plain argmax.
        self.assertEqual(int(preds[0]), 1)
        self.assertEqual(int(preds[1]), 0)


class DecodeOnceTests(unittest.TestCase):
    def test_metrics_ignore_whether_preds_were_precomputed(self):
        scores, targets = _scores(), _targets()
        for enforce in (False, True):
            preds = decoded_preds(scores, TAXONOMY, enforce)
            self.assertEqual(
                per_level_top1(scores, targets, TAXONOMY, enforce),
                per_level_top1(scores, targets, TAXONOMY, enforce, preds=preds),
            )
            for metric in (weighted_average_precision, full_path_accuracy, average_hierarchical_distance):
                self.assertEqual(
                    metric(scores, targets, TAXONOMY, enforce),
                    metric(scores, targets, TAXONOMY, enforce, preds=preds),
                )
            self.assertEqual(
                tice_score(scores, TAXONOMY, enforce),
                tice_score(scores, TAXONOMY, enforce, preds=preds),
            )
            self.assertEqual(
                consistency_rate(scores, TAXONOMY, enforce),
                consistency_rate(scores, TAXONOMY, enforce, preds=preds),
            )

    def test_top_down_decoding_matches_a_naive_loop(self):
        scores, _ = _scores(batch=8, seed=3), None
        preds = _hierarchical_argmax_preds(scores, TAXONOMY)
        expected = [scores[0].argmax(dim=-1)]
        for level in (1, 2):
            mapping = TAXONOMY["parent_of"][level]
            allowed = _naive_allowed(mapping, NUM_CLASSES[level - 1], NUM_CLASSES[level])
            masked = scores[level].masked_fill(~allowed[expected[level - 1]], float("-inf"))
            expected.append(masked.argmax(dim=-1))
        for reference, actual in zip(expected, preds):
            self.assertTrue(torch.equal(reference, actual))


class GroundTruthRankTests(unittest.TestCase):
    """The vectorized rank must keep the per-sample loop's skip rules."""

    MASK = torch.tensor(
        [
            [True, True, False, False],   # parent 0 owns children 0 and 1
            [False, False, True, False],  # parent 1 owns child 2
            [False, False, False, False],  # parent 2 is childless
        ]
    )

    @staticmethod
    def _rank(probs, parents, children, mask):
        return _mean_gt_rank_within_parent(
            torch.tensor(probs),
            torch.tensor(parents, dtype=torch.long),
            torch.tensor(children, dtype=torch.long),
            mask,
        )

    def test_rank_counts_only_siblings(self):
        # Child 1 is beaten by child 0 among siblings, but child 2 scores higher
        # and belongs to another parent, so it must not affect the rank.
        probs = [[0.4, 0.3, 0.9, 0.0]]
        self.assertEqual(self._rank(probs, [0], [1], self.MASK), 2.0)
        self.assertEqual(self._rank(probs, [0], [0], self.MASK), 1.0)

    def test_ties_do_not_increase_the_rank(self):
        probs = [[0.5, 0.5, 0.0, 0.0]]
        self.assertEqual(self._rank(probs, [0], [0], self.MASK), 1.0)

    def test_samples_are_skipped_like_the_original_loop(self):
        probs = [[0.4, 0.3, 0.9, 0.0]]
        # Childless parent, parent that does not own the child, and out-of-range
        # parents or children all drop out of the average.
        self.assertIsNone(self._rank(probs, [2], [0], self.MASK))
        self.assertIsNone(self._rank(probs, [1], [0], self.MASK))
        self.assertIsNone(self._rank(probs, [-1], [0], self.MASK))
        self.assertIsNone(self._rank(probs, [99], [0], self.MASK))
        self.assertIsNone(self._rank(probs, [0], [99], self.MASK))

    def test_average_uses_only_usable_samples(self):
        probs = [[0.4, 0.3, 0.9, 0.0], [0.4, 0.3, 0.9, 0.0], [0.4, 0.3, 0.9, 0.0]]
        # Ranks 2 and 1 are usable; the middle sample's parent is childless.
        self.assertEqual(self._rank(probs, [0, 2, 0], [1, 0, 0], self.MASK), 1.5)

    def test_out_of_range_index_does_not_raise(self):
        probs = [[0.4, 0.3, 0.9, 0.0]]
        self.assertIsNone(self._rank(probs, [7], [7], self.MASK))


class DiagnosticsOptOutTests(unittest.TestCase):
    def test_headline_metrics_are_unaffected(self):
        output = {"logits_per_level": _scores()}
        targets = _targets()
        full = evaluate_batch(output, targets, TAXONOMY)
        lean = evaluate_batch(output, targets, TAXONOMY, include_diagnostics=False)
        headline_keys = [key for key in full if key.startswith(HEADLINE)]
        self.assertTrue(headline_keys)
        for key in headline_keys:
            self.assertEqual(full[key], lean[key], key)

    def test_only_diagnostics_are_dropped(self):
        output = {"logits_per_level": _scores()}
        targets = _targets()
        full = evaluate_batch(output, targets, TAXONOMY)
        lean = evaluate_batch(output, targets, TAXONOMY, include_diagnostics=False)
        self.assertTrue(set(lean).issubset(set(full)))
        self.assertTrue(set(full) - set(lean), 'expected some diagnostics to exist')
        self.assertFalse([key for key in lean if not key.startswith(HEADLINE)])

    def test_diagnostics_are_on_by_default_for_training(self):
        scores = _scores()
        output = {
            "logits_per_level": scores,
            "effective_logits_per_level": [score * 1.5 for score in scores],
        }
        metrics = evaluate_batch(output, _targets(), TAXONOMY)
        self.assertIn("proj_logit_delta_l1_level_2", metrics)
        self.assertIn("gt_parent_mass_post_l2", metrics)


if __name__ == "__main__":
    unittest.main()
