import unittest

import torch

from models.hiercos.losses import (
    _asymmetric_global_log_probs,
    _resolve_softmax_detach,
    _weighted_target_ce_level_losses,
)


LEVEL_SIZES = (3, 5, 8)


def _topology(device="cpu"):
    """Three levels laid out contiguously over one node axis."""
    ids, start = [], 0
    for size in LEVEL_SIZES:
        ids.append(torch.arange(start, start + size, dtype=torch.long, device=device))
        start += size
    # Every leaf maps to itself at the fine level; coarse/mid indices are a
    # deterministic fan-in, which is all the CE needs.
    num_leaf = LEVEL_SIZES[-1]
    leaf_to_level_local = torch.stack(
        [
            torch.arange(num_leaf, device=device) % LEVEL_SIZES[0],
            torch.arange(num_leaf, device=device) % LEVEL_SIZES[1],
            torch.arange(num_leaf, device=device),
        ],
        dim=1,
    ).to(torch.long)
    return ids, leaf_to_level_local


def _ce(abs_node_logits, detach, level_node_ids, leaf_to_level_local, leaf_targets):
    return _weighted_target_ce_level_losses(
        abs_node_logits=abs_node_logits,
        level_node_ids=level_node_ids,
        leaf_targets=leaf_targets,
        leaf_to_level_local=leaf_to_level_local,
        level_weights=torch.ones(len(LEVEL_SIZES)),
        softmax_scope="global",
        softmax_detach=detach,
    )


class HierCosSoftmaxDetachTest(unittest.TestCase):
    """`model.softmax_detach` must keep the loss value and change only the Jacobian.

    Under a shared normaliser every level's loss gradient is dense over every
    other level's nodes, so a lower level writes straight into a higher level's
    head. That is the route the LH-DNN projection does not cover. Detaching the
    higher-priority levels inside each lower-priority level's normaliser must
    drive that leak to exactly zero while leaving the priority level's gradient
    dense, which is what keeps the cross-level conflict alive.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.level_node_ids, self.leaf_to_level_local = _topology()
        self.batch = 6
        self.leaf_targets = torch.randint(0, LEVEL_SIZES[-1], (self.batch,))

    def _logits(self):
        return torch.randn(self.batch, sum(LEVEL_SIZES), requires_grad=True)

    def test_loss_values_are_unchanged_by_detaching(self):
        logits = self._logits()
        base = _ce(logits, False, self.level_node_ids, self.leaf_to_level_local, self.leaf_targets)
        got = _ce(logits, True, self.level_node_ids, self.leaf_to_level_local, self.leaf_targets)
        for level, (want, have) in enumerate(zip(base, got)):
            self.assertAlmostEqual(
                float(want.detach()), float(have.detach()), places=6,
                msg=f"detaching changed the level {level} loss value",
            )

    def test_detach_confines_lower_levels_and_leaves_coarse_dense(self):
        logits = self._logits()
        losses = _ce(
            logits, True, self.level_node_ids, self.leaf_to_level_local, self.leaf_targets
        )
        for level, loss in enumerate(losses):
            grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            for other, nodes in enumerate(self.level_node_ids):
                norm = float(grad[:, nodes].norm())
                if other < level:
                    # Higher priority: the LH leak this flag exists to remove.
                    self.assertEqual(
                        norm, 0.0,
                        msg=f"level {level} leaked gradient onto higher level {other}",
                    )
                else:
                    self.assertGreater(
                        norm, 0.0,
                        msg=f"level {level} lost gradient on level {other}",
                    )

    def test_plain_global_softmax_does_leak(self):
        """The control: without the flag the leak is real, so the test above bites."""
        logits = self._logits()
        losses = _ce(logits, False, self.level_node_ids, self.leaf_to_level_local, self.leaf_targets)
        grad = torch.autograd.grad(losses[-1], logits, retain_graph=True)[0]
        higher = torch.cat(list(self.level_node_ids[:-1]))
        self.assertGreater(float(grad[:, higher].norm()), 0.0)

    def test_conflict_survives_detaching(self):
        """cos(fine, coarse) must stay negative: the coarse gradient stays dense.

        The conflict is a property of the *converged* regime, where a shared
        normaliser forces the three level targets to split one unit of mass:
        `<dL_L/dz, dL_K/dz> = ||p||^2 - p[n_L] - p[n_K]`, which is negative once
        each target holds a comparable share. Random logits concentrate on one
        arbitrary node instead and give a positive inner product, so the fixture
        below builds a confident, taxonomically valid path.
        """
        batch = 64
        targets = torch.randint(0, LEVEL_SIZES[-1], (batch,))
        raw = torch.full((batch, sum(LEVEL_SIZES)), -4.0)
        rows = torch.arange(batch)
        for level, nodes in enumerate(self.level_node_ids):
            local = self.leaf_to_level_local[targets, level]
            raw[rows, nodes[local]] = 4.0
        logits = raw.clone().requires_grad_(True)
        cosines = {}
        for mode in (False, True):
            losses = _ce(logits, mode, self.level_node_ids, self.leaf_to_level_local, targets)
            grads = [
                torch.autograd.grad(loss, logits, retain_graph=True)[0].flatten()
                for loss in losses
            ]
            cosines[mode] = float(
                torch.nn.functional.cosine_similarity(grads[2], grads[0], dim=0)
            )
        self.assertLess(cosines[False], 0.0, "shared normaliser should conflict")
        self.assertLess(
            cosines[True], 0.0,
            "detaching must preserve the sign of the cross-level conflict",
        )

    def test_asymmetric_log_probs_match_plain_global_in_value(self):
        logits = self._logits()
        want = torch.log_softmax(logits, dim=1)
        for got in _asymmetric_global_log_probs(
            abs_node_logits=logits,
            level_node_ids=self.level_node_ids,
        ):
            torch.testing.assert_close(got, want)

    def test_resolver_accepts_bool_like_values(self):
        for raw, want in ((True, True), (False, False), ("true", True), ("false", False)):
            class Cfg:
                model = {"softmax_detach": raw}
            self.assertIs(_resolve_softmax_detach(Cfg()), want, msg=f"raw={raw!r}")

    def test_detach_rejected_for_level_softmax(self):
        logits = self._logits()
        with self.assertRaises(ValueError):
            _weighted_target_ce_level_losses(
                abs_node_logits=logits,
                level_node_ids=self.level_node_ids,
                leaf_targets=self.leaf_targets,
                leaf_to_level_local=self.leaf_to_level_local,
                level_weights=torch.ones(len(LEVEL_SIZES)),
                softmax_scope="level",
                softmax_detach=True,
            )

    def test_resolver_defaults_to_false(self):
        class Cfg:
            model = {}

        class Configured:
            model = {"softmax_detach": True}

        self.assertIs(_resolve_softmax_detach(Cfg()), False)
        self.assertIs(_resolve_softmax_detach(Configured()), True)


if __name__ == "__main__":
    unittest.main()
