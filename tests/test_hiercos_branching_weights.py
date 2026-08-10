import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from models.hiercos.losses import _shared_level_weights
from train.config_validation import validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_COUNTS = [8, 20, 100]
LEVEL_NODE_IDS = [torch.arange(count) for count in CLASS_COUNTS]


def _native_cfg(weight_mode: str, weight_beta=None):
    model = {
        "name": "hiercos",
        "weight_mode": weight_mode,
    }
    if weight_beta is not None:
        model["weight_beta"] = weight_beta
    return SimpleNamespace(model=model)


def _weights(cfg):
    return _shared_level_weights(
        output={"node_prob_weights": torch.tensor([0.162, 0.225, 0.613])},
        cfg=cfg,
        level_node_ids=LEVEL_NODE_IDS,
        num_levels=len(CLASS_COUNTS),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


class HierCosBranchingWeightTests(unittest.TestCase):
    def test_cumulative_branching_uses_class_count_power_family(self):
        counts = torch.tensor(CLASS_COUNTS, dtype=torch.float64)
        for beta in (0.0, 0.5, 1.0, 1.5):
            with self.subTest(beta=beta):
                actual = _weights(_native_cfg("cumulative_branching", beta))
                expected = counts.pow(beta)
                expected = expected / expected.sum()
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_cumulative_branching_defaults_to_beta_half(self):
        actual = _weights(_native_cfg("cumulative_branching"))
        expected = torch.sqrt(torch.tensor(CLASS_COUNTS, dtype=torch.float64))
        expected = expected / expected.sum()
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_cumulative_branching_stays_finite_for_large_finite_beta(self):
        actual = _weights(_native_cfg("cumulative_branching", 1.0e308))
        expected = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_marginal_branching_uses_coarse_reference_unit(self):
        actual = _weights(_native_cfg("marginal_branching"))
        expected_scores = torch.tensor([1.0, 20.0 / 8.0, 100.0 / 20.0], dtype=torch.float64)
        expected = expected_scores / expected_scores.sum()
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_cumulative_beta_must_be_finite_nonnegative_number(self):
        for invalid in (-0.1, float("inf"), True, "not-a-number"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "weight_beta"):
                    _weights(_native_cfg("cumulative_branching", invalid))

    def test_native_static_config_accepts_modes_and_validates_beta(self):
        with (REPO_ROOT / "configs/hiercos/hiercos_cifar100.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            base = yaml.safe_load(handle)

        cumulative = copy.deepcopy(base)
        cumulative["model"]["weight_mode"] = "cumulative_branching"
        cumulative["model"]["weight_beta"] = 0.5
        validate_config(cumulative)

        marginal = copy.deepcopy(base)
        marginal["model"]["weight_mode"] = "marginal_branching"
        validate_config(marginal)

        invalid = copy.deepcopy(cumulative)
        invalid["model"]["weight_beta"] = -1.0
        with self.assertRaisesRegex(ValueError, "model.weight_beta"):
            validate_config(invalid)


if __name__ == "__main__":
    unittest.main()
