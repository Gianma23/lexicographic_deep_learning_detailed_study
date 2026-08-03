import copy
import unittest
from pathlib import Path

import yaml

from train.config_validation import validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]

# CLI-override-style HCC block, matching what the run_*_hcc_grid.sh scripts
# pass on top of each model's plain baseline config.
HCC_CFG = {
    "enabled": True,
    "temperature": 10,
    "eps": 1e-12,
    "alpha_schedule": "step",
    "alpha_start_epoch": 0,
    "alpha_ramp_epochs": 0,
}


class HccModelSupportTests(unittest.TestCase):
    def _load(self, relative_path):
        with (REPO_ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _with_hcc(self, relative_path):
        cfg = copy.deepcopy(self._load(relative_path))
        cfg["hcc"] = copy.deepcopy(HCC_CFG)
        return cfg

    def test_hcc_on_hcast_passes_static_validation(self):
        # H-CAST already ships a dedicated hcc preset config.
        validate_config(self._load("configs/hcast/hcast_hcc_cifar100.yaml"))

    def test_hcc_on_ht_capsnet_passes_static_validation_from_baseline(self):
        capsnet = self._with_hcc("configs/capsnet/capsnet_cifar100.yaml")
        validate_config(capsnet)

    def test_hcc_on_hrn_requires_level_marginal_loss(self):
        hrn = self._with_hcc("configs/hrn/hrn_cifar100.yaml")
        with self.assertRaisesRegex(ValueError, "model.loss: level_marginal"):
            validate_config(hrn)
        hrn["model"]["loss"] = "level_marginal"
        validate_config(hrn)

    def test_hcc_on_hiercos_supports_every_native_loss_and_fixed_frame(self):
        for loss_mode in ("kl_reg", "global_softmax_ce_reg", "level_softmax_ce_reg"):
            for frame_mode, frame_per_level in (
                ("orthonormal_random", False),
                ("orthonormal_random", True),
                ("orthonormal_block_random", False),
                ("identity", False),
            ):
                with self.subTest(
                    loss_mode=loss_mode,
                    frame_mode=frame_mode,
                    frame_per_level=frame_per_level,
                ):
                    hiercos = self._with_hcc("configs/hiercos/hiercos_cifar100.yaml")
                    hiercos["model"]["loss"] = loss_mode
                    hiercos["model"]["fixed_frame_mode"] = frame_mode
                    hiercos["model"]["fixed_frame_per_level"] = frame_per_level
                    validate_config(hiercos)

    def test_hcc_on_hiercos_rejects_projection_enabled(self):
        hiercos = self._with_hcc("configs/hiercos/hiercos_cifar100.yaml")
        hiercos["model"]["projection"] = {"enabled": True}
        with self.assertRaisesRegex(ValueError, "model.projection.enabled=true"):
            validate_config(hiercos)

    def test_hcc_is_rejected_for_lhdnn(self):
        lhdnn = self._with_hcc("configs/lhdnn/lhdnn_cifar100.yaml")
        with self.assertRaisesRegex(ValueError, "supported only for model.name in"):
            validate_config(lhdnn)

    # Note: dataset.hierarchy_depth != 3 is already unreachable under the
    # repo's only supported `runtime.protocol` (corrected_unified_v1), which
    # enforces depth==3 for every model well before config_validation.py's
    # HCC-specific depth check ever runs. No test mutates hierarchy_depth
    # here for that reason; see `train/config_validation.py`'s
    # `_validate_common_sections`.


if __name__ == "__main__":
    unittest.main()
