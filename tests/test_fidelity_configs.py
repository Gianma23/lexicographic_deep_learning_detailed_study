import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class FidelityPresetTests(unittest.TestCase):
    def _load(self, relative_path):
        with (REPO_ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_ht_capsnet_presets(self):
        expected_sizes = {
            "configs/capsnet/capsnet_cifar100.yaml": 32,
            "configs/capsnet/capsnet_cub200.yaml": 64,
            "configs/capsnet/capsnet_aircraft.yaml": 64,
        }
        for path, image_size in expected_sizes.items():
            with self.subTest(path=path):
                cfg = self._load(path)
                self.assertEqual(cfg["model"]["attn_heads"], 16)
                self.assertEqual(cfg["model"]["attn_key_dim"], 32)
                self.assertEqual(cfg["model"]["secondary_dims"], [64, 32, 16])
                self.assertEqual(cfg["dataset"]["image_size"], image_size)
                transforms = cfg["dataset"]["transforms"]
                self.assertEqual(transforms["mixup"], 0.2)
                self.assertEqual(transforms["mixup_mode"], "elem")
                self.assertEqual(transforms["mixup_pairing"], "random")
                self.assertEqual(cfg["dataloader"]["batch_size"], 32)
                self.assertEqual(cfg["train"]["epochs"], 200)
                self.assertFalse(cfg["train"]["amp"])
                self.assertEqual(cfg["optim"]["name"], "adam")
                self.assertEqual(cfg["optim"]["lr"], 0.001)
                self.assertEqual(cfg["optim"]["opt_eps"], 1.0e-7)
                self.assertEqual(cfg["scheduler"]["name"], "ht_capsnet_exponential")
                self.assertEqual(cfg["scheduler"]["start_epoch"], 10)
                self.assertEqual(cfg["scheduler"]["decay_rate"], 0.95)

    def test_hrn_presets(self):
        cifar = self._load("configs/hrn/hrn_cifar100.yaml")
        self.assertEqual(cifar["dataset"]["image_size"], 32)
        self.assertEqual(cifar["dataset"]["mean"], [0.5071, 0.4867, 0.4408])
        self.assertEqual(cifar["dataset"]["transforms"]["eval"]["resize_mode"], "none")
        self.assertEqual(cifar["dataloader"]["batch_size"], 8)
        self.assertEqual(cifar["train"]["epochs"], 200)
        self.assertEqual(cifar["optim"]["lr"], 0.002)
        self.assertEqual(cifar["scheduler"]["name"], "cosine")

        for path in (
            "configs/hrn/hrn_cub200.yaml",
            "configs/hrn/hrn_aircraft.yaml",
        ):
            with self.subTest(path=path):
                cfg = self._load(path)
                self.assertEqual(cfg["dataset"]["image_size"], 448)
                self.assertEqual(cfg["dataset"]["mean"], [0.5, 0.5, 0.5])
                self.assertEqual(cfg["dataset"]["std"], [0.5, 0.5, 0.5])
                manual = cfg["dataset"]["transforms"]["manual"]
                self.assertEqual(manual["resize_before_crop_size"], 550)
                self.assertEqual(cfg["dataloader"]["batch_size"], 8)
                self.assertEqual(cfg["train"]["epochs"], 200)
                self.assertEqual(cfg["optim"]["lr"], 0.002)
                self.assertEqual(cfg["scheduler"]["name"], "cosine")


if __name__ == "__main__":
    unittest.main()
