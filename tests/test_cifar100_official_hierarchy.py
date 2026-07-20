from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from datasets.cifar100 import (
    B_CNN_COARSE_TO_SUPER,
    CIFAR100Dataset,
    load_official_cifar100_fine_to_coarse,
)

# Exact CIFAR-100 fine-ID -> official coarse-ID relation used by B-CNN.
# This regression oracle intentionally remains independent of the runtime
# metadata loader.
B_CNN_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

B_CNN_EXPECTED_COARSE_TO_SUPER = [
    0, 0, 1, 2, 1, 2, 2, 3, 4, 5,
    5, 4, 4, 3, 6, 4, 4, 1, 7, 7,
]


class OfficialCifar100HierarchyTest(unittest.TestCase):
    @staticmethod
    def _write_official_train_file(root: Path, mapping: list[int]) -> None:
        archive_root = root / "cifar-100-python"
        archive_root.mkdir(parents=True)
        payload = {
            b"fine_labels": list(range(100)),
            b"coarse_labels": [int(mapping[fine]) for fine in range(100)],
        }
        with (archive_root / "train").open("wb") as handle:
            pickle.dump(payload, handle)

    @staticmethod
    def _config(root: Path, depth: int = 2):
        return OmegaConf.create(
            {
                "dataset": {
                    "name": "cifar-100",
                    "root": str(root),
                    "hierarchy_depth": depth,
                    "levels": (
                        ["coarse", "fine"]
                        if depth == 2
                        else ["super", "coarse", "fine"]
                    ),
                    "download": False,
                    "val_split_ratio": 0.0,
                },
                "train": {"seed": 0},
            }
        )

    def test_derives_exact_bcnn_fine_to_coarse_mapping_from_official_pickle(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            self._write_official_train_file(root, B_CNN_FINE_TO_COARSE)

            self.assertEqual(
                load_official_cifar100_fine_to_coarse(root),
                B_CNN_FINE_TO_COARSE,
            )

    def test_dataset_uses_downloaded_coarse_labels(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            expected = [(fine * 7) % 20 for fine in range(100)]
            self._write_official_train_file(root, expected)

            fake_torchvision_dataset = type(
                "FakeCIFAR100",
                (),
                {
                    "data": np.zeros((100, 2, 2, 3), dtype=np.uint8),
                    "targets": list(range(100)),
                },
            )()
            with patch("datasets.cifar100.CIFAR100", return_value=fake_torchvision_dataset):
                dataset = CIFAR100Dataset(
                    cfg=self._config(root, depth=2),
                    split="test",
                )

            self.assertEqual(dataset.num_classes_per_level, [20, 100])
            self.assertEqual(
                [sample["labels"] for sample in dataset.samples],
                [[expected[fine], fine] for fine in range(100)],
            )

    def test_three_level_mode_uses_exact_bcnn_super_grouping(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            self._write_official_train_file(root, B_CNN_FINE_TO_COARSE)

            fake_torchvision_dataset = type(
                "FakeCIFAR100",
                (),
                {
                    "data": np.zeros((100, 2, 2, 3), dtype=np.uint8),
                    "targets": list(range(100)),
                },
            )()
            with patch("datasets.cifar100.CIFAR100", return_value=fake_torchvision_dataset):
                dataset = CIFAR100Dataset(
                    cfg=self._config(root, depth=3),
                    split="test",
                )

            self.assertEqual(dataset.num_classes_per_level, [8, 20, 100])
            self.assertEqual(B_CNN_COARSE_TO_SUPER, B_CNN_EXPECTED_COARSE_TO_SUPER)
            self.assertEqual(
                [sample["labels"] for sample in dataset.samples],
                [
                    [
                        B_CNN_EXPECTED_COARSE_TO_SUPER[B_CNN_FINE_TO_COARSE[fine]],
                        B_CNN_FINE_TO_COARSE[fine],
                        fine,
                    ]
                    for fine in range(100)
                ],
            )
            self.assertEqual(
                dataset.taxonomy["parent_of"],
                {
                    1: {
                        coarse: B_CNN_EXPECTED_COARSE_TO_SUPER[coarse]
                        for coarse in range(20)
                    },
                    2: {
                        fine: B_CNN_FINE_TO_COARSE[fine]
                        for fine in range(100)
                    },
                },
            )


if __name__ == "__main__":
    unittest.main()
