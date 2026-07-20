from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from datasets import build_dataloader
from datasets.aircraft import AircraftDataset
from models.hiercos.model import HierCosModel
from models.orthonormal_plugin.losses import compute_loss
from models.orthonormal_plugin.topology import build_topology
from train.evaluation import evaluate_batch


class OfficialAircraftHierarchyTest(unittest.TestCase):
    def _write_official_download(self, root: Path) -> Path:
        data_root = root / "fgvc-aircraft-2013b" / "data"
        images_root = data_root / "images"
        images_root.mkdir(parents=True)

        (data_root / "manufacturers.txt").write_text("Maker B\nMaker A\n", encoding="utf-8")
        (data_root / "families.txt").write_text("Family Y\nFamily X\n", encoding="utf-8")
        (data_root / "variants.txt").write_text(
            "Variant C\nVariant A\nVariant B\n",
            encoding="utf-8",
        )

        rows = [
            ("1", "Variant C", "Family X", "Maker A"),
            ("2", "Variant A", "Family Y", "Maker B"),
            ("3", "Variant B", "Family X", "Maker A"),
        ]
        for split in ("train", "val", "test"):
            variant_lines = []
            family_lines = []
            manufacturer_lines = []
            for suffix, variant, family, manufacturer in rows:
                image_id = f"{split}_{suffix}"
                Image.new("RGB", (8, 8), color=(32, 64, 96)).save(
                    images_root / f"{image_id}.jpg"
                )
                variant_lines.append(f"{image_id} {variant}")
                family_lines.append(f"{image_id} {family}")
                manufacturer_lines.append(f"{image_id} {manufacturer}")

            (data_root / f"images_variant_{split}.txt").write_text(
                "\n".join(variant_lines) + "\n",
                encoding="utf-8",
            )
            (data_root / f"images_family_{split}.txt").write_text(
                "\n".join(family_lines) + "\n",
                encoding="utf-8",
            )
            (data_root / f"images_manufacturer_{split}.txt").write_text(
                "\n".join(manufacturer_lines) + "\n",
                encoding="utf-8",
            )

        # A conflicting legacy file must not influence the official mapping.
        (root / "Air.csv").write_text('"Variant A",99\n', encoding="utf-8")
        return data_root

    @staticmethod
    def _config(root: Path):
        return OmegaConf.create(
            {
                "dataset": {
                    "name": "fgvc-aircraft",
                    "root": str(root),
                    "hierarchy_depth": 3,
                    "image_size": 8,
                    "levels": ["manufacturer", "family", "variant"],
                    "transforms": {
                        "normalization": "none",
                        "fixed_resize_only": True,
                        "fixed_resize_interpolation": "bilinear",
                    },
                },
                "dataloader": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "drop_last_train": False,
                    "drop_last_eval": False,
                },
                "train": {"seed": 0},
                "model": {
                    "loss": "global_softmax_ce_reg",
                    "weight_mode": "kl_leaf",
                    "alpha": 0.1,
                },
            }
        )

    def test_uses_official_class_order_and_parallel_annotations(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            self._write_official_download(root)

            dataset = AircraftDataset(cfg=self._config(root), split="train")

            self.assertEqual(dataset.num_classes_per_level, [2, 2, 3])
            self.assertEqual(
                [sample["labels"] for sample in dataset.samples],
                [
                    [1, 1, 0],
                    [0, 0, 1],
                    [1, 1, 2],
                ],
            )
            self.assertEqual(
                dataset.taxonomy["parent_of"],
                {
                    1: {0: 0, 1: 1},
                    2: {0: 1, 1: 0, 2: 1},
                },
            )
            self.assertEqual(dataset.samples[0]["meta"]["manufacturer_name"], "Maker A")
            self.assertEqual(dataset.samples[0]["meta"]["family_name"], "Family X")

    def test_official_hierarchy_flows_through_hiercos_loss_and_decoding(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            self._write_official_download(root)
            cfg = self._config(root)

            split_data = {
                split: build_dataloader(cfg=cfg, split=split)
                for split in ("train", "val", "test")
            }
            train_loader, num_classes, taxonomy = split_data["train"]
            self.assertEqual(num_classes, [2, 2, 3])
            for split, (loader, split_num_classes, split_taxonomy) in split_data.items():
                self.assertEqual(split_num_classes, num_classes, msg=split)
                self.assertEqual(split_taxonomy, taxonomy, msg=split)
                self.assertEqual(len(loader.dataset), 3, msg=split)

            _, batch_labels, _ = next(iter(train_loader))
            self.assertEqual(tuple(batch_labels.shape), (2, 3))

            topology = build_topology(
                num_classes_per_level=num_classes,
                taxonomy=taxonomy,
                owner="Hier-COS integration test",
            )
            expected_leaf_paths = torch.tensor(
                [
                    [1, 1, 0],
                    [0, 0, 1],
                    [1, 1, 2],
                ],
                dtype=torch.long,
            )
            torch.testing.assert_close(topology["leaf_to_level_local"], expected_leaf_paths)

            # Global node ids are maker=[0,1], family=[2,3], variant=[4,5,6].
            # The Family X subspace must contain Maker A, Family X, Variant C,
            # and Variant B, and no unrelated nodes.
            expected_family_x_mask = torch.tensor(
                [False, True, False, True, True, False, True]
            )
            torch.testing.assert_close(
                topology["level_subspace_masks"][1][1],
                expected_family_x_mask,
            )

            model = HierCosModel(
                num_classes_per_level=num_classes,
                taxonomy=taxonomy,
                variant="haframe_wide_resnet",
                pretrained=False,
                fixed_frame_mode="identity",
                wide_depth=10,
                wide_widen_factor=1,
            )
            torch.testing.assert_close(model.leaf_to_level_local, expected_leaf_paths)
            model_output = model(torch.randn(2, 3, 32, 32))
            self.assertEqual(
                [tuple(scores.shape) for scores in model_output["logits_per_level"]],
                [(2, 2), (2, 2), (2, 3)],
            )
            model_loss, _ = compute_loss(
                output=model_output,
                targets=expected_leaf_paths[:2],
                cfg=cfg,
            )
            self.assertTrue(bool(torch.isfinite(model_loss)))
            model_loss.backward()

            node_logits = torch.zeros((2, 7), dtype=torch.float64, requires_grad=True)
            with torch.no_grad():
                node_logits[0, [1, 3, 4]] = 1.0  # Variant C official path.
                node_logits[1, [0, 2, 5]] = 1.0  # Variant A official path.

            logits_per_level = []
            squared_logits = node_logits.pow(2)
            for mask in topology["level_subspace_masks"]:
                scores_sq = squared_logits @ mask.to(dtype=node_logits.dtype).transpose(0, 1)
                logits_per_level.append(scores_sq.clamp_min(0.0).sqrt())

            output = {
                "logits_per_level": logits_per_level,
                "effective_probs_per_level": [
                    torch.softmax(scores, dim=-1) for scores in logits_per_level
                ],
                "node_logits": node_logits,
                "hiercos_level_node_ids": topology["level_node_ids"],
                "leaf_to_level_local": topology["leaf_to_level_local"],
                "node_prob_weights": topology["node_prob_weights"],
            }
            targets = expected_leaf_paths[:2]
            loss, _ = compute_loss(output=output, targets=targets, cfg=cfg)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertIsNotNone(node_logits.grad)

            metrics = evaluate_batch(output=output, targets=targets, taxonomy=taxonomy)
            self.assertEqual(metrics["fpa_independent"], 1.0)
            self.assertEqual(metrics["fpa_topdown"], 1.0)
            self.assertEqual(metrics["tice_independent"], 0.0)

    def test_rejects_inconsistent_official_parent_annotations(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            data_root = self._write_official_download(root)
            family_test = data_root / "images_family_test.txt"
            family_test.write_text(
                family_test.read_text(encoding="utf-8").replace(
                    "test_1 Family X",
                    "test_1 Family Y",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Inconsistent official FGVC-Aircraft"):
                AircraftDataset(cfg=self._config(root), split="train")

    def test_rejects_misaligned_parallel_annotation_files(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            data_root = self._write_official_download(root)
            manufacturer_val = data_root / "images_manufacturer_val.txt"
            lines = manufacturer_val.read_text(encoding="utf-8").splitlines()
            manufacturer_val.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "do not contain identical image IDs"):
                AircraftDataset(cfg=self._config(root), split="val")


if __name__ == "__main__":
    unittest.main()
