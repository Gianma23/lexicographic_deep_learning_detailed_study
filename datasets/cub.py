import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import BaseHierDataset, split_train_val_samples, taxonomy_from_parent_of
from .cub_tree import TREES


class CUBDataset(BaseHierDataset):
    """CUB-200 adapter with H-CAST order/family/species hierarchy."""

    def load_samples(self) -> List[Dict[str, Any]]:
        ann_file = self._annotation_file_for_split()
        if ann_file is not None:
            return self._read_json_samples(ann_file)

        train_samples, test_samples = self._load_from_split_folders()
        if not train_samples and not test_samples:
            train_samples, test_samples = self._load_from_official_files()

        if self.split == "test":
            return test_samples

        return split_train_val_samples(train_samples, split=self.split, cfg=self.cfg, stratify_level=-1)

    def _load_from_split_folders(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        candidates = [
            (self.root / "train", self.root / "test", "cub_folder"),
            (self.root / "images_split" / "train", self.root / "images_split" / "test", "cub_images_split"),
        ]

        for train_dir, test_dir, source in candidates:
            if not train_dir.exists() or not test_dir.exists():
                continue
            train_samples = self._read_folder_classes(train_dir, source=f"{source}_train")
            test_samples = self._read_folder_classes(test_dir, source=f"{source}_test")
            return train_samples, test_samples

        return [], []

    def _read_folder_classes(self, root_dir: Path, source: str) -> List[Dict[str, Any]]:
        class_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
        fallback_species = {p.name: i for i, p in enumerate(class_dirs)}

        samples: List[Dict[str, Any]] = []
        for class_dir in class_dirs:
            species = self._species_from_class_name(class_dir.name, fallback_species[class_dir.name])
            if species < 0 or species >= len(TREES):
                continue

            tree = TREES[species]
            order = int(tree[1]) - 1
            family = int(tree[2]) - 1
            for img in class_dir.rglob("*"):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                samples.append(
                    {
                        "image": img,
                        "labels": [order, family, species],
                        "meta": {"source": source, "class": class_dir.name},
                    }
                )

        return samples

    @staticmethod
    def _species_from_class_name(class_name: str, fallback: int) -> int:
        match = re.match(r"^(\d+)", class_name)
        if match:
            species = int(match.group(1)) - 1
            if species >= 0:
                return species
        return fallback

    def _load_from_official_files(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        roots = [self.root, self.root / "CUB_200_2011"]
        for base in roots:
            images_txt = base / "images.txt"
            labels_txt = base / "image_class_labels.txt"
            split_txt = base / "train_test_split.txt"
            images_dir = base / "images"

            if not (images_txt.exists() and labels_txt.exists() and split_txt.exists() and images_dir.exists()):
                continue

            image_map = self._read_int_str_map(images_txt)
            class_map = self._read_int_int_map(labels_txt)
            split_map = self._read_int_int_map(split_txt)

            train_samples: List[Dict[str, Any]] = []
            test_samples: List[Dict[str, Any]] = []
            for image_id in sorted(image_map.keys()):
                if image_id not in class_map or image_id not in split_map:
                    continue

                species = int(class_map[image_id]) - 1
                if species < 0 or species >= len(TREES):
                    continue

                tree = TREES[species]
                order = int(tree[1]) - 1
                family = int(tree[2]) - 1

                rel = image_map[image_id]
                image_path = images_dir / rel
                if not image_path.exists():
                    image_path = base / rel

                sample = {
                    "image": image_path,
                    "labels": [order, family, species],
                    "meta": {"source": "cub_official", "image_id": image_id},
                }

                if int(split_map[image_id]) == 1:
                    train_samples.append(sample)
                else:
                    test_samples.append(sample)

            return train_samples, test_samples

        return [], []

    @staticmethod
    def _read_int_str_map(path: Path) -> Dict[int, str]:
        out: Dict[int, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = line.strip().split(maxsplit=1)
                if len(row) != 2:
                    continue
                try:
                    out[int(row[0])] = row[1]
                except ValueError:
                    continue
        return out

    @staticmethod
    def _read_int_int_map(path: Path) -> Dict[int, int]:
        out: Dict[int, int] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = line.strip().split(maxsplit=1)
                if len(row) != 2:
                    continue
                try:
                    out[int(row[0])] = int(row[1])
                except ValueError:
                    continue
        return out

    def load_taxonomy(self):
        tax = super().load_taxonomy()
        if tax is not None:
            return tax

        parent_of = {
            1: {int(s["labels"][1]): int(s["labels"][0]) for s in self.samples},
            2: {int(s["labels"][2]): int(s["labels"][1]) for s in self.samples},
        }
        levels = list(self.cfg.dataset.get("levels", [])) or ["order", "family", "species"]
        return taxonomy_from_parent_of(parent_of, levels)
