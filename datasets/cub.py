from pathlib import Path
from typing import Any, Dict, List

from .base import BaseHierDataset, taxonomy_from_parent_of
from .cub_tree import TREES


class CUBDataset(BaseHierDataset):
    """CUB-200 adapter with H-CAST order/family/species hierarchy."""

    def load_samples(self) -> List[Dict[str, Any]]:
        ann_file = self._annotation_file_for_split()
        if ann_file is not None:
            return self._read_json_samples(ann_file)

        split_dir = "train" if self.split == "train" else "test"
        folder = self.root / split_dir
        if not folder.exists():
            return []

        class_dirs = sorted([p for p in folder.iterdir() if p.is_dir()])
        class_to_idx = {p.name: i for i, p in enumerate(class_dirs)}

        samples: List[Dict[str, Any]] = []
        for class_dir in class_dirs:
            species = class_to_idx[class_dir.name]
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
                        "meta": {"source": "cub_folder", "class": class_dir.name},
                    }
                )

        return samples

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
