import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseHierDataset, infer_parent_of_from_samples, taxonomy_from_parent_of


class INatDataset(BaseHierDataset):
    """iNaturalist adapter supporting iNat18 and iNat21-mini style metadata."""

    def load_samples(self) -> List[Dict[str, Any]]:
        ann_file = self._annotation_file_for_split()
        if ann_file is not None:
            if ann_file.suffix.lower() == ".json":
                return self._read_json_samples(ann_file)
            return self._read_txt_annotations(ann_file)

        # iNat18 format: root/iNaturalist18_train.txt with class id + tree json.
        split_txt = "iNaturalist18_train.txt" if self.split == "train" else "iNaturalist18_val.txt"
        inat18_path = self.root / split_txt
        if inat18_path.exists():
            return self._read_inat18(inat18_path)

        # iNat21 mini format used in H-CAST lists: path class family order.
        inat21_default = Path("data") / ("inat21_mini_train.txt" if self.split == "train" else "inat21_mini_val.txt")
        if inat21_default.exists():
            return self._read_txt_annotations(inat21_default)

        return []

    def _read_txt_annotations(self, txt_path: Path) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                rel = parts[0]
                species = int(parts[1])
                family = int(parts[2])
                order = int(parts[3])
                samples.append(
                    {
                        "image": self.root / rel,
                        "labels": [order, family, species],
                        "meta": {"source": "inat_txt", "raw": parts},
                    }
                )
        return samples

    def _read_inat18(self, txt_path: Path) -> List[Dict[str, Any]]:
        tree_candidates = [
            self.root / "inat18_tree.json",
            self.root / "data" / "inat18_tree.json",
            Path("data") / "inat18_tree.json",
        ]

        tree_path: Optional[Path] = None
        for p in tree_candidates:
            if p.exists():
                tree_path = p
                break

        if tree_path is None:
            return []

        with tree_path.open("r", encoding="utf-8") as f:
            trees = json.load(f)

        samples: List[Dict[str, Any]] = []
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                rel = parts[0]
                species = int(parts[1])
                if species >= len(trees):
                    continue
                row = trees[species]
                order = int(row[3])
                super_cls = int(row[7])
                samples.append(
                    {
                        "image": self.root / rel,
                        "labels": [super_cls, order, species],
                        "meta": {"source": "inat18_txt", "raw": parts},
                    }
                )

        return samples

    def load_taxonomy(self):
        tax = super().load_taxonomy()
        if tax is not None:
            return tax

        parent_of = infer_parent_of_from_samples(self.samples, self.depth)
        levels = list(self.cfg.dataset.get("levels", [])) or None
        return taxonomy_from_parent_of(parent_of, levels)
