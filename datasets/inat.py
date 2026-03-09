import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseHierDataset, infer_parent_of_from_samples, split_train_val_samples, taxonomy_from_parent_of


class INatDataset(BaseHierDataset):
    """iNaturalist adapter supporting iNat18 and iNat21-mini style metadata."""

    def load_samples(self) -> List[Dict[str, Any]]:
        ann_file = self._annotation_file_for_split()
        if ann_file is not None:
            if ann_file.suffix.lower() == ".json":
                return self._read_json_samples(ann_file)
            return self._read_txt_annotations(ann_file)

        inat18_samples = self._load_inat18_split()
        if inat18_samples is not None:
            return inat18_samples

        return self._load_inat21_split()

    def _load_inat18_split(self) -> Optional[List[Dict[str, Any]]]:
        train_path = self._find_existing([self.root / "iNaturalist18_train.txt"])
        val_path = self._find_existing([self.root / "iNaturalist18_val.txt"])
        test_path = self._find_existing([self.root / "iNaturalist18_test.txt"])

        if train_path is None and val_path is None and test_path is None:
            return None

        if self.split == "train":
            return self._read_inat18(train_path) if train_path is not None else []

        if self.split == "val":
            if val_path is not None:
                return self._read_inat18(val_path)
            if train_path is not None:
                pool = self._read_inat18(train_path)
                return split_train_val_samples(pool, split="val", cfg=self.cfg, stratify_level=-1)
            return []

        # test split
        if test_path is not None:
            return self._read_inat18(test_path)
        if val_path is not None:
            return self._read_inat18(val_path)
        if train_path is not None:
            pool = self._read_inat18(train_path)
            return split_train_val_samples(pool, split="val", cfg=self.cfg, stratify_level=-1)
        return []

    def _load_inat21_split(self) -> List[Dict[str, Any]]:
        train_path = self._find_inat21_split_file("train")
        val_path = self._find_inat21_split_file("val")
        test_path = self._find_inat21_split_file("test")
        trainval_path = self._find_inat21_split_file("trainval")

        if self.split == "train":
            if train_path is not None:
                return self._read_txt_annotations(train_path)
            if trainval_path is not None:
                pool = self._read_txt_annotations(trainval_path)
                return split_train_val_samples(pool, split="train", cfg=self.cfg, stratify_level=-1)
            return []

        if self.split == "val":
            if val_path is not None:
                return self._read_txt_annotations(val_path)
            if train_path is not None:
                pool = self._read_txt_annotations(train_path)
                return split_train_val_samples(pool, split="val", cfg=self.cfg, stratify_level=-1)
            if trainval_path is not None:
                pool = self._read_txt_annotations(trainval_path)
                return split_train_val_samples(pool, split="val", cfg=self.cfg, stratify_level=-1)
            return []

        # test split
        if test_path is not None:
            return self._read_txt_annotations(test_path)
        if val_path is not None:
            return self._read_txt_annotations(val_path)
        if train_path is not None:
            pool = self._read_txt_annotations(train_path)
            return split_train_val_samples(pool, split="val", cfg=self.cfg, stratify_level=-1)
        if trainval_path is not None:
            pool = self._read_txt_annotations(trainval_path)
            return split_train_val_samples(pool, split="val", cfg=self.cfg, stratify_level=-1)
        return []

    def _find_inat21_split_file(self, split_name: str) -> Optional[Path]:
        names_by_split = {
            "train": [
                "inat21_mini_train.txt",
                "iNat21_mini_train.txt",
                "inat21_train.txt",
                "iNat21_train.txt",
            ],
            "val": [
                "inat21_mini_val.txt",
                "iNat21_mini_val.txt",
                "inat21_val.txt",
                "iNat21_val.txt",
            ],
            "test": [
                "inat21_mini_test.txt",
                "iNat21_mini_test.txt",
                "inat21_test.txt",
                "iNat21_test.txt",
            ],
            "trainval": [
                "inat21_mini_trainval.txt",
                "iNat21_mini_trainval.txt",
                "inat21_trainval.txt",
                "iNat21_trainval.txt",
            ],
        }

        roots = [self.root, self.root / "data", Path("data")]
        candidates = []
        for root in roots:
            for name in names_by_split.get(split_name, []):
                candidates.append(root / name)
        return self._find_existing(candidates)

    @staticmethod
    def _find_existing(candidates: List[Path]) -> Optional[Path]:
        for path in candidates:
            if path.exists():
                return path
        return None

    def _read_txt_annotations(self, txt_path: Path) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = line.strip().rsplit(maxsplit=3)
                if len(row) < 4:
                    continue
                rel = row[0]
                try:
                    species = int(row[1])
                    family = int(row[2])
                    order = int(row[3])
                except ValueError:
                    continue

                image_path = self._resolve_image_path(rel, txt_path)
                samples.append(
                    {
                        "image": image_path,
                        "labels": [order, family, species],
                        "meta": {"source": "inat_txt", "raw": row, "list_file": str(txt_path)},
                    }
                )
        return samples

    def _resolve_image_path(self, rel_path: str, list_path: Path) -> Path:
        rel = Path(rel_path)
        if rel.is_absolute():
            return rel

        candidates = [
            self.root / rel,
            list_path.parent / rel,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

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
