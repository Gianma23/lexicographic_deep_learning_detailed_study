import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


def infer_parent_of_from_samples(samples: List[Dict[str, Any]], depth: int) -> Optional[Dict[int, Dict[int, int]]]:
    parent_of: Dict[int, Dict[int, int]] = {}
    for level in range(1, depth):
        mapping: Dict[int, int] = {}
        for s in samples:
            labels = s["labels"]
            p = int(labels[level - 1])
            c = int(labels[level])
            if c in mapping and mapping[c] != p:
                return None
            mapping[c] = p
        parent_of[level] = mapping
    return parent_of


def taxonomy_from_parent_of(parent_of: Optional[Dict[int, Dict[int, int]]], levels: Optional[List[str]] = None):
    if not parent_of:
        return None
    return {
        "levels": levels,
        "parent_of": {int(k): {int(ck): int(pk) for ck, pk in v.items()} for k, v in parent_of.items()},
    }


class BaseHierDataset(Dataset, ABC):
    def __init__(self, cfg: Any, split: str, transform=None):
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.root = Path(str(cfg.dataset.root))
        self.depth = int(cfg.dataset.get("hierarchy_depth", 3))
        self.allow_synthetic = bool(cfg.dataset.get("allow_synthetic_fallback", True))

        self.samples = self.load_samples()
        if not self.samples and self.allow_synthetic:
            warnings.warn(
                f"No samples found for split={split} in dataset={cfg.dataset.name}. Using synthetic fallback.",
                RuntimeWarning,
            )
            self.samples = self._build_synthetic_samples()

        if not self.samples:
            raise RuntimeError(f"No samples found for split={split} in dataset={cfg.dataset.name}")

        self.samples, self.level_id_maps = self._remap_labels_to_contiguous(self.samples, self.depth)
        self.num_classes_per_level = self._compute_num_classes_per_level(self.samples, self.depth)
        taxonomy = self.load_taxonomy()
        if taxonomy is not None:
            taxonomy = self._remap_taxonomy_ids(taxonomy)
        self.taxonomy = taxonomy or self._infer_taxonomy()

    @abstractmethod
    def load_samples(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def load_taxonomy(self) -> Optional[Dict[str, Any]]:
        tax_file = self.cfg.dataset.get("taxonomy_file")
        if not tax_file:
            return None

        tax_path = self.root / str(tax_file)
        if not tax_path.exists():
            warnings.warn(f"Taxonomy file not found: {tax_path}. Falling back to inference.", RuntimeWarning)
            return None

        with tax_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _infer_taxonomy(self) -> Optional[Dict[str, Any]]:
        parent_of = infer_parent_of_from_samples(self.samples, self.depth)
        if parent_of is None:
            warnings.warn("Could not infer a consistent taxonomy from labels. Taxonomy metrics will be skipped.", RuntimeWarning)
            return None

        levels = list(self.cfg.dataset.get("levels", [])) or None
        return taxonomy_from_parent_of(parent_of, levels)

    def _remap_taxonomy_ids(self, taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        if "parent_of" not in taxonomy:
            return taxonomy

        remapped: Dict[int, Dict[int, int]] = {}
        for level_key, mapping in taxonomy["parent_of"].items():
            level = int(level_key)
            if level <= 0 or level >= self.depth:
                continue
            child_map = self.level_id_maps[level]
            parent_map = self.level_id_maps[level - 1]
            fixed = {}
            for child_old, parent_old in mapping.items():
                child_old = int(child_old)
                parent_old = int(parent_old)
                if child_old in child_map and parent_old in parent_map:
                    fixed[child_map[child_old]] = parent_map[parent_old]
            remapped[level] = fixed

        out = dict(taxonomy)
        out["parent_of"] = remapped
        return out

    @staticmethod
    def _remap_labels_to_contiguous(samples: List[Dict[str, Any]], depth: int):
        per_level_values = [sorted({int(s["labels"][l]) for s in samples}) for l in range(depth)]
        maps = [{old: idx for idx, old in enumerate(values)} for values in per_level_values]

        remapped = []
        for s in samples:
            labels = [maps[l][int(s["labels"][l])] for l in range(depth)]
            remapped.append({"image": s.get("image"), "labels": labels, "meta": s.get("meta", {})})

        return remapped, maps

    @staticmethod
    def _compute_num_classes_per_level(samples: List[Dict[str, Any]], depth: int) -> List[int]:
        classes = [set() for _ in range(depth)]
        for s in samples:
            for level, value in enumerate(s["labels"]):
                classes[level].add(int(value))
        return [max(values) + 1 if values else 1 for values in classes]

    def _build_synthetic_samples(self, n: int = 256) -> List[Dict[str, Any]]:
        class_counts = [4, 8, 16][: self.depth]
        while len(class_counts) < self.depth:
            class_counts.append(class_counts[-1] * 2)

        g = torch.Generator().manual_seed(7)
        samples: List[Dict[str, Any]] = []
        for idx in range(n):
            labels = []
            for level, c in enumerate(class_counts):
                if level == 0:
                    label = int(torch.randint(0, c, (1,), generator=g).item())
                else:
                    prev = labels[level - 1]
                    span = max(1, c // class_counts[level - 1])
                    start = prev * span
                    label = int(min(c - 1, start + int(torch.randint(0, span, (1,), generator=g).item())))
                labels.append(label)
            samples.append({"image": None, "labels": labels, "meta": {"synthetic_id": idx}})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(sample["labels"], dtype=torch.long)
        return image, labels, sample.get("meta", {})

    def _load_image(self, image_path: Optional[Path]):
        image_size = int(self.cfg.dataset.get("image_size", 224))
        if image_path is None or not Path(image_path).exists():
            return Image.new("RGB", (image_size, image_size), color=(127, 127, 127))
        return Image.open(image_path).convert("RGB")

    def _read_json_samples(self, ann_path: Path) -> List[Dict[str, Any]]:
        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("samples", [])

        out: List[Dict[str, Any]] = []
        for row in data:
            labels = row.get("labels") or row.get("levels")
            if labels is None:
                continue
            labels = [int(x) for x in labels][: self.depth]
            if len(labels) != self.depth:
                continue
            image_rel = row.get("image") or row.get("path")
            image_path = self.root / str(image_rel) if image_rel else None
            out.append({"image": image_path, "labels": labels, "meta": row})
        return out

    def _annotation_file_for_split(self) -> Optional[Path]:
        ann_cfg = self.cfg.dataset.get("annotations", {})
        split_file = ann_cfg.get(self.split)
        if not split_file:
            return None
        path = self.root / str(split_file)
        if path.exists():
            return path
        return None
