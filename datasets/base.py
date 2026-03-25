import json
import random
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


def infer_parent_of_from_samples(samples: List[Dict[str, Any]], depth: int) -> Optional[Dict[int, Dict[int, int]]]:
    """Infer child-to-parent mappings for each hierarchy level transition."""
    parent_of: Dict[int, Dict[int, int]] = {}
    for level in range(1, depth):
        mapping: Dict[int, int] = {}
        for s in samples:
            labels = s["labels"]
            p = int(labels[level - 1])
            c = int(labels[level])
            # inconsistent labels
            if c in mapping and mapping[c] != p:
                return None
            mapping[c] = p
        parent_of[level] = mapping
    return parent_of


def taxonomy_from_parent_of(parent_of: Optional[Dict[int, Dict[int, int]]], levels: Optional[List[str]] = None):
    """Convert level mappings into the normalized taxonomy dictionary format."""
    if not parent_of:
        return None
    return {
        "levels": levels,
        "parent_of": {int(k): {int(ck): int(pk) for ck, pk in v.items()} for k, v in parent_of.items()},
    }


def resolve_split_seed(cfg: Any) -> int:
    """Resolve the seed used for deterministic train/validation splitting."""
    split_seed = cfg.dataset.get("split_seed", None)
    if split_seed is not None:
        return int(split_seed)

    train_cfg = cfg.get("train", None) if hasattr(cfg, "get") else getattr(cfg, "train", None)
    if train_cfg is not None:
        seed = train_cfg.get("seed", None) if hasattr(train_cfg, "get") else getattr(train_cfg, "seed", None)
        if seed is not None:
            return int(seed)

    return 42


def resolve_val_split_ratio(cfg: Any) -> float:
    """Read and clamp the validation split ratio to ``[0.0, 0.99]``."""
    raw = cfg.dataset.get("val_split_ratio", 0.1)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.1
    return min(max(value, 0.0), 0.99)


def stratified_train_val_indices(stratify_labels: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """Build deterministic stratified train/validation indices from labels."""
    if not stratify_labels:
        return [], []
    if val_ratio <= 0.0:
        return list(range(len(stratify_labels))), []

    per_class: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(stratify_labels):
        per_class[int(label)].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for cls_id, indices in sorted(per_class.items(), key=lambda item: item[0]):
        shuffled = list(indices)
        random.Random(seed * 1_000_003 + cls_id).shuffle(shuffled)

        n = len(shuffled)
        if n <= 1:
            n_val = 0
        else:
            n_val = int(round(n * val_ratio))
            if n_val <= 0:
                n_val = 1
            if n_val >= n:
                n_val = n - 1

        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    if not val_indices and len(stratify_labels) > 1:
        shuffled = list(range(len(stratify_labels)))
        random.Random(seed).shuffle(shuffled)
        val_indices = [shuffled[0]]
        train_indices = shuffled[1:]

    return sorted(train_indices), sorted(val_indices)


def split_train_val_samples(
    samples: List[Dict[str, Any]],
    split: str,
    cfg: Any,
    stratify_level: int = -1,
) -> List[Dict[str, Any]]:
    """Return split-specific samples using a stratified train/val partition."""
    if split not in {"train", "val"}:
        return samples
    if not samples:
        return []

    val_ratio = resolve_val_split_ratio(cfg)
    if val_ratio <= 0.0:
        return samples if split == "train" else []

    stratify_labels: List[int] = []
    for sample in samples:
        labels = [int(x) for x in sample.get("labels", [])]
        if not labels:
            stratify_labels.append(0)
            continue
        level = stratify_level
        if level < 0:
            level = len(labels) + level
        if level < 0 or level >= len(labels):
            level = len(labels) - 1
        stratify_labels.append(int(labels[level]))

    seed = resolve_split_seed(cfg)
    train_idx, val_idx = stratified_train_val_indices(stratify_labels, val_ratio=val_ratio, seed=seed)
    chosen = train_idx if split == "train" else val_idx
    return [samples[i] for i in chosen]


class BaseHierDataset(Dataset, ABC):
    """Abstract base dataset for hierarchical image classification tasks."""

    def __init__(self, cfg: Any, split: str, transform=None):
        """Load samples, remap labels, and prepare taxonomy metadata."""
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.root = Path(str(cfg.dataset.root))
        self.depth = int(cfg.dataset.get("hierarchy_depth", 3))

        self.samples = self.load_samples()
        if not self.samples:
            raise RuntimeError(f"No samples found for split={split} in dataset={cfg.dataset.name}")
        self._validate_image_references(self.samples)

        self.samples, self.level_id_maps = self._remap_labels_to_contiguous(self.samples, self.depth)
        self.num_classes_per_level = self._compute_num_classes_per_level(self.samples, self.depth)
        taxonomy = self.load_taxonomy()
        if taxonomy is not None:
            taxonomy = self._remap_taxonomy_ids(taxonomy)
        self.taxonomy = taxonomy or self._infer_taxonomy()

    @abstractmethod
    def load_samples(self) -> List[Dict[str, Any]]:
        """Load raw samples for the active split.

        Expected sample keys are ``image`` and ``labels`` (and optional ``meta``).
        """
        raise NotImplementedError

    def load_taxonomy(self) -> Optional[Dict[str, Any]]:
        """Return ``None`` so taxonomy is always inferred from sample labels."""
        return None

    def default_levels(self) -> Optional[List[str]]:
        """Return default hierarchy level names when config does not provide them."""
        return None

    def _taxonomy_levels(self) -> Optional[List[str]]:
        configured = list(self.cfg.dataset.get("levels", []))
        if configured:
            return configured
        return self.default_levels()

    def _infer_taxonomy(self) -> Optional[Dict[str, Any]]:
        """Infer taxonomy from loaded sample labels when no file is available."""
        parent_of = infer_parent_of_from_samples(self.samples, self.depth)
        if parent_of is None:
            warnings.warn("Could not infer a consistent taxonomy from labels. Taxonomy metrics will be skipped.", RuntimeWarning)
            return None

        levels = self._taxonomy_levels()
        return taxonomy_from_parent_of(parent_of, levels)

    def _remap_taxonomy_ids(self, taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """Remap taxonomy IDs to contiguous IDs used by this dataset instance."""
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
        """Remap level labels to contiguous ranges and return remapping tables."""
        per_level_values = [sorted({int(s["labels"][l]) for s in samples}) for l in range(depth)]
        maps = [{old: idx for idx, old in enumerate(values)} for values in per_level_values]

        remapped = []
        for s in samples:
            labels = [maps[l][int(s["labels"][l])] for l in range(depth)]
            remapped.append({"image": s.get("image"), "labels": labels, "meta": s.get("meta", {})})

        return remapped, maps

    @staticmethod
    def _compute_num_classes_per_level(samples: List[Dict[str, Any]], depth: int) -> List[int]:
        """Compute class counts per hierarchy level from remapped labels."""
        classes = [set() for _ in range(depth)]
        for s in samples:
            for level, value in enumerate(s["labels"]):
                classes[level].add(int(value))
        return [max(values) + 1 if values else 1 for values in classes]

    def _validate_image_references(self, samples: List[Dict[str, Any]]) -> None:
        """Fail fast when any path-based sample image is missing."""
        missing: List[Tuple[int, str]] = []
        for idx, sample in enumerate(samples):
            image_ref = sample.get("image")
            if isinstance(image_ref, int):
                # Index-backed datasets (e.g., CIFAR) resolve images outside filesystem paths.
                continue
            if image_ref is None:
                missing.append((idx, "<none>"))
            else:
                image_path = Path(image_ref)
                if not image_path.exists():
                    missing.append((idx, str(image_path)))
            if len(missing) >= 8:
                break

        if missing:
            details = "; ".join([f"idx={idx} image={path}" for idx, path in missing])
            raise FileNotFoundError(
                "Missing image files detected in dataset samples. "
                "All image paths must exist. "
                f"First missing entries: {details}"
            )

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
        if image_path is None:
            raise FileNotFoundError("Sample has no image path.")
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")
        return Image.open(path).convert("RGB")

    def _read_json_samples(self, ann_path: Path) -> List[Dict[str, Any]]:
        """Parse JSON annotations into normalized sample dictionaries."""
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
        """Resolve the annotation file path configured for the current split."""
        ann_cfg = self.cfg.dataset.get("annotations", {})
        split_file = ann_cfg.get(self.split)
        if not split_file:
            return None
        path = self.root / str(split_file)
        if path.exists():
            return path
        return None
