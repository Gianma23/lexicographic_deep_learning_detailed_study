"""Common PyTorch dataset lifecycle and normalized annotation I/O."""

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .hierarchy import (
    apply_label_space,
    build_training_label_space,
    infer_parent_of_from_samples,
    normalize_samples,
    taxonomy_from_parent_of,
)
from .splitting import (  # compatibility exports used by iNat and older callers
    resolve_split_seed,
    resolve_val_split_ratio,
    split_train_val_samples,
    stratified_train_val_indices,
)
from .types import DatasetLabelSpace


class BaseHierDataset(Dataset, ABC):
    """Load one split and apply the shared canonical hierarchy."""

    def __init__(
        self,
        cfg: Any,
        split: str,
        transform=None,
        label_space: Optional[DatasetLabelSpace] = None,
    ):
        self.cfg, self.split, self.transform = cfg, split, transform
        self.root = Path(str(cfg.dataset.root))
        self.depth = int(cfg.dataset.get("hierarchy_depth", 3))
        self.dataset_name = str(cfg.dataset.name)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported dataset split '{split}'.")
        if self.depth < 2:
            raise ValueError(f"dataset.hierarchy_depth must be >= 2, got {self.depth}.")

        raw_samples = self.load_samples()
        if not raw_samples:
            raise RuntimeError(f"No samples found for split={split} in dataset={self.dataset_name}")
        raw_samples = normalize_samples(
            raw_samples,
            depth=self.depth,
            dataset_name=self.dataset_name,
            split=split,
        )
        self._validate_image_references(raw_samples)
        raw_taxonomy = self.load_taxonomy()

        if label_space is None:
            self.samples, self.label_space = build_training_label_space(
                raw_samples,
                raw_taxonomy,
                self._taxonomy_levels() or [],
                depth=self.depth,
                dataset_name=self.dataset_name,
                split=split,
            )
        else:
            self.samples = apply_label_space(
                raw_samples,
                raw_taxonomy,
                label_space,
                depth=self.depth,
                dataset_name=self.dataset_name,
                split=split,
            )
            self.label_space = label_space

        self.level_id_maps = [
            dict(mapping) for mapping in self.label_space.raw_to_contiguous
        ]
        self.num_classes_per_level = list(self.label_space.num_classes_per_level)
        self.taxonomy = deepcopy(self.label_space.taxonomy)

    @abstractmethod
    def load_samples(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def load_taxonomy(self) -> Optional[Dict[str, Any]]:
        return None

    def default_levels(self) -> Optional[List[str]]:
        return None

    def _taxonomy_levels(self) -> Optional[List[str]]:
        return list(self.cfg.dataset.get("levels", [])) or self.default_levels()

    def _validate_image_references(self, samples: Sequence[Mapping[str, Any]]) -> None:
        missing: List[Tuple[int, str]] = []
        for index, sample in enumerate(samples):
            image = sample.get("image")
            if not isinstance(image, int) and (image is None or not Path(image).exists()):
                missing.append((index, str(image or "<none>")))
            if len(missing) == 8:
                break
        if missing:
            preview = "; ".join(f"idx={index} image={path}" for index, path in missing)
            raise FileNotFoundError(f"Missing image files detected. First entries: {preview}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._load_image(sample["image"])
        if self.transform is not None:
            image = self.transform(image)
        return (
            image,
            torch.tensor(sample["labels"], dtype=torch.long),
            sample.get("meta", {}),
        )

    def _load_image(self, image_path: Optional[Path]):
        if image_path is None or not Path(image_path).exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        return Image.open(image_path).convert("RGB")

    def _read_json_samples(self, path: Path) -> List[Dict[str, Any]]:
        """Read the repository's normalized JSON manifest."""
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("samples")
        if not isinstance(data, list):
            raise TypeError(f"Annotation file {path} must contain a `samples` list.")

        samples = []
        for row_index, row in enumerate(data):
            if not isinstance(row, Mapping):
                raise TypeError(f"Malformed annotation {path} row {row_index}: expected a mapping.")
            labels = row.get("labels") or row.get("levels")
            if not isinstance(labels, (list, tuple)) or len(labels) != self.depth:
                actual = len(labels) if isinstance(labels, (list, tuple)) else "non-list"
                raise ValueError(
                    f"Malformed annotation {path} row {row_index}: "
                    f"expected {self.depth} labels, got {actual}."
                )
            image = row.get("image") or row.get("path")
            if image is None or not str(image).strip():
                raise ValueError(f"Malformed annotation {path} row {row_index}: missing image/path.")
            image = Path(str(image)).expanduser()
            if not image.is_absolute():
                image = self.root / image
            metadata = dict(row)
            metadata.update({"annotation_file": str(path), "annotation_row": row_index})
            samples.append({"image": image, "labels": labels, "meta": metadata})
        return samples

    def _annotation_file_for_split(self) -> Optional[Path]:
        configured = self.cfg.dataset.get("annotations", {}).get(self.split)
        if not configured:
            return None
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = self.root / path
        if not path.is_file():
            raise FileNotFoundError(
                f"Configured annotation for dataset={self.dataset_name} "
                f"split={self.split} does not exist: {path}"
            )
        return path


__all__ = [
    "BaseHierDataset",
    "DatasetLabelSpace",
    "infer_parent_of_from_samples",
    "resolve_split_seed",
    "resolve_val_split_ratio",
    "split_train_val_samples",
    "stratified_train_val_indices",
    "taxonomy_from_parent_of",
]
