from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torchvision.datasets import CIFAR100

from .base import (
    BaseHierDataset,
    resolve_split_seed,
    resolve_val_split_ratio,
    stratified_train_val_indices,
)

# Canonical CIFAR-100 fine->coarse mapping aligned with torchvision fine-label order.
_FINE_TO_COARSE = [
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

# HT-CapsNet CIFAR-100 coarse(20)->super(8) mapping.
_COARSE_TO_SUPER = [
    0, 0, 1, 2, 1, 2, 2, 3, 4, 5,
    5, 4, 4, 3, 6, 4, 4, 1, 7, 7,
]


class CIFAR100Dataset(BaseHierDataset):
    """CIFAR-100 adapter with hierarchy super->coarse->fine."""

    def default_levels(self) -> List[str]:
        """Default hierarchy names used when config does not provide levels."""
        return ["coarse1", "coarse2", "fine"]

    def __init__(self, cfg: Any, split: str, transform=None):
        """Initialize CIFAR storage used for index-based image retrieval."""
        self._cifar_images = None
        self._cifar_targets: List[int] = []
        super().__init__(cfg=cfg, split=split, transform=transform)

    def _label_path(self, fine: int, coarse: int) -> List[int]:
        """Build hierarchical labels for the configured depth (2 or 3 levels)."""
        if self.depth == 2:
            return [coarse, fine]
        if self.depth == 3:
            super_cls = int(_COARSE_TO_SUPER[coarse])
            return [super_cls, coarse, fine]
        raise ValueError(
            f"CIFAR-100 supports hierarchy_depth in {{2, 3}}, got {self.depth}. "
            "Use 3 for super->coarse->fine."
        )

    def load_samples(self) -> List[Dict[str, Any]]:
        """Load split samples from torchvision CIFAR-100 with optional val split."""
        ann_file = self._annotation_file_for_split()
        if ann_file is not None:
            return self._read_json_samples(ann_file)

        if len(_FINE_TO_COARSE) != 100:
            raise RuntimeError("Invalid CIFAR-100 fine->coarse mapping. Expected 100 entries.")
        if len(_COARSE_TO_SUPER) != 20:
            raise RuntimeError("Invalid CIFAR-100 coarse->super mapping. Expected 20 entries.")

        val_ratio = resolve_val_split_ratio(self.cfg)
        val_source = str(self.cfg.dataset.get("val_source", "train_split")).strip().lower()
        val_uses_test = self.split == "val" and val_ratio <= 0.0 and val_source == "test"
        split_is_train_pool = self.split in {"train", "val"} and not val_uses_test
        download = bool(self.cfg.dataset.get("download", False))

        try:
            dataset = CIFAR100(root=str(self.root), train=split_is_train_pool, download=download)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not load CIFAR-100 from root={self.root}. "
                f"Set dataset.download=true or provide an existing dataset at this path. Original error: {exc}"
            ) from exc

        self._cifar_images = dataset.data
        self._cifar_targets = [int(x) for x in dataset.targets]

        if split_is_train_pool:
            train_idx, val_idx = stratified_train_val_indices(
                self._cifar_targets,
                val_ratio=val_ratio,
                seed=resolve_split_seed(self.cfg),
            )
            chosen = train_idx if self.split == "train" else val_idx
        else:
            chosen = list(range(len(self._cifar_targets)))

        samples: List[Dict[str, Any]] = []
        for idx in chosen:
            fine = int(self._cifar_targets[idx])
            coarse = int(_FINE_TO_COARSE[fine])
            samples.append(
                {
                    "image": int(idx),
                    "labels": self._label_path(fine=fine, coarse=coarse),
                    "meta": {
                        "source": "cifar100_torchvision",
                        "index": int(idx),
                        "split": self.split,
                    },
                }
            )

        return samples

    def _load_image(self, image_ref: Optional[Path]):
        """Resolve int-backed CIFAR image references or defer to path-based loader."""
        if isinstance(image_ref, int):
            if self._cifar_images is None:
                raise RuntimeError(
                    "CIFAR image storage is unavailable while trying to resolve an index-backed sample."
                )
            return Image.fromarray(self._cifar_images[image_ref]).convert("RGB")
        return super()._load_image(image_ref)
