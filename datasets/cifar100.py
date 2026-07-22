import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torchvision.datasets import CIFAR100

from .base import BaseHierDataset
from .splitting import (
    resolve_split_seed,
    resolve_val_split_ratio,
    stratified_train_val_indices,
)
from .types import DatasetLabelSpace

# B-CNN manually groups CIFAR-100's 20 official coarse classes into 8
# coarse-1 classes. This edge is absent from the dataset and therefore must
# remain explicit. Source:
# https://github.com/zhuxinqimac/B-CNN/blob/main/CIFAR_100_keras_vgg16_hierarchy_dynamic.py
B_CNN_COARSE_TO_SUPER = [
    0, 0, 1, 2, 1, 2, 2, 3, 4, 5,
    5, 4, 4, 3, 6, 4, 4, 1, 7, 7,
]


def _pickle_field(payload: Dict[Any, Any], name: str):
    if name in payload:
        return payload[name]
    encoded = name.encode("utf-8")
    if encoded in payload:
        return payload[encoded]
    return None


def load_official_cifar100_fine_to_coarse(root: Path) -> List[int]:
    """Derive fine-to-coarse IDs from the official CIFAR-100 Python archive."""
    root = Path(root)
    candidates = [
        root / "cifar-100-python" / "train",
        root / "train",
    ]
    train_file = next((path for path in candidates if path.is_file()), None)
    if train_file is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Could not locate the official CIFAR-100 Python training file needed "
            f"for fine/coarse labels. Searched: {searched}"
        )

    with train_file.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed official CIFAR-100 payload in {train_file}: expected a dictionary.")

    fine_labels = _pickle_field(payload, "fine_labels")
    coarse_labels = _pickle_field(payload, "coarse_labels")
    if not isinstance(fine_labels, list) or not isinstance(coarse_labels, list):
        raise ValueError(
            f"Official CIFAR-100 payload {train_file} must contain fine_labels and coarse_labels lists."
        )
    if len(fine_labels) != len(coarse_labels) or not fine_labels:
        raise ValueError(
            f"Official CIFAR-100 label arrays in {train_file} are empty or misaligned: "
            f"fine={len(fine_labels)}, coarse={len(coarse_labels)}."
        )

    fine_to_coarse: Dict[int, int] = {}
    for fine_raw, coarse_raw in zip(fine_labels, coarse_labels):
        fine = int(fine_raw)
        coarse = int(coarse_raw)
        previous = fine_to_coarse.get(fine)
        if previous is not None and previous != coarse:
            raise ValueError(
                f"Inconsistent official CIFAR-100 parent for fine class {fine}: "
                f"coarse class {previous} versus {coarse}."
            )
        fine_to_coarse[fine] = coarse

    expected_fine = set(range(100))
    if set(fine_to_coarse) != expected_fine:
        missing = sorted(expected_fine - set(fine_to_coarse))
        extra = sorted(set(fine_to_coarse) - expected_fine)
        raise ValueError(
            "Official CIFAR-100 training labels do not define exactly 100 fine classes. "
            f"Missing: {missing[:10]}, extra: {extra[:10]}."
        )
    coarse_counts = Counter(fine_to_coarse.values())
    expected_coarse = set(range(20))
    if set(coarse_counts) != expected_coarse:
        missing = sorted(expected_coarse - set(coarse_counts))
        extra = sorted(set(coarse_counts) - expected_coarse)
        raise ValueError(
            "Official CIFAR-100 training labels do not define exactly 20 coarse classes. "
            f"Missing: {missing}, extra: {extra}."
        )
    invalid_fanout = {coarse: count for coarse, count in coarse_counts.items() if count != 5}
    if invalid_fanout:
        raise ValueError(
            "B-CNN expects the official CIFAR-100 hierarchy with exactly five fine "
            f"classes per coarse class; found {invalid_fanout}."
        )
    return [fine_to_coarse[fine] for fine in range(100)]


class CIFAR100Dataset(BaseHierDataset):
    """CIFAR-100 adapter with hierarchy super->coarse->fine."""

    def default_levels(self) -> List[str]:
        """Default hierarchy names used when config does not provide levels."""
        return ["coarse1", "coarse2", "fine"]

    def __init__(
        self,
        cfg: Any,
        split: str,
        transform=None,
        label_space: Optional[DatasetLabelSpace] = None,
    ):
        """Initialize CIFAR storage used for index-based image retrieval."""
        self._cifar_images = None
        self._cifar_targets: List[int] = []
        super().__init__(
            cfg=cfg,
            split=split,
            transform=transform,
            label_space=label_space,
        )

    def _label_path(self, fine: int, coarse: int) -> List[int]:
        """Build hierarchical labels for the configured depth (2 or 3 levels)."""
        if self.depth == 2:
            return [coarse, fine]
        if self.depth == 3:
            super_cls = int(B_CNN_COARSE_TO_SUPER[coarse])
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

        if len(B_CNN_COARSE_TO_SUPER) != 20:
            raise RuntimeError("Invalid B-CNN CIFAR-100 coarse->super mapping. Expected 20 entries.")
        fine_to_coarse = load_official_cifar100_fine_to_coarse(self.root)

        val_ratio = resolve_val_split_ratio(self.cfg)
        val_source = self.cfg.dataset.get("val_source", "train_split")
        if not isinstance(val_source, str):
            raise ValueError("dataset.val_source must be a string.")
        if val_source not in {"train_split", "test"}:
            raise ValueError("dataset.val_source must be one of ['train_split', 'test'].")
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
            coarse = int(fine_to_coarse[fine])
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
