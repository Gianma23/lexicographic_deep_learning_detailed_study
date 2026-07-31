"""Dataset registry and deterministic DataLoader construction."""

import os
import random
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .aircraft import AircraftDataset
from .cifar100 import CIFAR100Dataset
from .cub import CUBDataset
from .transforms import build_transforms
from .types import DatasetLabelSpace, DatasetMetadata


_DATASET_REGISTRY = {
    "cub-200-2011": CUBDataset,
    "fgvc-aircraft": AircraftDataset,
    "cifar-100": CIFAR100Dataset,
}


def _collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    metadata = [item[2] for item in batch]
    return images, labels, metadata


def _seed_worker(_worker_id: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _build_loader(cfg: Any, split: str, label_space: Optional[DatasetLabelSpace]):
    dataset_name = cfg.dataset.name
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Expected one of {sorted(_DATASET_REGISTRY)}."
        )
    dataset = _DATASET_REGISTRY[dataset_name](
        cfg=cfg,
        split=split,
        transform=build_transforms(cfg, split),
        label_space=label_space,
    )

    workers = int(cfg.dataloader.get("num_workers", 4))
    if (
        os.name == "nt"
        and bool(cfg.dataloader.get("windows_spawn_safe", True))
        and torch.cuda.is_available()
        and workers > 0
    ):
        warnings.warn(
            "Windows + CUDA detected: forcing dataloader.num_workers=0 to "
            "avoid worker CUDA DLL exhaustion.",
            RuntimeWarning,
        )
        workers = 0

    generator = torch.Generator().manual_seed(
        int(cfg.train.get("seed", 42)) + {"train": 0, "val": 1, "test": 2}[split]
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=split == "train",
        num_workers=workers,
        pin_memory=bool(cfg.dataloader.get("pin_memory", True)),
        collate_fn=_collate_fn,
        drop_last=bool(
            cfg.dataloader.get(
                "drop_last_train" if split == "train" else "drop_last_eval",
                split == "train",
            )
        ),
        generator=generator,
        worker_init_fn=_seed_worker,
    )


def build_dataloader(
    cfg: Any,
    split: str,
    label_space: Optional[DatasetLabelSpace] = None,
) -> Tuple[DataLoader, list, Dict]:
    """Compatibility API for building one split."""
    loader = _build_loader(cfg, split, label_space)
    dataset = loader.dataset
    return loader, dataset.num_classes_per_level, dataset.taxonomy


def build_dataloaders(cfg: Any):
    """Build every split with one canonical hierarchy created by training."""
    train_loader = _build_loader(cfg, "train", label_space=None)
    train_dataset = train_loader.dataset
    label_space = getattr(train_dataset, "label_space", None)
    if not isinstance(label_space, DatasetLabelSpace):
        raise RuntimeError("Training dataset did not expose a valid DatasetLabelSpace.")

    loaders = {
        "train": train_loader,
        "val": _build_loader(cfg, "val", label_space=label_space),
        "test": _build_loader(cfg, "test", label_space=label_space),
    }
    for split, loader in loaders.items():
        dataset = loader.dataset
        if list(dataset.num_classes_per_level) != list(label_space.num_classes_per_level):
            raise RuntimeError(f"{split} class counts diverged from the canonical label space.")
        if dataset.taxonomy != label_space.taxonomy:
            raise RuntimeError(f"{split} taxonomy diverged from the canonical label space.")
    return loaders, DatasetMetadata(label_space=label_space)
