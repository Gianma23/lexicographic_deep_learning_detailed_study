import os
import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .aircraft import AircraftDataset
from .breeds import BreedsDataset
from .cifar100 import CIFAR100Dataset
from .cub import CUBDataset
from .inat import INatDataset


_DATASET_REGISTRY = {
    "breeds": BreedsDataset,
    "cub": CUBDataset,
    "aircraft": AircraftDataset,
    "inat": INatDataset,
    "cifar100": CIFAR100Dataset,
}

_DATASET_ALIASES = {
    "cifar-100": "cifar100",
    "cub-200-2011": "cub",
    "fgvc-aircraft": "aircraft",
    "inat21-mini": "inat",
    "inat21_mini": "inat",
}


def build_transforms(cfg: Any, split: str):
    image_size = int(cfg.dataset.get("image_size", 224))
    mean = cfg.dataset.get("mean", [0.485, 0.456, 0.406])
    std = cfg.dataset.get("std", [0.229, 0.224, 0.225])

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    metas = [item[2] for item in batch]
    return images, labels, metas


def build_dataloader(cfg: Any, split: str):
    dataset_name_raw = str(cfg.dataset.name).lower()
    dataset_name = _DATASET_ALIASES.get(dataset_name_raw, dataset_name_raw)

    if dataset_name not in _DATASET_REGISTRY:
        supported = sorted(set(_DATASET_REGISTRY.keys()) | set(_DATASET_ALIASES.keys()))
        raise ValueError(f"Unsupported dataset '{dataset_name_raw}'. Expected one of {supported}")

    dataset_cls = _DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(cfg=cfg, split=split, transform=build_transforms(cfg, split))

    batch_size = int(cfg.dataloader.batch_size)
    workers = int(cfg.dataloader.get("num_workers", 4))
    pin_memory = bool(cfg.dataloader.get("pin_memory", True))
    windows_spawn_safe = bool(cfg.dataloader.get("windows_spawn_safe", True))

    # On Windows, CUDA + worker spawning can hit WinError 1455 when each
    # subprocess imports torch CUDA DLLs (e.g., cublas64_12.dll).
    if os.name == "nt" and windows_spawn_safe and torch.cuda.is_available() and workers > 0:
        warnings.warn(
            "Windows + CUDA detected: forcing dataloader.num_workers=0 to avoid WinError 1455. "
            "Set dataloader.windows_spawn_safe=false to opt out.",
            RuntimeWarning,
        )
        workers = 0

    if os.name == "nt" and windows_spawn_safe and torch.cuda.is_available() and workers == 0 and pin_memory:
        warnings.warn(
            "Windows + CUDA detected: forcing dataloader.pin_memory=false to reduce host memory pressure.",
            RuntimeWarning,
        )
        pin_memory = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
        drop_last=split == "train",
    )

    return loader, dataset.num_classes_per_level, dataset.taxonomy
