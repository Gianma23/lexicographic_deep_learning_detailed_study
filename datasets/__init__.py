from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .aircraft import AircraftDataset
from .breeds import BreedsDataset
from .cub import CUBDataset
from .inat import INatDataset


_DATASET_REGISTRY = {
    "breeds": BreedsDataset,
    "cub": CUBDataset,
    "aircraft": AircraftDataset,
    "inat": INatDataset,
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
    dataset_name = str(cfg.dataset.name).lower()
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of {list(_DATASET_REGISTRY)}")

    dataset_cls = _DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(cfg=cfg, split=split, transform=build_transforms(cfg, split))

    batch_size = int(cfg.dataloader.batch_size)
    workers = int(cfg.dataloader.get("num_workers", 4))
    pin_memory = bool(cfg.dataloader.get("pin_memory", True))

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
