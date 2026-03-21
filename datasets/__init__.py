import os
import random
import warnings
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:
    from timm.data import create_transform as timm_create_transform
except Exception:  # pragma: no cover
    timm_create_transform = None

from .aircraft import AircraftDataset
from .cifar100 import CIFAR100Dataset
from .cub import CUBDataset
from .inat import INatDataset


_DATASET_REGISTRY = {
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


def _get_aug_value(dataset_cfg: Any, key: str, default: Any, aliases=None):
    aliases = aliases or []
    transforms_cfg = dataset_cfg.get("transforms", {}) or {}

    if key in transforms_cfg:
        return transforms_cfg.get(key)
    for alias in aliases:
        if alias in transforms_cfg:
            return transforms_cfg.get(alias)

    if key in dataset_cfg:
        return dataset_cfg.get(key)
    for alias in aliases:
        if alias in dataset_cfg:
            return dataset_cfg.get(alias)
    return default


def _interp_mode_from_name(name: str) -> InterpolationMode:
    value = str(name).strip().lower()
    if value == "nearest":
        return InterpolationMode.NEAREST
    if value == "bilinear":
        return InterpolationMode.BILINEAR
    if value == "bicubic":
        return InterpolationMode.BICUBIC
    return InterpolationMode.BICUBIC


def build_transforms(cfg: Any, split: str):
    image_size = int(cfg.dataset.get("image_size", 224))
    mean = list(cfg.dataset.get("mean", [0.485, 0.456, 0.406]))
    std = list(cfg.dataset.get("std", [0.229, 0.224, 0.225]))

    use_timm = bool(_get_aug_value(cfg.dataset, "use_timm", True, aliases=["timm"]))
    color_jitter = float(_get_aug_value(cfg.dataset, "color_jitter", 0.3))
    auto_augment = str(_get_aug_value(cfg.dataset, "aa", "rand-m9-mstd0.5-inc1", aliases=["auto_augment"]))
    train_interpolation = str(
        _get_aug_value(cfg.dataset, "train_interpolation", "bicubic", aliases=["interpolation"])
    )
    reprob = float(_get_aug_value(cfg.dataset, "reprob", 0.25, aliases=["re_prob"]))
    remode = str(_get_aug_value(cfg.dataset, "remode", "pixel", aliases=["re_mode"]))
    recount = int(_get_aug_value(cfg.dataset, "recount", 1, aliases=["re_count"]))
    eval_crop_ratio = float(
        _get_aug_value(cfg.dataset, "eval_crop_ratio", 0.875, aliases=["crop_pct", "val_crop_ratio"])
    )

    if split == "train":
        if use_timm:
            if timm_create_transform is None:
                raise ImportError(
                    "dataset.transforms.use_timm=true requires timm.data.create_transform, but timm is unavailable."
                )
            transform = timm_create_transform(
                input_size=image_size,
                is_training=True,
                color_jitter=color_jitter,
                auto_augment=auto_augment,
                interpolation=train_interpolation,
                re_prob=reprob,
                re_mode=remode,
                re_count=recount,
                mean=tuple(mean),
                std=tuple(std),
            )
            if image_size <= 32 and hasattr(transform, "transforms") and transform.transforms:
                # Match H-CAST small-image behavior used for CIFAR-style inputs.
                transform.transforms[0] = transforms.RandomCrop(image_size, padding=4)
            return transform

        train_ops = []
        if image_size <= 32:
            train_ops.append(transforms.RandomCrop(image_size, padding=4))
        else:
            train_ops.append(
                transforms.RandomResizedCrop(
                    image_size,
                    interpolation=_interp_mode_from_name(train_interpolation),
                )
            )
        train_ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return transforms.Compose(train_ops)

    eval_ops = []
    if image_size > 32:
        resize_size = int(image_size / max(eval_crop_ratio, 1e-8))
        eval_ops.extend(
            [
                transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
            ]
        )
    eval_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(eval_ops)


def _collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    metas = [item[2] for item in batch]
    return images, labels, metas


def _loader_seed(cfg: Any, split: str) -> int:
    base_seed = int(cfg.train.get("seed", 42))
    split_offsets = {"train": 0, "val": 1, "test": 2}
    return int(base_seed + split_offsets.get(split, 0))


def _seed_worker(worker_id: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


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

    loader_seed = _loader_seed(cfg, split)
    generator = torch.Generator()
    generator.manual_seed(loader_seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        collate_fn=_collate_fn,
        drop_last=split == "train",
        generator=generator,
        worker_init_fn=_seed_worker,
    )

    return loader, dataset.num_classes_per_level, dataset.taxonomy
