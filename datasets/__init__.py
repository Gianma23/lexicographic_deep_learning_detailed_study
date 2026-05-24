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
    "cub-200-2011": CUBDataset,
    "fgvc-aircraft": AircraftDataset,
    "inat21-mini": INatDataset,
    "cifar-100": CIFAR100Dataset,
}


def _interp_mode_from_name(name: str) -> InterpolationMode:
    if not isinstance(name, str):
        raise ValueError(
            f"dataset interpolation mode must be a string. Got {type(name)!r}."
        )
    if name == "nearest":
        return InterpolationMode.NEAREST
    if name == "bilinear":
        return InterpolationMode.BILINEAR
    if name == "bicubic":
        return InterpolationMode.BICUBIC
    raise ValueError(
        f"Unsupported interpolation mode '{name}'. Expected one of: nearest, bilinear, bicubic."
    )


class _StandardScalerNormalize:
    """Per-image z-score normalization: (x - mean) / std."""

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}")
        tensor = tensor.float()
        mean = tensor.mean()
        std = tensor.std(unbiased=False).clamp_min(self.eps)
        return (tensor - mean) / std

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(eps={self.eps})"


class _MinMaxNormalize:
    """Per-image min-max normalization to [0, 1]."""

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}")
        tensor = tensor.float()
        min_val = tensor.amin()
        max_val = tensor.amax()
        denom = (max_val - min_val).clamp_min(self.eps)
        return (tensor - min_val) / denom

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(eps={self.eps})"


def _normalization_ops(mode: str, mean, std, eps: float):
    if mode == "torchvision":
        return [transforms.Normalize(mean=mean, std=std)]
    if mode == "standardscaler":
        return [_StandardScalerNormalize(eps=eps)]
    if mode == "minmax":
        return [_MinMaxNormalize(eps=eps)]
    return []


def _replace_timm_normalization(transform, replacement_ops):
    if hasattr(transform, "transforms"):
        ops = [op for op in transform.transforms if not isinstance(op, transforms.Normalize)]
        ops.extend(replacement_ops)
        transform.transforms = ops
        return transform
    if replacement_ops:
        return transforms.Compose([transform, *replacement_ops])
    return transform


class _CropBottomPixels:
    """Crop a fixed number of pixels from the image bottom edge."""

    def __init__(self, pixels: int):
        self.pixels = int(pixels)

    def __call__(self, image):
        if self.pixels <= 0:
            return image
        if not hasattr(image, "size"):
            return image
        width, height = image.size
        if self.pixels >= int(height):
            raise ValueError(
                "dataset.transforms.manual.crop_bottom_pixels is too large for the input image height."
            )
        return image.crop((0, 0, int(width), int(height) - self.pixels))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(pixels={self.pixels})"


def build_transforms(cfg: Any, split: str):
    dataset_cfg = cfg.dataset
    transforms_cfg = dataset_cfg["transforms"]

    image_size = int(dataset_cfg["image_size"])
    mean = list(dataset_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = list(dataset_cfg.get("std", [0.229, 0.224, 0.225]))
    normalization_mode = transforms_cfg["normalization"]
    if not isinstance(normalization_mode, str):
        raise ValueError("dataset.transforms.normalization must be a string.")
    if normalization_mode not in {"torchvision", "standardscaler", "minmax", "none"}:
        raise ValueError(
            f"Unsupported dataset.transforms.normalization='{normalization_mode}'. "
            "Expected one of: torchvision, standardscaler, minmax, none."
        )
    normalization_eps = float(transforms_cfg.get("normalization_eps", 1e-6))
    if normalization_eps <= 0:
        raise ValueError("dataset.transforms.normalization_eps must be > 0.")
    normalization_ops = _normalization_ops(normalization_mode, mean=mean, std=std, eps=normalization_eps)

    # Simple fixed-resize-only pipeline used for HT-CapsNet parity runs.
    if bool(transforms_cfg.get("fixed_resize_only", False)):
        fixed_resize_interpolation = transforms_cfg.get("fixed_resize_interpolation", "bilinear")
        if not isinstance(fixed_resize_interpolation, str):
            raise ValueError("dataset.transforms.fixed_resize_interpolation must be a string.")
        resize_interp = _interp_mode_from_name(fixed_resize_interpolation)
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=resize_interp),
                transforms.ToTensor(),
                *normalization_ops,
            ]
        )

    # -------------------------------------------------------------------------
    # Config groups:
    # - transforms.use_timm controls whether timm's train pipeline is used.
    # - transforms.timm.* controls timm-specific knobs (train only).
    # - transforms.manual.* controls train pipeline when use_timm=false.
    # - transforms.eval.* controls val/test resize+crop behavior.
    # -------------------------------------------------------------------------
    use_timm = bool(transforms_cfg["use_timm"])
    manual_cfg = transforms_cfg.get("manual", {})
    manual_crop_bottom_pixels = int(manual_cfg.get("crop_bottom_pixels", 0))
    if manual_crop_bottom_pixels < 0:
        raise ValueError("dataset.transforms.manual.crop_bottom_pixels must be >= 0.")
    crop_bottom_op = _CropBottomPixels(manual_crop_bottom_pixels)

    if use_timm:
        timm_cfg = transforms_cfg["timm"]
        random_erase_cfg = timm_cfg["random_erase"]
        color_jitter = float(timm_cfg["color_jitter"])
        auto_augment = timm_cfg["auto_augment"]
        if not isinstance(auto_augment, str):
            raise ValueError("dataset.transforms.timm.auto_augment must be a string.")
        train_interpolation = timm_cfg["train_interpolation"]
        if not isinstance(train_interpolation, str):
            raise ValueError("dataset.transforms.timm.train_interpolation must be a string.")
        reprob = float(random_erase_cfg["prob"])
        remode = random_erase_cfg["mode"]
        if not isinstance(remode, str):
            raise ValueError("dataset.transforms.timm.random_erase.mode must be a string.")
        recount = int(random_erase_cfg["count"])
        timm_small_image_crop_padding = int(timm_cfg.get("small_image_random_crop_padding", 4))
    else:
        manual_crop_mode = manual_cfg["crop_mode"]
        if not isinstance(manual_crop_mode, str):
            raise ValueError("dataset.transforms.manual.crop_mode must be a string.")
        manual_crop_padding = int(manual_cfg.get("random_crop_padding", 4))
        manual_crop_padding_mode = manual_cfg.get("random_crop_padding_mode", "constant")
        if not isinstance(manual_crop_padding_mode, str):
            raise ValueError("dataset.transforms.manual.random_crop_padding_mode must be a string.")
        valid_padding_modes = {"constant", "edge", "reflect", "symmetric"}
        if manual_crop_padding_mode not in valid_padding_modes:
            raise ValueError(
                f"Unsupported dataset.transforms.manual.random_crop_padding_mode='{manual_crop_padding_mode}'. "
                "Expected one of: constant, edge, reflect, symmetric."
            )
        manual_interpolation = manual_cfg.get("interpolation", "bicubic")
        if not isinstance(manual_interpolation, str):
            raise ValueError("dataset.transforms.manual.interpolation must be a string.")
        manual_resize_before_crop = bool(manual_cfg.get("resize_before_crop", False))
        manual_resize_before_crop_size = int(manual_cfg.get("resize_before_crop_size", image_size))
        manual_resize_before_crop_interpolation = manual_cfg.get(
            "resize_before_crop_interpolation",
            manual_interpolation,
        )
        if not isinstance(manual_resize_before_crop_interpolation, str):
            raise ValueError("dataset.transforms.manual.resize_before_crop_interpolation must be a string.")
        manual_rrc_scale_raw = manual_cfg.get("random_resized_crop_scale", [0.08, 1.0])
        manual_rrc_ratio_raw = manual_cfg.get("random_resized_crop_ratio", [3.0 / 4.0, 4.0 / 3.0])
        if len(manual_rrc_scale_raw) != 2:
            raise ValueError("dataset.transforms.manual.random_resized_crop_scale must have length 2.")
        if len(manual_rrc_ratio_raw) != 2:
            raise ValueError("dataset.transforms.manual.random_resized_crop_ratio must have length 2.")
        manual_rrc_scale = (float(manual_rrc_scale_raw[0]), float(manual_rrc_scale_raw[1]))
        manual_rrc_ratio = (float(manual_rrc_ratio_raw[0]), float(manual_rrc_ratio_raw[1]))
        manual_hflip_prob = float(manual_cfg.get("random_horizontal_flip_prob", 0.5))
        if manual_hflip_prob < 0.0 or manual_hflip_prob > 1.0:
            raise ValueError("dataset.transforms.manual.random_horizontal_flip_prob must be in [0, 1].")

    eval_cfg = transforms_cfg["eval"]
    eval_resize_mode = eval_cfg["resize_mode"]
    if not isinstance(eval_resize_mode, str):
        raise ValueError("dataset.transforms.eval.resize_mode must be a string.")
    eval_crop_ratio = float(eval_cfg["crop_ratio"])
    if eval_crop_ratio <= 0:
        raise ValueError("dataset.transforms.eval.crop_ratio must be > 0.")
    eval_resize_size = eval_cfg.get("resize_size", None)
    if eval_resize_size is not None:
        eval_resize_size = int(eval_resize_size)
        if eval_resize_size <= 0:
            raise ValueError("dataset.transforms.eval.resize_size must be > 0 when provided.")
    eval_resize_square = bool(eval_cfg.get("resize_square", False))
    eval_interpolation = eval_cfg["interpolation"]
    if not isinstance(eval_interpolation, str):
        raise ValueError("dataset.transforms.eval.interpolation must be a string.")

    # TRAIN transforms
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
                # Match small-image behavior with explicit random crop.
                transform.transforms[0] = transforms.RandomCrop(image_size, padding=timm_small_image_crop_padding)
            if normalization_mode != "torchvision":
                transform = _replace_timm_normalization(transform, normalization_ops)
            if manual_crop_bottom_pixels > 0:
                return transforms.Compose([crop_bottom_op, transform])
            return transform

        valid_crop_modes = {"auto", "random_crop", "random_resized_crop", "none"}
        if manual_crop_mode not in valid_crop_modes:
            raise ValueError(
                f"Unsupported dataset.transforms.manual.crop_mode='{manual_crop_mode}'. "
                "Expected one of: auto, random_crop, random_resized_crop, none."
            )
        if manual_crop_mode == "auto":
            manual_crop_mode = "random_crop" if image_size <= 32 else "random_resized_crop"

        train_ops = []
        if manual_crop_bottom_pixels > 0:
            train_ops.append(crop_bottom_op)
        if manual_resize_before_crop:
            train_ops.append(
                transforms.Resize(
                    (manual_resize_before_crop_size, manual_resize_before_crop_size),
                    interpolation=_interp_mode_from_name(manual_resize_before_crop_interpolation),
                )
            )
        if manual_crop_mode == "random_crop":
            train_ops.append(
                transforms.RandomCrop(
                    image_size,
                    padding=manual_crop_padding,
                    padding_mode=manual_crop_padding_mode,
                )
            )
        elif manual_crop_mode == "random_resized_crop":
            train_ops.append(
                transforms.RandomResizedCrop(
                    image_size,
                    scale=manual_rrc_scale,
                    ratio=manual_rrc_ratio,
                    interpolation=_interp_mode_from_name(manual_interpolation),
                )
            )

        if manual_hflip_prob > 0.0:
            train_ops.append(transforms.RandomHorizontalFlip(p=manual_hflip_prob))
        train_ops.extend(
            [
                transforms.ToTensor(),
                *normalization_ops,
            ]
        )
        return transforms.Compose(train_ops)

    # EVAL transforms (val/test)
    valid_resize_modes = {"auto", "resize_center_crop", "resize", "none"}
    if eval_resize_mode not in valid_resize_modes:
        raise ValueError(
            f"Unsupported dataset.transforms.eval.resize_mode='{eval_resize_mode}'. "
            "Expected one of: auto, resize_center_crop, resize, none."
        )
    if eval_resize_mode == "auto":
        eval_resize_mode = "none" if image_size <= 32 else "resize_center_crop"

    eval_ops = []
    if manual_crop_bottom_pixels > 0:
        eval_ops.append(crop_bottom_op)
    if eval_resize_mode == "resize_center_crop":
        resize_size = int(eval_resize_size) if eval_resize_size is not None else int(image_size / eval_crop_ratio)
        resize_arg = (resize_size, resize_size) if eval_resize_square else resize_size
        eval_ops.extend(
            [
                transforms.Resize(resize_arg, interpolation=_interp_mode_from_name(eval_interpolation)),
                transforms.CenterCrop(image_size),
            ]
        )
    elif eval_resize_mode == "resize":
        resize_size = int(eval_resize_size) if eval_resize_size is not None else int(image_size / eval_crop_ratio)
        eval_ops.append(
            transforms.Resize(
                (resize_size, resize_size), interpolation=_interp_mode_from_name(eval_interpolation)
            )
        )
    eval_ops.extend(
        [
            transforms.ToTensor(),
            *normalization_ops,
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
    dataset_name = cfg.dataset.name
    if not isinstance(dataset_name, str):
        raise ValueError("dataset.name must be a string.")
    if dataset_name not in _DATASET_REGISTRY:
        supported = sorted(_DATASET_REGISTRY.keys())
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of {supported}")

    dataset_cls = _DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(cfg=cfg, split=split, transform=build_transforms(cfg, split))

    batch_size = int(cfg.dataloader.batch_size)
    workers = int(cfg.dataloader.get("num_workers", 4))
    pin_memory = bool(cfg.dataloader.get("pin_memory", True))
    windows_spawn_safe = bool(cfg.dataloader.get("windows_spawn_safe", True))
    drop_last_train = bool(cfg.dataloader.get("drop_last_train", True))
    drop_last_eval = bool(cfg.dataloader.get("drop_last_eval", False))

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
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
        drop_last=bool(drop_last_train if split == "train" else drop_last_eval),
        generator=generator,
        worker_init_fn=_seed_worker,
    )

    return loader, dataset.num_classes_per_level, dataset.taxonomy
