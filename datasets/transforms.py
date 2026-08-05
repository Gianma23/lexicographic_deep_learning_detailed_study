"""Image transformation construction for train, validation, and test splits."""

from typing import Any

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:
    from timm.data import create_transform as timm_create_transform
except Exception:  # pragma: no cover
    timm_create_transform = None


def _interpolation(name: str) -> InterpolationMode:
    modes = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }
    if name not in modes:
        raise ValueError(
            f"Unsupported interpolation mode '{name}'. Expected one of {sorted(modes)}."
        )
    return modes[name]


class _StandardScalerNormalize:
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float()
        return (tensor - tensor.mean()) / tensor.std(unbiased=False).clamp_min(self.eps)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(eps={self.eps})"


class _MinMaxNormalize:
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float()
        minimum, maximum = tensor.amin(), tensor.amax()
        return (tensor - minimum) / (maximum - minimum).clamp_min(self.eps)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(eps={self.eps})"


class _CropBottomPixels:
    def __init__(self, pixels: int):
        self.pixels = int(pixels)

    def __call__(self, image):
        if self.pixels <= 0 or not hasattr(image, "size"):
            return image
        width, height = image.size
        if self.pixels >= int(height):
            raise ValueError(
                "dataset.transforms.manual.crop_bottom_pixels is too large "
                "for the input image height."
            )
        return image.crop((0, 0, int(width), int(height) - self.pixels))

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(pixels={self.pixels})"


def _normalization_ops(mode: str, mean, std, eps: float):
    if mode == "torchvision":
        return [transforms.Normalize(mean=mean, std=std)]
    if mode == "standardscaler":
        return [_StandardScalerNormalize(eps=eps)]
    if mode == "minmax":
        return [_MinMaxNormalize(eps=eps)]
    if mode == "none":
        return []
    raise ValueError(
        f"Unsupported dataset.transforms.normalization='{mode}'. "
        "Expected torchvision, standardscaler, minmax, or none."
    )


# Scopes over which a whole-tensor normalization statistic is reduced.
# `image` keeps the statistic per sample. `batch` reduces over the assembled
# batch instead, matching the upstream HT-CapsNet pipeline, which maps
# `Normalize_StandardScaler` over an already-batched `tf.data` dataset.
_BATCH_SCOPED_MODES = {"standardscaler", "minmax"}


def normalization_scope(transform_cfg: Any) -> str:
    scope = transform_cfg.get("normalization_scope", "image")
    if not isinstance(scope, str):
        raise ValueError("dataset.transforms.normalization_scope must be a string.")
    if scope not in {"image", "batch"}:
        raise ValueError(
            f"Unsupported dataset.transforms.normalization_scope='{scope}'. "
            "Expected image or batch."
        )
    mode = transform_cfg["normalization"]
    if scope == "batch" and mode not in _BATCH_SCOPED_MODES:
        raise ValueError(
            "dataset.transforms.normalization_scope='batch' requires "
            f"normalization to be one of {sorted(_BATCH_SCOPED_MODES)}, got '{mode}'."
        )
    return scope


def build_batch_normalizer(cfg: Any):
    """Return the batch-scoped normalization op, or None when scope is per-image."""
    transform_cfg = cfg.dataset["transforms"]
    if normalization_scope(transform_cfg) != "batch":
        return None
    operations = _normalization_ops(
        transform_cfg["normalization"],
        list(cfg.dataset.get("mean", [0.485, 0.456, 0.406])),
        list(cfg.dataset.get("std", [0.229, 0.224, 0.225])),
        float(transform_cfg.get("normalization_eps", 1e-6)),
    )
    if len(operations) != 1:
        raise ValueError(
            "dataset.transforms.normalization_scope='batch' expects exactly one "
            "normalization operation."
        )
    return operations[0]


def _replace_timm_normalization(transform, replacement_ops):
    if hasattr(transform, "transforms"):
        transform.transforms = [
            operation
            for operation in transform.transforms
            if not isinstance(operation, transforms.Normalize)
        ] + list(replacement_ops)
        return transform
    return (
        transforms.Compose([transform, *replacement_ops])
        if replacement_ops
        else transform
    )


def _build_fixed_resize(image_size: int, cfg: Any, crop_bottom, normalization_ops):
    interpolation = _interpolation(cfg.get("fixed_resize_interpolation", "bilinear"))
    antialias = cfg.get("fixed_resize_antialias", True)
    if not isinstance(antialias, bool):
        raise ValueError("dataset.transforms.fixed_resize_antialias must be a boolean.")

    operations = [crop_bottom] if crop_bottom.pixels > 0 else []
    if antialias:
        # torchvision resamples the PIL image, whose bilinear filter support
        # scales with the downsample ratio.
        operations.extend(
            [
                transforms.Resize((image_size, image_size), interpolation=interpolation),
                transforms.ToTensor(),
            ]
        )
    else:
        # `tf.image.resize` defaults to `antialias=False` and runs after the float
        # conversion, so the tensor is converted first and then resampled with a
        # fixed 2x2 kernel regardless of the downsample ratio. The operation order
        # differs from the antialiased branch because the two pipelines differ.
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=interpolation,
                    antialias=False,
                ),
            ]
        )
    operations.extend(normalization_ops)
    return transforms.Compose(operations)


def _build_timm_train(
    image_size: int,
    mean,
    std,
    cfg: Any,
    normalization_mode: str,
    normalization_ops,
    crop_bottom,
):
    if timm_create_transform is None:
        raise ImportError(
            "dataset.transforms.use_timm=true requires "
            "timm.data.create_transform, but timm is unavailable."
        )
    erase = cfg["random_erase"]
    transform = timm_create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=float(cfg["color_jitter"]),
        auto_augment=cfg["auto_augment"],
        interpolation=cfg["train_interpolation"],
        re_prob=float(erase["prob"]),
        re_mode=erase["mode"],
        re_count=int(erase["count"]),
        mean=tuple(mean),
        std=tuple(std),
    )
    if image_size <= 32 and hasattr(transform, "transforms") and transform.transforms:
        transform.transforms[0] = transforms.RandomCrop(
            image_size,
            padding=int(cfg.get("small_image_random_crop_padding", 4)),
        )
    if normalization_mode != "torchvision":
        transform = _replace_timm_normalization(transform, normalization_ops)
    return transforms.Compose([crop_bottom, transform]) if crop_bottom.pixels > 0 else transform


def _build_manual_train(image_size: int, cfg: Any, crop_bottom, normalization_ops):
    crop_mode = cfg["crop_mode"]
    if crop_mode == "auto":
        crop_mode = "random_crop" if image_size <= 32 else "random_resized_crop"
    if crop_mode not in {"random_crop", "random_resized_crop", "none"}:
        raise ValueError(
            f"Unsupported dataset.transforms.manual.crop_mode='{crop_mode}'."
        )

    operations = [crop_bottom] if crop_bottom.pixels > 0 else []
    interpolation_name = cfg.get("interpolation", "bicubic")
    if bool(cfg.get("resize_before_crop", False)):
        size = int(cfg.get("resize_before_crop_size", image_size))
        operations.append(
            transforms.Resize(
                (size, size),
                interpolation=_interpolation(
                    cfg.get("resize_before_crop_interpolation", interpolation_name)
                ),
            )
        )

    if crop_mode == "random_crop":
        operations.append(
            transforms.RandomCrop(
                image_size,
                padding=int(cfg.get("random_crop_padding", 4)),
                padding_mode=cfg.get("random_crop_padding_mode", "constant"),
            )
        )
    elif crop_mode == "random_resized_crop":
        scale = tuple(float(value) for value in cfg.get("random_resized_crop_scale", [0.08, 1.0]))
        ratio = tuple(
            float(value)
            for value in cfg.get(
                "random_resized_crop_ratio",
                [3.0 / 4.0, 4.0 / 3.0],
            )
        )
        operations.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=scale,
                ratio=ratio,
                interpolation=_interpolation(interpolation_name),
            )
        )

    flip_probability = float(cfg.get("random_horizontal_flip_prob", 0.5))
    if flip_probability > 0.0:
        operations.append(transforms.RandomHorizontalFlip(p=flip_probability))
    operations.extend([transforms.ToTensor(), *normalization_ops])
    return transforms.Compose(operations)


def _build_eval(image_size: int, cfg: Any, crop_bottom, normalization_ops):
    resize_mode = cfg["resize_mode"]
    if resize_mode == "auto":
        resize_mode = "none" if image_size <= 32 else "resize_center_crop"
    if resize_mode not in {"resize_center_crop", "resize", "none"}:
        raise ValueError(
            f"Unsupported dataset.transforms.eval.resize_mode='{resize_mode}'."
        )

    operations = [crop_bottom] if crop_bottom.pixels > 0 else []
    if resize_mode != "none":
        crop_ratio = float(cfg["crop_ratio"])
        configured_size = cfg.get("resize_size")
        resize_size = (
            int(configured_size)
            if configured_size is not None
            else int(image_size / crop_ratio)
        )
        interpolation = _interpolation(cfg["interpolation"])
        if resize_mode == "resize_center_crop":
            resize_arg = (
                (resize_size, resize_size)
                if bool(cfg.get("resize_square", False))
                else resize_size
            )
            operations.extend(
                [
                    transforms.Resize(resize_arg, interpolation=interpolation),
                    transforms.CenterCrop(image_size),
                ]
            )
        else:
            operations.append(
                transforms.Resize(
                    (resize_size, resize_size),
                    interpolation=interpolation,
                )
            )
    operations.extend([transforms.ToTensor(), *normalization_ops])
    return transforms.Compose(operations)


def build_transforms(cfg: Any, split: str):
    """Build the configured transformation pipeline for one dataset split."""
    dataset_cfg = cfg.dataset
    transform_cfg = dataset_cfg["transforms"]
    image_size = int(dataset_cfg["image_size"])
    mean = list(dataset_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = list(dataset_cfg.get("std", [0.229, 0.224, 0.225]))
    normalization_mode = transform_cfg["normalization"]
    normalization_eps = float(transform_cfg.get("normalization_eps", 1e-6))
    if normalization_eps <= 0.0:
        raise ValueError("dataset.transforms.normalization_eps must be > 0.")
    normalization_ops = _normalization_ops(
        normalization_mode,
        mean,
        std,
        normalization_eps,
    )
    if normalization_scope(transform_cfg) == "batch":
        # The statistic is reduced over the assembled batch in the collate
        # instead, so the per-sample pipeline must not normalize twice.
        normalization_ops = []

    # `crop_bottom_pixels` removes a fixed strip before any resize, so it applies
    # to the fixed-resize path as well as the manual/timm ones.
    manual_cfg = transform_cfg.get("manual", {})
    crop_bottom_pixels = int(manual_cfg.get("crop_bottom_pixels", 0))
    if crop_bottom_pixels < 0:
        raise ValueError(
            "dataset.transforms.manual.crop_bottom_pixels must be >= 0."
        )
    crop_bottom = _CropBottomPixels(crop_bottom_pixels)

    if bool(transform_cfg.get("fixed_resize_only", False)):
        return _build_fixed_resize(
            image_size,
            transform_cfg,
            crop_bottom,
            normalization_ops,
        )

    flip_probability = float(manual_cfg.get("random_horizontal_flip_prob", 0.5))
    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError(
            "dataset.transforms.manual.random_horizontal_flip_prob must be in [0, 1]."
        )
    eval_cfg = transform_cfg["eval"]
    if float(eval_cfg["crop_ratio"]) <= 0.0:
        raise ValueError("dataset.transforms.eval.crop_ratio must be > 0.")
    if eval_cfg.get("resize_size") is not None and int(eval_cfg["resize_size"]) <= 0:
        raise ValueError(
            "dataset.transforms.eval.resize_size must be > 0 when provided."
        )
    if split == "train":
        if bool(transform_cfg["use_timm"]):
            return _build_timm_train(
                image_size,
                mean,
                std,
                transform_cfg["timm"],
                normalization_mode,
                normalization_ops,
                crop_bottom,
            )
        return _build_manual_train(
            image_size,
            manual_cfg,
            crop_bottom,
            normalization_ops,
        )
    return _build_eval(
        image_size,
        eval_cfg,
        crop_bottom,
        normalization_ops,
    )
