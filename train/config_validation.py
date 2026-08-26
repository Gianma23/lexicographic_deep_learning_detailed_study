"""Fail-fast schema and compatibility validation for experiment configurations."""

from math import isfinite
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set

from omegaconf import OmegaConf

from .lexicographic.types import DEFAULT_GRADIENT_BLOCKS, GRADIENT_BLOCK_NAMES


SUPPORTED_MODELS = {"hcast", "lhdnn", "ht_capsnet", "hrn", "hiercos"}
SUPPORTED_DATASETS = {"cifar-100", "cub-200-2011", "fgvc-aircraft"}

_MODEL_KEYS: Dict[str, Set[str]] = {
    "hcast": {
        "name",
        "variant",
        "pretrained",
        "drop",
        "drop_path",
        "finetune",
        "segments",
        "loss",
    },
    "lhdnn": {"name", "projection", "adaptive_pool_size"},
    "ht_capsnet": {
        "name",
        "caps_dim",
        "secondary_dims",
        "routing_iters",
        "num_blocks",
        "initial_filters",
        "filter_increment",
        "backbone_net",
        "backbone_net_weights",
        "backbone_variant",
        "backbone_preprocessing",
        "backbone_bn_momentum",
        "backbone_drop_path",
        "taxonomy_temperature",
        "mask_threshold_high",
        "mask_threshold_low",
        "mask_temperature",
        "mask_center",
        "attn_heads",
        "attn_key_dim",
        "attn_dropout",
        "attn_postprocess",
        "attn_initializer",
        "loss",
    },
    "hrn": {
        "name",
        "backbone",
        "pretrained",
        "wide_depth",
        "wide_widen_factor",
        "wide_drop_rate",
        "branch_hidden_dim",
        "embedding_dim",
        "dropout",
        "trunk_lr_scale",
        "loss",
    },
    "hiercos": {
        "name",
        "loss",
        "weight_mode",
        "weight_beta",
        "variant",
        "transform_mode",
        "pool",
        "pretrained",
        "alpha",
        "backbone_lr_scale",
        "transform_lr_scale",
        "fixed_frame_mode",
        "fixed_frame_per_level",
        "projection",
        "wide_depth",
        "wide_widen_factor",
        "wide_drop_rate",
    },
}

_ALLOWED_CHILDREN: Dict[str, Set[str]] = {
    "": {
        "model",
        "dataset",
        "dataloader",
        "train",
        "optim",
        "scheduler",
        "runtime",
        "hcc",
    },
    "dataset": {
        "name",
        "root",
        "hierarchy_depth",
        "image_size",
        "in_channels",
        "mean",
        "std",
        "levels",
        "download",
        "split_policy",
        "val_split_ratio",
        "val_source",
        "split_seed",
        "annotations",
        "transforms",
    },
    "dataset.annotations": {"train", "val", "test"},
    "dataset.transforms": {
        "use_timm",
        "normalization",
        "normalization_eps",
        "normalization_scope",
        "fixed_resize_only",
        "fixed_resize_interpolation",
        "fixed_resize_intermediate_size",
        "fixed_resize_antialias",
        "timm",
        "manual",
        "eval",
        "mixup",
        "cutmix",
        "cutmix_minmax",
        "mixup_prob",
        "mixup_switch_prob",
        "mixup_mode",
        "mixup_pairing",
    },
    "dataset.transforms.timm": {
        "color_jitter",
        "auto_augment",
        "train_interpolation",
        "random_erase",
        "small_image_random_crop_padding",
    },
    "dataset.transforms.timm.random_erase": {"prob", "mode", "count"},
    "dataset.transforms.manual": {
        "crop_mode",
        "crop_bottom_pixels",
        "resize_before_crop",
        "resize_before_crop_size",
        "resize_before_crop_interpolation",
        "random_crop_padding",
        "random_crop_padding_mode",
        "random_resized_crop_scale",
        "random_resized_crop_ratio",
        "interpolation",
        "random_horizontal_flip_prob",
    },
    "dataset.transforms.eval": {
        "resize_mode",
        "resize_size",
        "resize_square",
        "crop_ratio",
        "interpolation",
    },
    "dataloader": {
        "batch_size",
        "num_workers",
        "pin_memory",
        "windows_spawn_safe",
        "drop_last_train",
        "drop_last_eval",
    },
    "train": {
        "epochs",
        "seed",
        "device",
        "amp",
        "progress_bar",
        "resume",
        "output_dir",
        "smoothing",
        "stop_epoch",
        "scale_lr",
        "scale_lr_reference_batch_size",
        "gradient_blocks",
        "lexicographic",
        "subspace_supervision",
    },
    "train.lexicographic": {
        "enabled",
        "projection_mode",
        "eps",
        "log_metrics",
    },
    "train.subspace_supervision": {
        "enabled",
        "tau",
        "eps",
    },
    "optim": {
        "name",
        "lr",
        "lr_base",
        "weight_decay",
        "momentum",
        "nesterov",
        "opt_eps",
        "opt_betas",
    },
    "scheduler": {
        "name",
        "base_lr",
        "warmup_lr",
        "min_lr",
        "noise",
        "noise_pct",
        "noise_std",
        "noise_seed",
        "decay_epochs",
        "warmup_epochs",
        "cooldown_epochs",
        "patience_epochs",
        "decay_rate",
        "start_epoch",
    },
    "runtime": {"deterministic", "protocol"},
    "hcc": {
        "enabled",
        "eps",
    },
    "model.segments": {
        "mode",
        "patch_size",
        "num_superpixels",
        "num_levels",
        "prior",
        "histogram_bins",
        "double_step",
        "num_iterations",
    },
    "model.projection": {
        "enabled",
        "advantage_enabled",
        "feature_dim",
        "eps",
    },
    "model.loss": {
        "globalkl",
        "gk_weight",
        "level_weighting",
        "margin_m_pos",
        "margin_m_neg",
        "lambda_downweight",
        "weight_mode",
        "dynamic_weight",
    },
    "model.loss.level_weighting": {"mode", "gamma", "eps"},
}

_REQUIRED_ROOT_SECTIONS = {
    "model",
    "dataset",
    "dataloader",
    "train",
    "optim",
    "scheduler",
    "runtime",
}


def _plain_config(cfg: Any) -> Dict[str, Any]:
    if OmegaConf.is_config(cfg):
        payload = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, Mapping):
        payload = dict(cfg)
    else:
        raise TypeError(f"Configuration must be an OmegaConf or mapping, got {type(cfg)!r}.")
    if not isinstance(payload, dict):
        raise TypeError("Resolved configuration root must be a mapping.")
    return payload


def _validate_unknown_keys(
    value: Any,
    path: str,
    model_name: Optional[str],
) -> None:
    if not isinstance(value, Mapping):
        return
    if path == "model":
        allowed = _MODEL_KEYS.get(model_name or "", set().union(*_MODEL_KEYS.values()))
    else:
        allowed = _ALLOWED_CHILDREN.get(path)
    if allowed is None:
        if value:
            raise ValueError(f"Configuration section `{path}` does not accept nested keys.")
        return

    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        location = path or "<root>"
        raise ValueError(f"Unknown configuration key(s) under `{location}`: {unknown}.")
    for key, child in value.items():
        child_path = f"{path}.{key}" if path else str(key)
        _validate_unknown_keys(child, child_path, model_name)


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = payload.get(key)
    if not isinstance(section, Mapping):
        raise ValueError(f"Configuration section `{key}` must be a mapping.")
    return section


def _finite_float(value: Any, path: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{path}` must be numeric, got {value!r}.") from exc
    if not isfinite(parsed):
        raise ValueError(f"`{path}` must be finite, got {value!r}.")
    return parsed


def _positive_int(value: Any, path: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"`{path}` must be an integer >= {minimum}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{path}` must be an integer >= {minimum}.") from exc
    if parsed < minimum or parsed != float(value):
        raise ValueError(f"`{path}` must be an integer >= {minimum}, got {value!r}.")
    return parsed


def _require_enum(value: Any, path: str, allowed: Set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"`{path}` must be one of {sorted(allowed)}, got {value!r}.")
    return value


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"`{path}` must be a YAML boolean, got {value!r}.")
    return value


def _validate_optional_bool(section: Mapping[str, Any], key: str, path: str) -> None:
    if key in section:
        _require_bool(section[key], path)


def _validate_common_sections(payload: Mapping[str, Any]) -> None:
    dataset = _require_mapping(payload, "dataset")
    dataloader = _require_mapping(payload, "dataloader")
    train = _require_mapping(payload, "train")
    optim = _require_mapping(payload, "optim")
    scheduler = _require_mapping(payload, "scheduler")
    runtime = _require_mapping(payload, "runtime")
    protocol = runtime.get("protocol", "corrected_unified_v1")
    legacy_resume = "protocol" not in runtime and bool(str(train.get("resume", "")).strip())

    dataset_name = _require_enum(dataset.get("name"), "dataset.name", SUPPORTED_DATASETS)
    depth = _positive_int(dataset.get("hierarchy_depth"), "dataset.hierarchy_depth", minimum=2)
    if depth != 3:
        raise ValueError("`corrected_unified_v1` dataset configs require exactly three hierarchy levels.")
    _positive_int(dataset.get("image_size"), "dataset.image_size")
    if not isinstance(dataset.get("root"), (str, Path)):
        raise ValueError("`dataset.root` must be a path string.")
    levels = dataset.get("levels")
    if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)) or len(levels) != depth:
        raise ValueError(
            f"`dataset.levels` must contain exactly hierarchy_depth={depth} names."
        )
    if len({str(level) for level in levels}) != depth:
        raise ValueError("`dataset.levels` names must be unique.")

    mean = dataset.get("mean")
    std = dataset.get("std")
    if not isinstance(mean, Sequence) or isinstance(mean, (str, bytes)) or len(mean) != 3:
        raise ValueError("`dataset.mean` must contain three numeric values.")
    if not isinstance(std, Sequence) or isinstance(std, (str, bytes)) or len(std) != 3:
        raise ValueError("`dataset.std` must contain three positive numeric values.")
    for index, value in enumerate(mean):
        _finite_float(value, f"dataset.mean[{index}]")
    for index, value in enumerate(std):
        if _finite_float(value, f"dataset.std[{index}]") <= 0.0:
            raise ValueError(f"`dataset.std[{index}]` must be > 0.")

    val_ratio = _finite_float(dataset.get("val_split_ratio", 0.1), "dataset.val_split_ratio")
    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError("`dataset.val_split_ratio` must be in [0, 1).")
    if dataset.get("split_seed") is not None:
        _positive_int(dataset["split_seed"], "dataset.split_seed", minimum=0)
    if dataset.get("in_channels") is not None:
        _positive_int(dataset["in_channels"], "dataset.in_channels")
    _validate_optional_bool(dataset, "download", "dataset.download")
    split_policy = dataset.get("split_policy", "official_val_test")
    _require_enum(split_policy, "dataset.split_policy", {"official_val_test", "explicit"})
    annotations = dataset.get("annotations")
    if annotations is not None and not isinstance(annotations, Mapping):
        raise ValueError("`dataset.annotations` must be a train/val/test mapping.")
    if split_policy == "explicit":
        if not isinstance(annotations, Mapping):
            raise ValueError("Explicit split policy requires a dataset.annotations mapping.")
        missing = [split for split in ("train", "val", "test") if not annotations.get(split)]
        if missing:
            raise ValueError(f"Explicit split policy is missing annotation paths for {missing}.")
    if isinstance(annotations, Mapping) and annotations:
        missing = [split for split in ("train", "val", "test") if not annotations.get(split)]
        if missing:
            raise ValueError(
                "Configured annotations must define a complete train/val/test set; "
                f"missing paths for {missing}."
            )
    if dataset_name == "fgvc-aircraft" and dataset.get("annotations"):
        raise ValueError(
            "FGVC-Aircraft uses the official parallel annotations; remove dataset.annotations."
        )
    if (
        dataset_name == "fgvc-aircraft"
        and "val_split_ratio" in dataset
        and not legacy_resume
    ):
        raise ValueError(
            "FGVC-Aircraft uses its official validation split; remove dataset.val_split_ratio."
        )

    transforms = dataset.get("transforms")
    if not isinstance(transforms, Mapping):
        raise ValueError("`dataset.transforms` must be a mapping.")
    _require_bool(transforms.get("use_timm"), "dataset.transforms.use_timm")
    _validate_optional_bool(
        transforms,
        "fixed_resize_only",
        "dataset.transforms.fixed_resize_only",
    )
    _validate_optional_bool(
        transforms,
        "fixed_resize_antialias",
        "dataset.transforms.fixed_resize_antialias",
    )
    if transforms.get("fixed_resize_intermediate_size") is not None:
        _positive_int(
            transforms.get("fixed_resize_intermediate_size"),
            "dataset.transforms.fixed_resize_intermediate_size",
        )
    manual_transforms = transforms.get("manual", {})
    if not isinstance(manual_transforms, Mapping):
        raise ValueError("`dataset.transforms.manual` must be a mapping.")
    _validate_optional_bool(
        manual_transforms,
        "resize_before_crop",
        "dataset.transforms.manual.resize_before_crop",
    )
    eval_transforms = transforms.get("eval", {})
    if not isinstance(eval_transforms, Mapping):
        raise ValueError("`dataset.transforms.eval` must be a mapping.")
    _validate_optional_bool(
        eval_transforms,
        "resize_square",
        "dataset.transforms.eval.resize_square",
    )
    timm_transforms = transforms.get("timm", {})
    if not isinstance(timm_transforms, Mapping):
        raise ValueError("`dataset.transforms.timm` must be a mapping.")
    _require_enum(
        transforms.get("normalization"),
        "dataset.transforms.normalization",
        {"torchvision", "standardscaler", "minmax", "none"},
    )
    _require_enum(
        transforms.get("normalization_scope", "image"),
        "dataset.transforms.normalization_scope",
        {"image", "batch", "dataset"},
    )
    normalization_scope_value = transforms.get("normalization_scope", "image")
    if normalization_scope_value in {"batch", "dataset"} and transforms.get(
        "normalization"
    ) not in {"standardscaler", "minmax"}:
        raise ValueError(
            f"`dataset.transforms.normalization_scope: {normalization_scope_value}` requires "
            "`dataset.transforms.normalization` to be standardscaler or minmax."
        )
    if normalization_scope_value == "dataset" and dataset.get("name") != "cifar-100":
        raise ValueError(
            "`dataset.transforms.normalization_scope: dataset` is currently supported "
            "only for the array-backed CIFAR-100 adapter."
        )
    for key in ("mixup", "cutmix"):
        if _finite_float(transforms.get(key, 0.0), f"dataset.transforms.{key}") < 0.0:
            raise ValueError(f"`dataset.transforms.{key}` must be >= 0.")
    for key in ("mixup_prob", "mixup_switch_prob"):
        value = _finite_float(transforms.get(key, 0.0), f"dataset.transforms.{key}")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"`dataset.transforms.{key}` must be in [0, 1].")
    _require_enum(
        transforms.get("mixup_mode", "batch"),
        "dataset.transforms.mixup_mode",
        {"batch", "pair", "elem"},
    )
    _require_enum(
        transforms.get("mixup_pairing", "flip"),
        "dataset.transforms.mixup_pairing",
        {"flip", "random"},
    )
    if transforms.get("mixup_pairing", "flip") == "random" and transforms.get("mixup_mode", "batch") != "elem":
        raise ValueError("dataset.transforms.mixup_pairing=random requires mixup_mode=elem.")

    _positive_int(dataloader.get("batch_size"), "dataloader.batch_size")
    _positive_int(dataloader.get("num_workers", 0), "dataloader.num_workers", minimum=0)
    for key in (
        "pin_memory",
        "windows_spawn_safe",
        "drop_last_train",
        "drop_last_eval",
    ):
        _validate_optional_bool(dataloader, key, f"dataloader.{key}")
    if protocol is not None:
        _require_enum(
            protocol,
            "runtime.protocol",
            {"corrected_unified_v1"},
        )

    if bool(dataloader.get("drop_last_eval", False)) and not legacy_resume:
        raise ValueError(
            "`dataloader.drop_last_eval` must be false under corrected_unified_v1."
        )

    epochs = _positive_int(train.get("epochs"), "train.epochs")
    _positive_int(train.get("seed"), "train.seed", minimum=0)
    for key in ("amp", "progress_bar", "scale_lr"):
        _validate_optional_bool(train, key, f"train.{key}")
    if not isinstance(train.get("device"), str) or not str(train.get("device")).strip():
        raise ValueError("`train.device` must be a non-empty device string.")
    stop_epoch = _positive_int(train.get("stop_epoch", epochs), "train.stop_epoch")
    if stop_epoch > epochs:
        raise ValueError("`train.stop_epoch` must be <= train.epochs.")
    smoothing = _finite_float(train.get("smoothing", 0.0), "train.smoothing")
    if smoothing < 0.0 or smoothing > 1.0:
        raise ValueError("`train.smoothing` must be in [0, 1].")
    if not isinstance(train.get("output_dir"), (str, Path)):
        raise ValueError("`train.output_dir` must be a path string.")
    _require_enum(
        optim.get("name"),
        "optim.name",
        {"sgd", "adam", "adamw"},
    )
    _validate_optional_bool(optim, "nesterov", "optim.nesterov")
    if _finite_float(optim.get("lr"), "optim.lr") <= 0.0:
        raise ValueError("`optim.lr` must be > 0.")
    # `train.train._apply_lr_scaling_if_enabled` records the pre-scaling LR in
    # resolved run configs. Accept and validate that provenance field so saved
    # configs remain loadable for checkpoint-only evaluation and analysis.
    if "lr_base" in optim and _finite_float(optim["lr_base"], "optim.lr_base") <= 0.0:
        raise ValueError("`optim.lr_base` must be > 0 when provided.")
    if _finite_float(optim.get("weight_decay", 0.0), "optim.weight_decay") < 0.0:
        raise ValueError("`optim.weight_decay` must be >= 0.")
    _require_enum(
        scheduler.get("name"),
        "scheduler.name",
        {"none", "cosine", "step", "hiercos_cosine", "ht_capsnet_exponential"},
    )
    if scheduler.get("name") == "ht_capsnet_exponential":
        _positive_int(scheduler.get("start_epoch", 10), "scheduler.start_epoch", minimum=0)
        decay_rate = _finite_float(scheduler.get("decay_rate", 0.95), "scheduler.decay_rate")
        if decay_rate <= 0.0 or decay_rate > 1.0:
            raise ValueError("HT-CapsNet scheduler.decay_rate must be in (0, 1].")
    _require_bool(runtime.get("deterministic", True), "runtime.deterministic")


def _validate_model_compatibility(payload: Mapping[str, Any]) -> None:
    model = _require_mapping(payload, "model")
    dataset = _require_mapping(payload, "dataset")
    train = _require_mapping(payload, "train")
    model_name = _require_enum(model.get("name"), "model.name", SUPPORTED_MODELS)
    depth = int(dataset["hierarchy_depth"])
    hiercos_projection_enabled = False
    hcc_enabled = False

    subspace_supervision = train.get("subspace_supervision")
    subspace_supervision_enabled = False
    if subspace_supervision is not None and not isinstance(
        subspace_supervision,
        Mapping,
    ):
        raise ValueError("`train.subspace_supervision` must be a mapping.")
    if isinstance(subspace_supervision, Mapping):
        subspace_supervision_enabled = _require_bool(
            subspace_supervision.get("enabled", False),
            "train.subspace_supervision.enabled",
        )
        if _finite_float(
            subspace_supervision.get("tau", 0.25),
            "train.subspace_supervision.tau",
        ) <= 0.0:
            raise ValueError("`train.subspace_supervision.tau` must be > 0.")
        if _finite_float(
            subspace_supervision.get("eps", 1e-12),
            "train.subspace_supervision.eps",
        ) <= 0.0:
            raise ValueError("`train.subspace_supervision.eps` must be > 0.")

    if model_name in {"hcast", "hrn"} and depth != 3:
        raise ValueError(f"{model_name} requires exactly three hierarchy levels.")
    if model_name == "lhdnn" and depth < 2:
        raise ValueError("LH-DNN requires at least two hierarchy levels.")

    if model_name == "hcast":
        _validate_optional_bool(model, "pretrained", "model.pretrained")
        segments = model.get("segments")
        if not isinstance(segments, Mapping):
            raise ValueError("H-CAST requires `model.segments` to be a mapping.")
        _validate_optional_bool(segments, "double_step", "model.segments.double_step")
        _require_enum(segments.get("mode", "grid"), "model.segments.mode", {"grid", "seeds"})
        _positive_int(segments.get("patch_size", 8), "model.segments.patch_size")
        _positive_int(segments.get("num_superpixels", 196), "model.segments.num_superpixels")
        loss = model.get("loss")
        if not isinstance(loss, Mapping):
            raise ValueError("H-CAST requires `model.loss` to be a mapping.")
        _validate_optional_bool(loss, "globalkl", "model.loss.globalkl")
        level_weighting = loss.get("level_weighting", {})
        if not isinstance(level_weighting, Mapping):
            raise ValueError("`model.loss.level_weighting` must be a mapping.")
        _require_enum(
            level_weighting.get("mode", "static"),
            "model.loss.level_weighting.mode",
            {"static", "dynamic"},
        )

    if model_name == "ht_capsnet":
        secondary_dims = model.get("secondary_dims")
        if not isinstance(secondary_dims, Sequence) or isinstance(secondary_dims, (str, bytes)):
            raise ValueError("HT-CapsNet requires `model.secondary_dims` as a list.")
        if len(secondary_dims) != depth or any(int(value) <= 0 for value in secondary_dims):
            raise ValueError("HT-CapsNet secondary_dims must be positive and match hierarchy depth.")
        _require_enum(
            model.get("backbone_net", "custom"),
            "model.backbone_net",
            {"custom", "efficientnet_b7"},
        )
        backbone_weights = model.get("backbone_net_weights")
        if backbone_weights is not None:
            _require_enum(
                backbone_weights,
                "model.backbone_net_weights",
                {"imagenet", "none"},
            )
        _require_enum(
            model.get("backbone_variant", "tf_efficientnet_b7.aa_in1k"),
            "model.backbone_variant",
            {
                "tf_efficientnet_b7.aa_in1k",
                "tf_efficientnet_b7.ns_jft_in1k",
                "tf_efficientnet_b7",
            },
        )
        backbone_preprocessing = model.get("backbone_preprocessing", "keras")
        _require_enum(
            backbone_preprocessing,
            "model.backbone_preprocessing",
            {"keras", "timm"},
        )
        bn_momentum = _finite_float(
            model.get("backbone_bn_momentum", 0.01),
            "model.backbone_bn_momentum",
        )
        if not 0.0 < bn_momentum <= 1.0:
            raise ValueError("HT-CapsNet backbone_bn_momentum must be in (0, 1].")
        drop_path = _finite_float(
            model.get("backbone_drop_path", 0.2),
            "model.backbone_drop_path",
        )
        if not 0.0 <= drop_path < 1.0:
            raise ValueError("HT-CapsNet backbone_drop_path must be in [0, 1).")
        _require_enum(
            model.get("attn_postprocess", "layernorm"),
            "model.attn_postprocess",
            {"layernorm", "squash"},
        )
        _require_enum(
            model.get("attn_initializer", "keras_glorot"),
            "model.attn_initializer",
            {"keras_glorot", "pytorch_xavier"},
        )
        if _finite_float(model.get("taxonomy_temperature", 0.5), "model.taxonomy_temperature") <= 0:
            raise ValueError("HT-CapsNet taxonomy_temperature must be > 0.")
        mask_low = _finite_float(model.get("mask_threshold_low", 0.1), "model.mask_threshold_low")
        mask_high = _finite_float(model.get("mask_threshold_high", 0.9), "model.mask_threshold_high")
        if not 0.0 <= mask_low < mask_high <= 1.0:
            raise ValueError(
                "HT-CapsNet mask thresholds must satisfy "
                "0 <= mask_threshold_low < mask_threshold_high <= 1."
            )
        dropout = _finite_float(model.get("attn_dropout", 0.0), "model.attn_dropout")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("`model.attn_dropout` must be in [0, 1).")
        _positive_int(model.get("attn_heads", 16), "model.attn_heads")
        _positive_int(model.get("attn_key_dim", 32), "model.attn_key_dim")
        loss = model.get("loss")
        if not isinstance(loss, Mapping):
            raise ValueError("HT-CapsNet requires `model.loss` to be a mapping.")
        _require_enum(
            loss.get("weight_mode", "dynamic"),
            "model.loss.weight_mode",
            {"dynamic", "static", "none"},
        )
        m_pos = _finite_float(loss.get("margin_m_pos", 0.9), "model.loss.margin_m_pos")
        m_neg = _finite_float(loss.get("margin_m_neg", 0.1), "model.loss.margin_m_neg")
        if not 0.0 <= m_neg < m_pos <= 1.0:
            raise ValueError(
                "HT-CapsNet margins must satisfy 0 <= margin_m_neg < margin_m_pos <= 1."
            )
        if not bool(payload["runtime"].get("deterministic", True)):
            raise ValueError("HT-CapsNet requires runtime.deterministic=true.")

    if model_name == "lhdnn":
        projection = model.get("projection")
        if not isinstance(projection, Mapping):
            raise ValueError("LH-DNN requires `model.projection` to be a mapping.")
        if "feature_dim" in projection:
            raise ValueError(
                "`model.projection.feature_dim` is supported only for model.name='hiercos'."
            )
        if "enabled" in projection and not _require_bool(
            projection.get("enabled"),
            "model.projection.enabled",
        ):
            raise ValueError("LH-DNN projection is always enabled.")
        if "advantage_enabled" in projection and not _require_bool(
            projection.get("advantage_enabled"),
            "model.projection.advantage_enabled",
        ):
            raise ValueError("LH-DNN advantage is always enabled.")
        if _finite_float(projection.get("eps", 0.0), "model.projection.eps") <= 0.0:
            raise ValueError("LH-DNN projection epsilon must be > 0.")
        adaptive_pool_size = model.get("adaptive_pool_size")
        if adaptive_pool_size is not None:
            _positive_int(adaptive_pool_size, "model.adaptive_pool_size")

    if model_name == "hrn":
        hrn_backbone = _require_enum(
            model.get("backbone", "resnet50"),
            "model.backbone",
            {"resnet50", "wide_resnet"},
        )
        _validate_optional_bool(model, "pretrained", "model.pretrained")
        if hrn_backbone == "wide_resnet":
            if model.get("pretrained", True):
                raise ValueError(
                    "HRN model.backbone='wide_resnet' requires model.pretrained=false."
                )
            wide_depth = _positive_int(model.get("wide_depth", 28), "model.wide_depth")
            if (wide_depth - 4) % 6 != 0:
                raise ValueError("HRN WideResNet depth must satisfy (wide_depth - 4) % 6 == 0.")
            _positive_int(model.get("wide_widen_factor", 8), "model.wide_widen_factor")
            wide_drop = _finite_float(model.get("wide_drop_rate", 0.0), "model.wide_drop_rate")
            if wide_drop < 0.0 or wide_drop >= 1.0:
                raise ValueError("HRN `model.wide_drop_rate` must be in [0, 1).")
        _require_enum(
            model.get("loss", "native"),
            "model.loss",
            {"native", "level_conditional"},
        )
        dropout = _finite_float(model.get("dropout", 0.0), "model.dropout")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("`model.dropout` must be in [0, 1).")

    if model_name == "hiercos":
        _validate_optional_bool(model, "pretrained", "model.pretrained")
        _require_enum(
            model.get("loss", "kl_reg"),
            "model.loss",
            {"kl_reg", "global_softmax_ce_reg", "level_softmax_ce_reg"},
        )
        weight_mode = _require_enum(
            model.get("weight_mode", "kl_leaf"),
            "model.weight_mode",
            {
                "equal",
                "kl_leaf",
                "kl_coarse",
                "cumulative_branching",
                "marginal_branching",
            },
        )
        if weight_mode == "cumulative_branching" or "weight_beta" in model:
            raw_weight_beta = model.get("weight_beta", 0.5)
            if isinstance(raw_weight_beta, bool):
                raise ValueError("`model.weight_beta` must be a finite number >= 0.")
            weight_beta = _finite_float(raw_weight_beta, "model.weight_beta")
            if weight_beta < 0.0:
                raise ValueError("`model.weight_beta` must be >= 0.")
        variant = _require_enum(
            model.get("variant", "haframe_resnet50"),
            "model.variant",
            {"haframe_resnet50", "haframe_wide_resnet"},
        )
        _require_enum(
            model.get("transform_mode", "full"),
            "model.transform_mode",
            {"full", "bn_linear", "final_only"},
        )
        _require_enum(
            model.get("fixed_frame_mode", "orthonormal_random"),
            "model.fixed_frame_mode",
            {"orthonormal_random", "orthonormal_block_random", "identity"},
        )
        if not isinstance(model.get("fixed_frame_per_level", False), bool):
            raise ValueError("`model.fixed_frame_per_level` must be boolean.")
        projection = model.get("projection")
        if projection is not None:
            if not isinstance(projection, Mapping):
                raise ValueError("Hier-COS `model.projection` must be a mapping.")
            hiercos_projection_enabled = _require_bool(
                projection.get("enabled", False),
                "model.projection.enabled",
            )
            advantage_enabled = _require_bool(
                projection.get("advantage_enabled", False),
                "model.projection.advantage_enabled",
            )
            if advantage_enabled and not hiercos_projection_enabled:
                raise ValueError(
                    "Hier-COS `model.projection.advantage_enabled=true` requires "
                    "`model.projection.enabled=true`."
                )
            feature_dim = projection.get("feature_dim", 0)
            if (
                isinstance(feature_dim, bool)
                or not isinstance(feature_dim, int)
                or feature_dim < 0
            ):
                raise ValueError(
                    "`model.projection.feature_dim` must be a non-negative "
                    "integer; use 0 for sum(num_classes_per_level)."
                )
            if _finite_float(projection.get("eps", 1e-6), "model.projection.eps") <= 0.0:
                raise ValueError("Hier-COS projection epsilon must be > 0.")
        if variant == "haframe_resnet50":
            _require_enum(model.get("pool", "max"), "model.pool", {"max", "average"})
        else:
            wide_depth = _positive_int(model.get("wide_depth", 28), "model.wide_depth")
            if (wide_depth - 4) % 6 != 0:
                raise ValueError("Hier-COS WideResNet depth must satisfy (wide_depth - 4) % 6 == 0.")
            _positive_int(model.get("wide_widen_factor", 8), "model.wide_widen_factor")
            wide_drop = _finite_float(model.get("wide_drop_rate", 0.0), "model.wide_drop_rate")
            if wide_drop < 0.0 or wide_drop >= 1.0:
                raise ValueError("`model.wide_drop_rate` must be in [0, 1).")
    elif model_name in {"hcast", "ht_capsnet"} and not isinstance(model.get("loss"), Mapping):
        raise ValueError(f"{model_name} requires `model.loss` to be a mapping.")

    hcc = payload.get("hcc")
    if hcc is not None:
        if not isinstance(hcc, Mapping):
            raise ValueError("Top-level `hcc` must be a mapping.")
        hcc_supported_models = {"hcast", "hrn", "ht_capsnet", "hiercos"}
        if model_name not in hcc_supported_models:
            raise ValueError(
                f"Top-level `hcc` is supported only for model.name in {sorted(hcc_supported_models)}."
            )
        if depth != 3:
            raise ValueError("HCC requires exactly three hierarchy levels.")
        hcc_enabled = _require_bool(hcc.get("enabled", False), "hcc.enabled")
        if hcc_enabled:
            if _finite_float(hcc.get("eps", 0.0), "hcc.eps") <= 0.0:
                raise ValueError("Enabled HCC requires hcc.eps > 0.")
            if model_name == "hiercos":
                projection = model.get("projection")
                projection_enabled = (
                    _require_bool(projection.get("enabled", False), "model.projection.enabled")
                    if isinstance(projection, Mapping)
                    else False
                )
                if projection_enabled:
                    raise ValueError(
                        "hcc.enabled=true is not supported together with "
                        "model.projection.enabled=true (Hier-COS's own LH-DNN-style "
                        "projection); disable one of the two."
                    )

    if hiercos_projection_enabled:
        effective_loss = model.get("loss", "kl_reg")
        if effective_loss != "level_softmax_ce_reg":
            raise ValueError(
                "Enabled Hier-COS LH-style projection requires `level_softmax_ce_reg` "
                "so each level loss uses its projected branch."
            )
        effective_frame_mode = model.get("fixed_frame_mode", "orthonormal_random")
        effective_frame_per_level = model.get("fixed_frame_per_level", False)
        if (
            effective_frame_mode != "identity"
            and effective_frame_mode != "orthonormal_block_random"
            and not effective_frame_per_level
        ):
            raise ValueError(
                "Enabled Hier-COS LH-style projection requires an identity or "
                "per-level block-diagonal fixed frame so level heads remain independent."
            )

    lex = train.get("lexicographic")
    lex_enabled = False
    gradient_blocks = train.get("gradient_blocks", DEFAULT_GRADIENT_BLOCKS)
    if (
        not isinstance(gradient_blocks, Sequence)
        or isinstance(gradient_blocks, (str, bytes))
        or not gradient_blocks
    ):
        raise ValueError(
            "`train.gradient_blocks` must be a non-empty list containing only "
            f"{list(GRADIENT_BLOCK_NAMES)}."
        )
    invalid_blocks = sorted(
        {
            str(block)
            for block in gradient_blocks
            if not isinstance(block, str) or block not in GRADIENT_BLOCK_NAMES
        }
    )
    if invalid_blocks:
        raise ValueError(
            "`train.gradient_blocks` contains unsupported entries "
            f"{invalid_blocks}; expected only {list(GRADIENT_BLOCK_NAMES)}."
        )
    if len(set(gradient_blocks)) != len(gradient_blocks):
        raise ValueError("`train.gradient_blocks` must not contain duplicates.")

    if lex is not None and not isinstance(lex, Mapping):
        raise ValueError("`train.lexicographic` must be a mapping.")
    if isinstance(lex, Mapping):
        lex_enabled = _require_bool(
            lex.get("enabled", False),
            "train.lexicographic.enabled",
        )
        _validate_optional_bool(
            lex,
            "log_metrics",
            "train.lexicographic.log_metrics",
        )
    if subspace_supervision_enabled:
        transforms = dataset.get("transforms", {})
        if _finite_float(train.get("smoothing", 0.0), "train.smoothing") > 0.0:
            raise ValueError(
                "Direct subspace supervision does not support soft targets or label "
                "smoothing; set `train.smoothing: 0.0`."
            )
        if _finite_float(transforms.get("mixup", 0.0), "dataset.transforms.mixup") > 0.0:
            raise ValueError(
                "Direct subspace supervision does not support mixup; set "
                "`dataset.transforms.mixup: 0.0`."
            )
        if _finite_float(transforms.get("cutmix", 0.0), "dataset.transforms.cutmix") > 0.0:
            raise ValueError(
                "Direct subspace supervision does not support cutmix; set "
                "`dataset.transforms.cutmix: 0.0`."
            )
        if transforms.get("cutmix_minmax") is not None:
            raise ValueError(
                "Direct subspace supervision does not support cutmix_minmax; set "
                "`dataset.transforms.cutmix_minmax: null`."
            )
        if hcc_enabled:
            raise ValueError(
                "Direct subspace supervision is mutually exclusive with `hcc.enabled=true`."
            )
        if lex_enabled:
            raise ValueError(
                "Direct subspace supervision is mutually exclusive with "
                "`train.lexicographic.enabled=true`."
            )
        if hiercos_projection_enabled:
            raise ValueError(
                "Direct subspace supervision requires the Hier-COS LH projection to be off; "
                "set `model.projection.enabled=false`."
            )
    if lex_enabled:
        if model_name == "lhdnn":
            raise ValueError("Lexicographic training is not supported for LH-DNN.")
        if hiercos_projection_enabled:
            raise ValueError(
                "Hier-COS `model.projection.enabled=true` is mutually exclusive with "
                "`train.lexicographic.enabled=true`."
            )
        if depth != 3:
            raise ValueError("Lexicographic training requires exactly three hierarchy levels.")
        if model_name not in {"hcast", "hiercos", "ht_capsnet", "hrn"}:
            raise ValueError(
                "Lexicographic training requires H-CAST, Hier-COS, HT-CapsNet, or HRN."
            )
        _require_enum(
            lex.get("projection_mode", "coarse_first"),
            "train.lexicographic.projection_mode",
            {"coarse_first", "fine_first"},
        )
        if _finite_float(lex.get("eps", 1e-12), "train.lexicographic.eps") <= 0.0:
            raise ValueError("Enabled lexicographic training requires a positive epsilon.")
        if model_name == "hcast":
            loss = model["loss"]
            if bool(loss.get("globalkl", False)):
                raise ValueError("Lexicographic H-CAST requires model.loss.globalkl=false.")
        elif model_name == "hiercos":
            if model.get("loss") not in {"global_softmax_ce_reg", "level_softmax_ce_reg"}:
                raise ValueError("Lexicographic Hier-COS requires a decomposed CE loss.")
        elif model_name == "hrn" and model.get("loss", "native") != "level_conditional":
            raise ValueError(
                "Lexicographic HRN requires model.loss=level_conditional to expose "
                "coarse, middle, and fine objectives."
            )


def validate_config(
    cfg: Any,
    *,
    allow_partial: bool = False,
    source: Optional[str] = None,
) -> None:
    """Validate unknown keys, required values, and cross-section compatibility."""
    payload = _plain_config(cfg)
    model = payload.get("model")
    model_name = model.get("name") if isinstance(model, Mapping) else None
    if model_name not in SUPPORTED_MODELS:
        model_name = None
    try:
        _validate_unknown_keys(payload, "", model_name)
        if allow_partial:
            return
        missing = sorted(_REQUIRED_ROOT_SECTIONS - set(payload))
        if missing:
            raise ValueError(f"Missing required configuration sections: {missing}.")
        _validate_common_sections(payload)
        _validate_model_compatibility(payload)
    except (TypeError, ValueError) as exc:
        prefix = f"Invalid configuration {source}: " if source else "Invalid configuration: "
        raise type(exc)(prefix + str(exc)) from exc
