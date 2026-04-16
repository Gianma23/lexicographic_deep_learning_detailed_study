from typing import Any, Dict, List, Optional

from .model import HCASTModel


def _section_to_dict(section: Any) -> Dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if hasattr(section, "items"):
        return {k: v for k, v in section.items()}
    return {}


def _resolve_hcc_cfg(cfg: Any) -> Dict[str, Any]:
    # Canonical location for model-agnostic constraints.
    hcc_cfg = _section_to_dict(getattr(cfg, "hcc", None))
    if hcc_cfg:
        return hcc_cfg
    if hasattr(cfg, "get"):
        hcc_cfg = _section_to_dict(cfg.get("hcc", None))
        if hcc_cfg:
            return hcc_cfg

    # Backward-compatible fallback.
    model_cfg = _section_to_dict(getattr(cfg, "model", None))
    return _section_to_dict(model_cfg.get("hcc", None))


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None):
    dataset_mean = list(cfg.dataset.get("mean", [0.485, 0.456, 0.406]))
    dataset_std = list(cfg.dataset.get("std", [0.229, 0.224, 0.225]))
    segment_cfg = cfg.model.get("segments", {})
    hcc_cfg = _resolve_hcc_cfg(cfg)
    model_drop = float(cfg.model.get("drop", 0.0))
    model_drop_path = float(cfg.model.get("drop_path", 0.1))
    return HCASTModel(
        num_classes_per_level=num_classes_per_level,
        variant=str(cfg.model.get("variant", "cast_small")),
        model_kwargs={
            "img_size": int(cfg.dataset.get("image_size", 224)),
            "pretrained": bool(cfg.model.get("pretrained", False)),
            "drop_rate": model_drop,
            "drop_path_rate": model_drop_path,
            "drop_block_rate": None,
        },
        segments_cfg={
            "mode": str(segment_cfg.get("mode", "grid")),
            "patch_size": int(segment_cfg.get("patch_size", 8)),
            "num_superpixels": int(segment_cfg.get("num_superpixels", 196)),
            "num_levels": int(segment_cfg.get("num_levels", 1)),
            "prior": int(segment_cfg.get("prior", 2)),
            "histogram_bins": int(segment_cfg.get("histogram_bins", 5)),
            "double_step": bool(segment_cfg.get("double_step", False)),
            "num_iterations": int(segment_cfg.get("num_iterations", 15)),
            "mean": list(segment_cfg.get("mean", dataset_mean)),
            "std": list(segment_cfg.get("std", dataset_std)),
        },
        taxonomy=taxonomy,
        hcc_cfg=hcc_cfg,
        train_epochs=int(cfg.train.get("epochs", 1)),
    )
