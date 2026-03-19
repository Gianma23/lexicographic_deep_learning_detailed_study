from typing import Any, Dict, List, Optional

from .model import HCASTModel


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None):
    dataset_mean = list(cfg.dataset.get("mean", [0.485, 0.456, 0.406]))
    dataset_std = list(cfg.dataset.get("std", [0.229, 0.224, 0.225]))
    segment_cfg = cfg.model.get("segments", {})
    lexproj_cfg = cfg.model.get("lexproj", {})
    design1_cfg = cfg.model.get("design1", {})
    return HCASTModel(
        num_classes_per_level=num_classes_per_level,
        variant=str(cfg.model.get("variant", "cast_small")),
        model_kwargs={
            "img_size": int(cfg.dataset.get("image_size", 224)),
            "pretrained": bool(cfg.model.get("pretrained", False)),
            "lexproj": {
                "enabled": bool(lexproj_cfg.get("enabled", False)),
                "scope": str(lexproj_cfg.get("scope", "train_eval")),
                "eta_lower": float(lexproj_cfg.get("eta_lower", 5.0)),
                "eta_upper": float(lexproj_cfg.get("eta_upper", 5.0)),
                "lower_iter_k": int(lexproj_cfg.get("lower_iter_k", 3)),
                "coarse_tol": float(lexproj_cfg.get("coarse_tol", 5e-3)),
                "middle_tol": float(lexproj_cfg.get("middle_tol", 5e-3)),
                "backtrack_factor": float(lexproj_cfg.get("backtrack_factor", 0.5)),
                "backtrack_max_steps": int(lexproj_cfg.get("backtrack_max_steps", 5)),
                "eps": float(lexproj_cfg.get("eps", 1e-12)),
            },
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
        design1_cfg={
            "enabled": bool(design1_cfg.get("enabled", False)),
            "scope": str(design1_cfg.get("scope", "train_eval")),
            "use_projected_for_loss": bool(design1_cfg.get("use_projected_for_loss", False)),
            "eps": float(design1_cfg.get("eps", 1e-12)),
        },
    )
