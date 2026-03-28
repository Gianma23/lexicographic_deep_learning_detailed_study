from typing import Any, Dict, List, Optional

from .model import LHDNNModel


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None):
    projection_cfg = cfg.model.get("projection", {})
    advantage_cfg = cfg.model.get("advantage", {})
    custom_cfg = cfg.model.get("custom", {})

    return LHDNNModel(
        num_classes_per_level=num_classes_per_level,
        taxonomy=taxonomy,
        image_size=int(cfg.dataset.get("image_size", 32)),
        variant=str(cfg.model.get("variant", "small")),
        custom_cfg={
            "channels": list(custom_cfg.get("channels", [])),
            "shared_dim": int(custom_cfg.get("shared_dim", 0)),
        },
        projection_cfg={
            "enabled": bool(projection_cfg.get("enabled", True)),
            "eps": float(projection_cfg.get("eps", 1e-6)),
            "use_relu_derivative": bool(projection_cfg.get("use_relu_derivative", True)),
        },
        advantage_cfg={
            "enabled": bool(advantage_cfg.get("enabled", True)),
        },
        in_channels=int(cfg.dataset.get("in_channels", 3)),
    )
