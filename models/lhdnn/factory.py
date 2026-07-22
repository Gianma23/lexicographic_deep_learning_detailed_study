from typing import Any, Dict, List, Optional

from .model import LHDNNModel


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None):
    projection_cfg = cfg.model.get("projection", {})

    return LHDNNModel(
        num_classes_per_level=num_classes_per_level,
        taxonomy=taxonomy,
        image_size=int(cfg.dataset.get("image_size", 32)),
        projection_cfg={
            "eps": float(projection_cfg.get("eps", 1e-6)),
        },
        in_channels=int(cfg.dataset.get("in_channels", 3)),
        adaptive_pool_size=cfg.model.get("adaptive_pool_size", None),
    )
