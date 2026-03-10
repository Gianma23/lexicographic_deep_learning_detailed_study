from typing import Any, Dict, List, Optional

from .model import HDCapsNet


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None):
    sec_dims_cfg = cfg.model.get("secondary_dims")
    if sec_dims_cfg is None:
        secondary_dims = [int(cfg.model.get("caps_dim", 16)) for _ in num_classes_per_level]
    else:
        secondary_dims = [int(v) for v in sec_dims_cfg]

    return HDCapsNet(
        num_classes_per_level=num_classes_per_level,
        taxonomy=taxonomy,
        primary_dim=int(cfg.model.get("caps_dim", 16)),
        secondary_dims=secondary_dims,
        routing_iters=int(cfg.model.get("routing_iters", 3)),
        num_blocks=int(cfg.model.get("num_blocks", 4)),
        initial_filters=int(cfg.model.get("initial_filters", 64)),
        filter_increment=int(cfg.model.get("filter_increment", 2)),
    )
