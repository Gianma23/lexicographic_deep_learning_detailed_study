from typing import Any, Dict, List, Optional

from .model import HTCapsNet


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None):
    sec_dims_cfg = cfg.model.get("secondary_dims")
    if sec_dims_cfg is None:
        secondary_dims = [int(cfg.model.get("caps_dim", 16)) for _ in num_classes_per_level]
    else:
        secondary_dims = [int(v) for v in sec_dims_cfg]
    backbone_name = str(cfg.model.get("backbone_net", cfg.model.get("backbone_name", "custom")))
    raw_backbone_weights = cfg.model.get("backbone_net_weights", cfg.model.get("backbone_weights", None))
    backbone_weights = None if raw_backbone_weights is None else str(raw_backbone_weights)

    return HTCapsNet(
        num_classes_per_level=num_classes_per_level,
        taxonomy=taxonomy,
        primary_dim=int(cfg.model.get("caps_dim", 16)),
        secondary_dims=secondary_dims,
        routing_iters=int(cfg.model.get("routing_iters", 3)),
        num_blocks=int(cfg.model.get("num_blocks", 4)),
        initial_filters=int(cfg.model.get("initial_filters", 64)),
        filter_increment=int(cfg.model.get("filter_increment", 2)),
        backbone_name=backbone_name,
        backbone_weights=backbone_weights,
        taxonomy_temperature=float(cfg.model.get("taxonomy_temperature", 0.5)),
        mask_threshold_high=float(cfg.model.get("mask_threshold_high", 0.9)),
        mask_threshold_low=float(cfg.model.get("mask_threshold_low", 0.1)),
        mask_temperature=float(cfg.model.get("mask_temperature", 0.5)),
        mask_center=float(cfg.model.get("mask_center", 0.5)),
        attn_heads=int(cfg.model.get("attn_heads", 16)),
        attn_dropout=float(cfg.model.get("attn_dropout", 0.0)),
        input_size=int(cfg.dataset.get("image_size", 224)),
    )
