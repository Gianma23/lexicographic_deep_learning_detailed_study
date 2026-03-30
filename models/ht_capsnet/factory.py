from typing import Any, Dict, List, Optional

from .model import HTCapsNet


def _resolve_secondary_dims(cfg: Any, num_classes_per_level: List[int], primary_dim: int) -> List[int]:
    sec_dims_cfg = cfg.model.get("secondary_dims")
    if sec_dims_cfg is None:
        return [int(primary_dim) for _ in num_classes_per_level]

    secondary_dims = [int(v) for v in sec_dims_cfg]
    if len(secondary_dims) != len(num_classes_per_level):
        raise ValueError(
            "model.secondary_dims length must match hierarchy depth "
            f"({len(num_classes_per_level)}), got {len(secondary_dims)}."
        )
    return secondary_dims


def _assert_reproducibility_cfg(cfg: Any) -> None:
    train_cfg = getattr(cfg, "train", {})
    runtime_cfg = getattr(cfg, "runtime", {})

    seed = train_cfg.get("seed", None) if hasattr(train_cfg, "get") else None
    if seed is None:
        raise ValueError("HT-CapsNet reproducibility check failed: `train.seed` must be set.")

    deterministic = (
        bool(runtime_cfg.get("deterministic", True))
        if hasattr(runtime_cfg, "get")
        else True
    )
    if not deterministic:
        raise ValueError(
            "HT-CapsNet reproducibility check failed: set `runtime.deterministic: true`."
        )


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None):
    """Build an `HTCapsNet` instance from the project config object."""
    _assert_reproducibility_cfg(cfg)

    primary_dim = int(cfg.model.get("caps_dim", 16))
    secondary_dims = _resolve_secondary_dims(cfg, num_classes_per_level, primary_dim)
    input_size = int(cfg.dataset.get("image_size", 224))

    return HTCapsNet(
        num_classes_per_level=num_classes_per_level,
        taxonomy=taxonomy,
        primary_dim=primary_dim,
        secondary_dims=secondary_dims,
        routing_iters=int(cfg.model.get("routing_iters", 3)),
        num_blocks=int(cfg.model.get("num_blocks", 4)),
        initial_filters=int(cfg.model.get("initial_filters", 64)),
        filter_increment=int(cfg.model.get("filter_increment", 2)),
        backbone_name=str(cfg.model.get("backbone_net", "custom")),
        backbone_weights=cfg.model.get("backbone_net_weights", None),
        taxonomy_temperature=float(cfg.model.get("taxonomy_temperature", 0.5)),
        mask_threshold_high=float(cfg.model.get("mask_threshold_high", 0.9)),
        mask_threshold_low=float(cfg.model.get("mask_threshold_low", 0.1)),
        mask_temperature=float(cfg.model.get("mask_temperature", 0.5)),
        mask_center=float(cfg.model.get("mask_center", 0.5)),
        attn_heads=int(cfg.model.get("attn_heads", 16)),
        attn_dropout=float(cfg.model.get("attn_dropout", 0.0)),
        attn_postprocess=str(cfg.model.get("attn_postprocess", "layernorm")),
        input_size=input_size,
    )
