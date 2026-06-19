from typing import Any, Dict, List, Optional

from models.orthonormal_plugin.config import is_enabled, plugin_section, validate_disabled_mixup

from .model import HierCosModel


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None):
    if taxonomy is None:
        raise ValueError("Hier-COS requires taxonomy with `parent_of` mappings.")

    model_cfg = getattr(cfg, "model", {})
    plugin_enabled = is_enabled(cfg)
    if plugin_enabled:
        validate_disabled_mixup(cfg)
    plugin_cfg = plugin_section(cfg) if plugin_enabled else {}
    return HierCosModel(
        num_classes_per_level=num_classes_per_level,
        taxonomy=taxonomy,
        variant=model_cfg.get("variant", "haframe_resnet50"),
        transform_mode=plugin_cfg.get("transform_mode", model_cfg.get("transform_mode", "full")),
        pretrained=bool(model_cfg.get("pretrained", True)),
        pool=model_cfg.get("pool", "max"),
        backbone_lr_scale=float(model_cfg.get("backbone_lr_scale", 0.1)),
        transform_lr_scale=float(plugin_cfg.get("transform_lr_scale", model_cfg.get("transform_lr_scale", 1.0))),
        fixed_frame_mode=plugin_cfg.get("fixed_frame_mode", model_cfg.get("fixed_frame_mode", "orthonormal_random")),
        wide_depth=int(model_cfg.get("wide_depth", 28)),
        wide_widen_factor=int(model_cfg.get("wide_widen_factor", 8)),
        wide_drop_rate=float(model_cfg.get("wide_drop_rate", 0.0)),
    )
