from typing import Any, Dict, List, Optional

from models.common.hcc import resolve_hcc_cfg_from_top_level

from .config import parse_bool, section_to_dict
from .model import HierCosModel


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None):
    if taxonomy is None:
        raise ValueError("Hier-COS requires taxonomy with `parent_of` mappings.")

    model_cfg = getattr(cfg, "model", {})
    return HierCosModel(
        num_classes_per_level=num_classes_per_level,
        taxonomy=taxonomy,
        variant=model_cfg.get("variant", "haframe_resnet50"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        pool=model_cfg.get("pool", "max"),
        backbone_lr_scale=float(model_cfg.get("backbone_lr_scale", 0.1)),
        transform_lr_scale=float(model_cfg.get("transform_lr_scale", 1.0)),
        fixed_frame_mode=model_cfg.get("fixed_frame_mode", "orthonormal_random"),
        fixed_frame_per_level=parse_bool(
            model_cfg.get("fixed_frame_per_level", False),
            default=False,
        ),
        projection_cfg=section_to_dict(model_cfg.get("projection", None)),
        wide_depth=int(model_cfg.get("wide_depth", 28)),
        wide_widen_factor=int(model_cfg.get("wide_widen_factor", 8)),
        wide_drop_rate=float(model_cfg.get("wide_drop_rate", 0.0)),
        hcc_cfg=resolve_hcc_cfg_from_top_level(cfg),
    )
