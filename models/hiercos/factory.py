from typing import Any, Dict, List, Optional

from .model import HierCosModel


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None):
    if taxonomy is None:
        raise ValueError("Hier-COS requires taxonomy with `parent_of` mappings.")

    model_cfg = getattr(cfg, "model", {})
    return HierCosModel(
        num_classes_per_level=num_classes_per_level,
        taxonomy=taxonomy,
        variant=str(model_cfg.get("variant", "haframe_resnet50")),
        feature_space=str(model_cfg.get("feature_space", "hier-cos")),
        pretrained=bool(model_cfg.get("pretrained", True)),
        pool=str(model_cfg.get("pool", "average")),
        backbone_lr_scale=float(model_cfg.get("backbone_lr_scale", 0.1)),
        fixed_frame_mode=str(model_cfg.get("fixed_frame_mode", "orthonormal_random")),
        wide_depth=int(model_cfg.get("wide_depth", 28)),
        wide_widen_factor=int(model_cfg.get("wide_widen_factor", 8)),
        wide_drop_rate=float(model_cfg.get("wide_drop_rate", 0.0)),
    )
