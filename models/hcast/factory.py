from typing import Any, Dict, List, Optional

from .model import HCASTModel


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None):
    _ = taxonomy
    return HCASTModel(
        num_classes_per_level=num_classes_per_level,
        variant=str(cfg.model.get("variant", "cast_small")),
        fallback_cfg={
            "hidden_dim": int(cfg.model.get("hidden_dim", 512)),
            "dropout": float(cfg.model.get("dropout", 0.2)),
        },
    )
