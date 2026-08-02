from typing import Any, Dict, List, Optional

from ..common.hcc import resolve_hcc_cfg_from_top_level
from .model import HRNModel


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None):
    if len(num_classes_per_level) != 3:
        raise ValueError(
            f"HRN supports only 3 hierarchy levels. Received {len(num_classes_per_level)}: {num_classes_per_level}"
        )

    return HRNModel(
        num_classes_per_level=num_classes_per_level,
        backbone=cfg.model.get("backbone", "resnet50"),
        pretrained=bool(cfg.model.get("pretrained", True)),
        branch_hidden_dim=int(cfg.model.get("branch_hidden_dim", 1024)),
        embedding_dim=int(cfg.model.get("embedding_dim", 512)),
        dropout=float(cfg.model.get("dropout", 0.0)),
        trunk_lr_scale=float(cfg.model.get("trunk_lr_scale", 0.1)),
        taxonomy=taxonomy,
        hcc_cfg=resolve_hcc_cfg_from_top_level(cfg),
        train_epochs=int(cfg.train.get("epochs", 1)),
    )
