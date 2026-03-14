from typing import Any, Dict, List, Optional, Tuple

import torch


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None) -> torch.nn.Module:
    name = str(cfg.model.name).lower()
    if name == "hcast":
        from .hcast.factory import build_model as build_hcast

        return build_hcast(cfg, num_classes_per_level, taxonomy)
    if name == "ht_capsnet":
        from .ht_capsnet.factory import build_model as build_caps

        return build_caps(cfg, num_classes_per_level, taxonomy)
    raise ValueError(f"Unsupported model '{name}'. Expected one of ['hcast', 'ht_capsnet']")


def compute_loss(
    cfg: Any,
    output: Dict[str, Any],
    targets: torch.Tensor,
    taxonomy: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    name = str(cfg.model.name).lower()
    if name == "hcast":
        from .hcast.losses import compute_loss as loss_hcast

        return loss_hcast(output, targets, cfg, taxonomy)
    if name == "ht_capsnet":
        from .ht_capsnet.losses import compute_loss as loss_caps

        return loss_caps(output, targets, cfg, taxonomy)
    raise ValueError(f"Unsupported model '{name}'. Expected one of ['hcast', 'ht_capsnet']")


__all__ = ["build_model", "compute_loss"]
