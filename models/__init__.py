from typing import Any, Dict, List, Optional, Tuple, Union

import torch


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None) -> torch.nn.Module:
    name = str(cfg.model.name).lower()
    if name == "hcast":
        from .hcast.factory import build_model as build_hcast

        return build_hcast(cfg, num_classes_per_level, taxonomy)
    if name == "lhdnn":
        from .lhdnn.factory import build_model as build_lhdnn

        return build_lhdnn(cfg, num_classes_per_level, taxonomy)
    if name == "ht_capsnet":
        from .ht_capsnet.factory import build_model as build_caps

        return build_caps(cfg, num_classes_per_level, taxonomy)
    if name == "hrn":
        from .hrn.factory import build_model as build_hrn

        return build_hrn(cfg, num_classes_per_level, taxonomy)
    if name == "hiercos":
        from .hiercos.factory import build_model as build_hiercos

        return build_hiercos(cfg, num_classes_per_level, taxonomy)
    raise ValueError(f"Unsupported model '{name}'. Expected one of ['hcast', 'lhdnn', 'ht_capsnet', 'hrn', 'hiercos']")


def compute_loss(
    cfg: Any,
    output: Dict[str, Any],
    targets: Any,
    taxonomy: Optional[Dict[str, Any]] = None,
    return_aux: bool = False,
) -> Union[
    Tuple[torch.Tensor, Dict[str, float]],
    Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]],
]:
    name = str(cfg.model.name).lower()
    if name == "hcast":
        from .hcast.losses import compute_loss as loss_hcast

        return loss_hcast(output, targets, cfg, taxonomy, return_aux=return_aux)
    if name == "lhdnn":
        from .lhdnn.losses import compute_loss as loss_lhdnn

        return loss_lhdnn(output, targets, cfg, taxonomy, return_aux=return_aux)
    if name == "ht_capsnet":
        from .ht_capsnet.losses import compute_loss as loss_caps

        return loss_caps(output, targets, cfg, taxonomy, return_aux=return_aux)
    if name == "hrn":
        from .hrn.losses import compute_loss as loss_hrn

        return loss_hrn(output, targets, cfg, taxonomy, return_aux=return_aux)
    if name == "hiercos":
        from .hiercos.losses import compute_loss as loss_hiercos

        return loss_hiercos(output, targets, cfg, taxonomy, return_aux=return_aux)
    raise ValueError(f"Unsupported model '{name}'. Expected one of ['hcast', 'lhdnn', 'ht_capsnet', 'hrn', 'hiercos']")


__all__ = ["build_model", "compute_loss"]
