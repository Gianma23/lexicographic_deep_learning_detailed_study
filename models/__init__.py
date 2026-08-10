from typing import Any, Dict, List, Optional, Tuple, Union

import torch


def build_model(cfg: Any, num_classes_per_level: List[int], taxonomy: Optional[Dict] = None) -> torch.nn.Module:
    name = cfg.model.name
    if not isinstance(name, str):
        raise ValueError("model.name must be a string.")
    if name == "hcast":
        from .hcast.factory import build_model as build_hcast

        model = build_hcast(cfg, num_classes_per_level, taxonomy)
    elif name == "lhdnn":
        from .lhdnn.factory import build_model as build_lhdnn

        model = build_lhdnn(cfg, num_classes_per_level, taxonomy)
    elif name == "ht_capsnet":
        from .ht_capsnet.factory import build_model as build_caps

        model = build_caps(cfg, num_classes_per_level, taxonomy)
    elif name == "hrn":
        from .hrn.factory import build_model as build_hrn

        model = build_hrn(cfg, num_classes_per_level, taxonomy)
    elif name == "hiercos":
        from .hiercos.factory import build_model as build_hiercos

        model = build_hiercos(cfg, num_classes_per_level, taxonomy)
    else:
        raise ValueError(f"Unsupported model '{name}'. Expected one of ['hcast', 'lhdnn', 'ht_capsnet', 'hrn', 'hiercos']")

    return model


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
    from .common.subspace_supervision import (
        compute_subspace_supervision_loss,
        subspace_supervision_enabled,
    )

    if subspace_supervision_enabled(cfg):
        return compute_subspace_supervision_loss(
            output,
            targets,
            cfg,
            taxonomy,
            return_aux=return_aux,
        )

    name = cfg.model.name
    if not isinstance(name, str):
        raise ValueError("model.name must be a string.")
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
