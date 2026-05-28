from typing import Any, List

import torch

from .types import LexicographicConfig
from ..runtime.common import section_to_dict


def _parse_nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        parsed = int(default)
    return max(parsed, 0)


def _resolve_hiercos_loss_mode(model_cfg) -> str:
    raw_mode = model_cfg.get("loss", "kl_reg")
    if raw_mode is None:
        raw_mode = "kl_reg"
    if not isinstance(raw_mode, str):
        raise ValueError(
            "train.lexicographic.enabled=true with model.name='hiercos' requires "
            "scalar `model.loss` set to one of "
            "['per_level_kl_reg', 'per_level_ce']."
        )
    mode = raw_mode
    if mode == "kl_reg":
        return "kl_reg"
    if mode == "per_level_kl_reg":
        return "per_level_kl_reg"
    if mode == "per_level_ce":
        return "per_level_ce"
    raise ValueError(
        f"Unsupported Hier-COS model.loss '{raw_mode}'. "
        "Expected one of ['kl_reg', 'per_level_kl_reg', 'per_level_ce']."
    )


def resolve_lexicographic_config(cfg: Any) -> LexicographicConfig:
    train_cfg = section_to_dict(getattr(cfg, "train", None))
    raw_lex_cfg = section_to_dict(train_cfg.get("lexicographic", None))
    enabled = bool(raw_lex_cfg.get("enabled", False))
    eps = float(raw_lex_cfg.get("eps", 1e-12))
    if eps <= 0.0:
        eps = 1e-12
    log_metrics = bool(raw_lex_cfg.get("log_metrics", True))
    start_epoch = _parse_nonnegative_int(raw_lex_cfg.get("start_epoch", 0))
    return LexicographicConfig(
        enabled=enabled,
        eps=eps,
        log_metrics=log_metrics,
        start_epoch=start_epoch,
    )


def validate_lexicographic_requirements(cfg: Any, level_losses: List[torch.Tensor]) -> None:
    model_cfg = section_to_dict(getattr(cfg, "model", None))
    model_name = model_cfg.get("name", "")
    if not isinstance(model_name, str):
        raise ValueError("model.name must be a string.")
    if model_name not in {"hcast", "hiercos"}:
        raise ValueError(
            "train.lexicographic.enabled=true is currently supported only for "
            "model.name in ['hcast', 'hiercos']."
        )

    if model_name == "hcast":
        loss_cfg = section_to_dict(model_cfg.get("loss", None))
        if bool(loss_cfg.get("globalkl", False)):
            raise ValueError(
                "train.lexicographic.enabled=true requires model.loss.globalkl=false "
                "(pure level-loss lexicographic mode)."
            )

    if model_name == "hiercos" and _resolve_hiercos_loss_mode(model_cfg) not in {
        "per_level_kl_reg",
        "per_level_ce",
    }:
        raise ValueError(
            "train.lexicographic.enabled=true with model.name='hiercos' requires "
            "`model.loss: per_level_kl_reg` or `model.loss: per_level_ce`; "
            "plain `kl_reg` does not expose "
            "differentiable per-level losses."
        )

    if len(level_losses) != 3:
        raise ValueError(
            "train.lexicographic.enabled=true requires exactly 3 differentiable level losses "
            "(coarse, mid, fine)."
        )
