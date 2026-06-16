from typing import Any, List

import torch

from .types import LexicographicConfig
from ..runtime.common import section_to_dict


_LEX_PROJECTION_MODES = ("coarse_first", "fine_first", "pairwise_orthogonal")
_LEX_PROJECTION_RULES = ("orthogonalize_all", "conflict_only")


def _parse_nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        parsed = int(default)
    return max(parsed, 0)


def _parse_positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        parsed = float(default)
    return parsed if parsed > 0.0 else float(default)


def _resolve_hiercos_loss_mode(model_cfg) -> str:
    raw_mode = model_cfg.get("loss", "kl_reg")
    if raw_mode is None:
        raw_mode = "kl_reg"
    if not isinstance(raw_mode, str):
        raise ValueError(
            "train.lexicographic.enabled=true with model.name='hiercos' requires "
            "scalar `model.loss` set to one of "
            "['global_softmax_ce_reg', 'level_softmax_ce_reg']."
        )
    mode = raw_mode
    if mode == "kl_reg":
        return "kl_reg"
    if mode == "global_softmax_ce_reg":
        return "global_softmax_ce_reg"
    if mode == "level_softmax_ce_reg":
        return "level_softmax_ce_reg"
    raise ValueError(
        f"Unsupported Hier-COS model.loss '{raw_mode}'. "
        "Expected one of ['kl_reg', 'global_softmax_ce_reg', 'level_softmax_ce_reg']."
    )


def _resolve_lex_projection_mode(value: Any, default: str = "coarse_first") -> str:
    raw_mode = default if value is None else value
    if not isinstance(raw_mode, str):
        raise ValueError(
            "train.lexicographic.projection_mode must be a string: one of "
            f"{list(_LEX_PROJECTION_MODES)}."
        )
    if raw_mode not in _LEX_PROJECTION_MODES:
        raise ValueError(
            f"Unsupported train.lexicographic.projection_mode '{raw_mode}'. "
            f"Expected one of {list(_LEX_PROJECTION_MODES)}."
        )
    return raw_mode


def _resolve_lex_projection_rule(value: Any, default: str = "orthogonalize_all") -> str:
    raw_rule = default if value is None else value
    if not isinstance(raw_rule, str):
        raise ValueError(
            "train.lexicographic.projection_rule must be a string: one of "
            f"{list(_LEX_PROJECTION_RULES)}."
        )
    if raw_rule not in _LEX_PROJECTION_RULES:
        raise ValueError(
            f"Unsupported train.lexicographic.projection_rule '{raw_rule}'. "
            f"Expected one of {list(_LEX_PROJECTION_RULES)}."
        )
    return raw_rule


def resolve_lexicographic_config(cfg: Any) -> LexicographicConfig:
    train_cfg = section_to_dict(getattr(cfg, "train", None))
    raw_lex_cfg = section_to_dict(train_cfg.get("lexicographic", None))
    enabled = bool(raw_lex_cfg.get("enabled", False))
    eps = _parse_positive_float(raw_lex_cfg.get("eps", 1e-12), default=1e-12)
    log_metrics = bool(raw_lex_cfg.get("log_metrics", True))
    start_epoch = _parse_nonnegative_int(raw_lex_cfg.get("start_epoch", 0))
    projection_mode = _resolve_lex_projection_mode(raw_lex_cfg.get("projection_mode", "coarse_first"))
    projection_rule = _resolve_lex_projection_rule(raw_lex_cfg.get("projection_rule", "orthogonalize_all"))

    return LexicographicConfig(
        enabled=enabled,
        eps=eps,
        log_metrics=log_metrics,
        start_epoch=start_epoch,
        projection_mode=projection_mode,
        projection_rule=projection_rule,
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
        "global_softmax_ce_reg",
        "level_softmax_ce_reg",
    }:
        raise ValueError(
            "train.lexicographic.enabled=true with model.name='hiercos' requires "
            "`model.loss: global_softmax_ce_reg` or `model.loss: level_softmax_ce_reg`; "
            "plain `kl_reg` does not expose "
            "differentiable per-level losses."
        )

    if len(level_losses) != 3:
        raise ValueError(
            "train.lexicographic.enabled=true requires exactly 3 differentiable level losses "
            "(coarse, mid, fine)."
        )
