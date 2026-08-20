import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .config import section_to_dict


HierCosTargets = Union[torch.Tensor, Dict[str, Any]]
_EPS = 1e-12
_LOSS_MODES = ("kl_reg", "global_softmax_ce_reg", "level_softmax_ce_reg")
_HIER_COS_WEIGHT_MODES = (
    "equal",
    "kl_leaf",
    "kl_coarse",
    "cumulative_branching",
    "marginal_branching",
)


def _to_scalar(value: torch.Tensor) -> float:
    return float(value.detach().item())


def _hard_targets_from_input(targets: HierCosTargets, num_levels: int) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        hard_targets = targets
    elif isinstance(targets, dict):
        hard_targets = targets.get("hard_targets")
        if hard_targets is None:
            hard_targets = targets.get("labels_a")
    else:
        hard_targets = None

    if not isinstance(hard_targets, torch.Tensor):
        raise TypeError("Hier-COS expects hard targets as tensor [B, L].")
    if hard_targets.ndim != 2 or int(hard_targets.size(1)) != int(num_levels):
        raise ValueError(
            f"Hier-COS expected hard targets with shape [B, {num_levels}], "
            f"got {tuple(hard_targets.shape)}."
        )
    return hard_targets.long()


def _loss_settings(cfg: Any) -> Tuple[Dict[str, Any], str, str, float]:
    model_cfg = section_to_dict(getattr(cfg, "model", None))
    if not model_cfg and hasattr(cfg, "get"):
        model_cfg = section_to_dict(cfg.get("model", None))
    return model_cfg, "model", "kl_reg", 1.0


def _resolve_mode(
    cfg: Any,
    key: str,
    valid_modes: Tuple[str, ...],
) -> str:
    settings, source, default, _alpha_default = _loss_settings(cfg)
    raw_mode = settings.get(key, default)
    if raw_mode is None:
        raw_mode = default
    if not isinstance(raw_mode, str):
        raise ValueError(f"{source}.{key} must be a scalar string: one of {list(valid_modes)}.")
    if raw_mode not in valid_modes:
        raise ValueError(f"Unsupported {source}.{key} '{raw_mode}'. Expected one of {list(valid_modes)}.")
    return raw_mode


def _resolve_loss_mode(cfg: Any) -> str:
    return _resolve_mode(cfg=cfg, key="loss", valid_modes=_LOSS_MODES)


def _resolve_weight_mode(cfg: Any) -> str:
    return _resolve_mode(cfg=cfg, key="weight_mode", valid_modes=_HIER_COS_WEIGHT_MODES)


def _resolve_weight_beta(cfg: Any) -> float:
    settings, source, _default, _alpha_default = _loss_settings(cfg)
    raw_beta = settings.get("weight_beta", 0.5)
    if isinstance(raw_beta, bool):
        raise ValueError(f"{source}.weight_beta must be a finite number >= 0, got {raw_beta!r}.")
    try:
        beta = float(raw_beta)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source}.weight_beta must be a finite number >= 0, got {raw_beta!r}."
        ) from exc
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError(f"{source}.weight_beta must be a finite number >= 0, got {raw_beta!r}.")
    return beta


def _resolve_alpha(cfg: Any) -> float:
    settings, _source, _default_loss, alpha_default = _loss_settings(cfg)
    return float(settings.get("alpha", alpha_default))


def _resolve_leaf_targets(hard_targets: torch.Tensor) -> torch.Tensor:
    if hard_targets.numel() == 0:
        return hard_targets.new_zeros((hard_targets.size(0),), dtype=torch.long)
    return hard_targets[:, -1].long()


def _check_index_range(indices: torch.Tensor, upper_bound: int, message: str) -> None:
    if bool((indices < 0).any()) or bool((indices >= upper_bound).any()):
        raise ValueError(message)


def _get_tensor(output: Dict[str, Any], *keys: str) -> Optional[torch.Tensor]:
    for key in keys:
        value = output.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _get_list(output: Dict[str, Any], *keys: str) -> Optional[List[torch.Tensor]]:
    for key in keys:
        value = output.get(key)
        if isinstance(value, list):
            return value
    return None


def _validate_output(
    output: Dict[str, Any],
    num_levels: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    node_logits = _get_tensor(output, "node_logits")
    if not isinstance(node_logits, torch.Tensor) or node_logits.ndim != 2:
        raise ValueError("Hier-COS output must provide `node_logits` with shape [B, N].")

    level_node_ids = _get_list(
        output,
        "hiercos_level_node_ids",
    )
    if not isinstance(level_node_ids, list) or len(level_node_ids) != num_levels:
        raise ValueError("Hier-COS output must provide level node ids aligned with hierarchy depth.")
    if not all(isinstance(node_ids, torch.Tensor) and node_ids.ndim == 1 for node_ids in level_node_ids):
        raise ValueError("Hier-COS level node id entries must be 1D tensors.")

    leaf_to_level_local = _get_tensor(
        output,
        "leaf_to_level_local",
    )
    if not isinstance(leaf_to_level_local, torch.Tensor) or leaf_to_level_local.ndim != 2:
        raise ValueError("Hier-COS output must provide `leaf_to_level_local` with shape [num_leaf, depth].")
    if int(leaf_to_level_local.size(1)) != num_levels:
        raise ValueError(
            "Hier-COS `leaf_to_level_local` depth mismatch: "
            f"expected {num_levels}, found {int(leaf_to_level_local.size(1))}."
        )

    node_prob_weights = _get_tensor(
        output,
        "node_prob_weights",
    )
    if not isinstance(node_prob_weights, torch.Tensor) or node_prob_weights.ndim != 1:
        raise ValueError("Hier-COS output must provide `node_prob_weights` with shape [depth].")
    if int(node_prob_weights.numel()) != num_levels:
        raise ValueError(
            "Hier-COS `node_prob_weights` depth mismatch: "
            f"expected {num_levels}, found {int(node_prob_weights.numel())}."
        )
    return node_logits, level_node_ids, leaf_to_level_local


def _validate_level_logits_list(
    logits_per_level: List[torch.Tensor],
    level_node_ids: List[torch.Tensor],
    num_levels: int,
    field_name: str,
) -> List[torch.Tensor]:
    if len(logits_per_level) != int(num_levels):
        raise ValueError(
            f"Hier-COS {field_name} must be aligned with hierarchy depth: "
            f"expected {num_levels}, found {len(logits_per_level)}."
        )
    validated: List[torch.Tensor] = []
    batch_size: Optional[int] = None
    for level, (level_logits, node_ids) in enumerate(zip(logits_per_level, level_node_ids)):
        if not isinstance(level_logits, torch.Tensor) or level_logits.ndim != 2:
            raise ValueError(f"Hier-COS {field_name} for level {level} must have shape [B, C].")
        if batch_size is None:
            batch_size = int(level_logits.size(0))
        elif int(level_logits.size(0)) != batch_size:
            raise ValueError(f"Hier-COS {field_name} must share one batch size.")
        expected_width = int(node_ids.numel())
        if int(level_logits.size(1)) != expected_width:
            raise ValueError(
                f"Hier-COS {field_name} for level {level} has width {int(level_logits.size(1))}; "
                f"expected {expected_width}."
            )
        validated.append(level_logits)
    return validated


def _validate_node_logits_per_level(
    output: Dict[str, Any],
    level_node_ids: List[torch.Tensor],
    num_levels: int,
) -> Optional[List[torch.Tensor]]:
    node_logits_per_level = _get_list(
        output,
        "node_logits_per_level",
    )
    if node_logits_per_level is None:
        return None
    return _validate_level_logits_list(
        logits_per_level=node_logits_per_level,
        level_node_ids=level_node_ids,
        num_levels=num_levels,
        field_name="node logits per level",
    )


def _validate_effective_node_logits_per_level(
    output: Dict[str, Any],
    level_node_ids: List[torch.Tensor],
    num_levels: int,
) -> Optional[List[torch.Tensor]]:
    """Validate HCC-modified fixed-layer node logits when they are active."""
    effective = output.get("effective_node_logits_per_level")
    if effective is None:
        return None
    if not isinstance(effective, list):
        raise ValueError(
            "`effective_node_logits_per_level` must be None or a list aligned with hierarchy depth."
        )
    return _validate_level_logits_list(
        logits_per_level=effective,
        level_node_ids=level_node_ids,
        num_levels=num_levels,
        field_name="effective node logits per level",
    )


def _path_global_node_ids(
    leaf_targets: torch.Tensor,
    level_node_ids: List[torch.Tensor],
    leaf_to_level_local: torch.Tensor,
    device: torch.device,
) -> List[torch.Tensor]:
    global_node_ids: List[torch.Tensor] = []
    for level, level_nodes in enumerate(level_node_ids):
        level_nodes = level_nodes.to(device=device, dtype=torch.long)
        level_size = int(level_nodes.numel())
        local_ids = leaf_to_level_local[leaf_targets, level].to(device=device, dtype=torch.long)
        _check_index_range(local_ids, level_size, f"Hier-COS invalid local node indices for level {level}.")
        global_node_ids.append(level_nodes[local_ids])
    return global_node_ids


def _build_node_targets(
    leaf_targets: torch.Tensor,
    level_node_ids: List[torch.Tensor],
    leaf_to_level_local: torch.Tensor,
    level_weights: torch.Tensor,
    total_nodes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = int(leaf_targets.size(0))
    device = leaf_targets.device
    targets = torch.zeros((batch_size, total_nodes), device=device, dtype=dtype)
    row_ids = torch.arange(batch_size, device=device, dtype=torch.long)

    for level, global_ids in enumerate(
        _path_global_node_ids(
            leaf_targets=leaf_targets,
            level_node_ids=level_node_ids,
            leaf_to_level_local=leaf_to_level_local,
            device=device,
        )
    ):
        targets[row_ids, global_ids] = level_weights[level]

    return targets / targets.sum(dim=1, keepdim=True).clamp_min(_EPS)


def _level_regularization_loss(
    abs_node_logits: torch.Tensor,
    leaf_targets: torch.Tensor,
    level_node_ids: List[torch.Tensor],
    leaf_to_level_local: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    level_losses: List[torch.Tensor] = []
    for level, level_nodes in enumerate(level_node_ids):
        level_nodes = level_nodes.to(device=abs_node_logits.device, dtype=torch.long)
        level_size = int(level_nodes.numel())
        level_logits = abs_node_logits.index_select(dim=1, index=level_nodes)
        level_logits = level_logits / level_logits.norm(dim=1, keepdim=True).clamp_min(_EPS)

        level_targets = leaf_to_level_local[leaf_targets, level].to(device=abs_node_logits.device, dtype=torch.long)
        _check_index_range(
            level_targets,
            level_size,
            f"Hier-COS invalid target labels for regularization at level {level}.",
        )

        one_hot = F.one_hot(level_targets, num_classes=level_size).to(dtype=level_logits.dtype)
        level_losses.append((one_hot - level_logits).abs().sum(dim=1).mean())

    reg = torch.stack(level_losses).sum() if level_losses else abs_node_logits.new_zeros(())
    return reg, level_losses


def _level_regularization_loss_from_level_logits(
    abs_node_logits_per_level: List[torch.Tensor],
    leaf_targets: torch.Tensor,
    leaf_to_level_local: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    level_losses: List[torch.Tensor] = []
    for level, level_logits in enumerate(abs_node_logits_per_level):
        level_size = int(level_logits.size(1))
        level_logits = level_logits / level_logits.norm(dim=1, keepdim=True).clamp_min(_EPS)
        level_targets = leaf_to_level_local[leaf_targets, level].to(device=level_logits.device, dtype=torch.long)
        _check_index_range(
            level_targets,
            level_size,
            f"Hier-COS invalid target labels for regularization at level {level}.",
        )

        one_hot = F.one_hot(level_targets, num_classes=level_size).to(dtype=level_logits.dtype)
        level_losses.append((one_hot - level_logits).abs().sum(dim=1).mean())

    if level_losses:
        return torch.stack(level_losses).sum(), level_losses
    return leaf_targets.new_zeros((), dtype=torch.float32), level_losses


def _weighted_target_ce_level_losses(
    abs_node_logits: torch.Tensor,
    level_node_ids: List[torch.Tensor],
    leaf_targets: torch.Tensor,
    leaf_to_level_local: torch.Tensor,
    level_weights: torch.Tensor,
    softmax_scope: str,
) -> List[torch.Tensor]:
    if softmax_scope not in {"global", "level"}:
        raise ValueError(f"Unsupported Hier-COS softmax scope '{softmax_scope}'.")

    weights = level_weights / level_weights.sum().clamp_min(_EPS)
    row_ids = torch.arange(int(leaf_targets.size(0)), device=abs_node_logits.device, dtype=torch.long)
    global_log_probs = F.log_softmax(abs_node_logits, dim=1) if softmax_scope == "global" else None
    level_losses: List[torch.Tensor] = []
    for level, level_nodes in enumerate(level_node_ids):
        level_nodes = level_nodes.to(device=abs_node_logits.device, dtype=torch.long)
        level_size = int(level_nodes.numel())
        if level_size <= 0:
            raise ValueError(f"Hier-COS level {level} has no node ids.")

        level_targets = leaf_to_level_local[leaf_targets, level].to(device=abs_node_logits.device, dtype=torch.long)
        _check_index_range(
            level_targets,
            level_size,
            f"Hier-COS invalid target labels for CE at level {level}: expected [0, {level_size}).",
        )

        if global_log_probs is not None:
            target_log_probs = global_log_probs[row_ids, level_nodes[level_targets]]
        else:
            level_logits = abs_node_logits.index_select(dim=1, index=level_nodes)
            level_log_probs = F.log_softmax(level_logits, dim=1)
            target_log_probs = level_log_probs[row_ids, level_targets]
        level_losses.append(-(weights[level] * target_log_probs).mean())
    return level_losses


def _weighted_target_ce_level_losses_from_level_logits(
    abs_node_logits_per_level: List[torch.Tensor],
    leaf_targets: torch.Tensor,
    leaf_to_level_local: torch.Tensor,
    level_weights: torch.Tensor,
) -> List[torch.Tensor]:
    weights = level_weights / level_weights.sum().clamp_min(_EPS)
    level_losses: List[torch.Tensor] = []
    for level, level_logits in enumerate(abs_node_logits_per_level):
        level_size = int(level_logits.size(1))
        level_targets = leaf_to_level_local[leaf_targets, level].to(device=level_logits.device, dtype=torch.long)
        _check_index_range(
            level_targets,
            level_size,
            f"Hier-COS invalid target labels for CE at level {level}: expected [0, {level_size}).",
        )
        row_ids = torch.arange(int(leaf_targets.size(0)), device=level_logits.device, dtype=torch.long)
        level_log_probs = F.log_softmax(level_logits, dim=1)
        target_log_probs = level_log_probs[row_ids, level_targets]
        level_weight = weights[level].to(device=level_logits.device, dtype=level_logits.dtype)
        level_losses.append(-(level_weight * target_log_probs).mean())
    return level_losses


def _shared_level_weights(
    output: Dict[str, Any],
    cfg: Any,
    level_node_ids: List[torch.Tensor],
    num_levels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mode = _resolve_weight_mode(cfg)
    if num_levels <= 0:
        raise ValueError("Hier-COS expected at least one hierarchy level to build weights.")
    if mode == "equal":
        return torch.full((num_levels,), 1.0 / float(num_levels), device=device, dtype=dtype)

    if mode in {"cumulative_branching", "marginal_branching"}:
        if len(level_node_ids) != int(num_levels):
            raise ValueError(
                "Hier-COS branching weights require level node ids aligned with hierarchy depth."
            )
        class_counts = torch.tensor(
            [int(node_ids.numel()) for node_ids in level_node_ids],
            dtype=torch.float64,
        )
        if bool((class_counts <= 0).any()):
            raise ValueError("Hier-COS branching weights require at least one class at every level.")

        if mode == "cumulative_branching":
            beta = _resolve_weight_beta(cfg)
            # Softmax(beta * log(C_l)) is C_l**beta / sum_k C_k**beta,
            # while remaining stable for large finite beta values.
            log_counts = torch.log(class_counts)
            scaled_log_counts = float(beta) * (log_counts - log_counts.max())
            weights = torch.softmax(scaled_log_counts, dim=0)
        else:
            # The coarse level is the reference unit. Each later score is the
            # average expansion relative to the immediately preceding level.
            scores = torch.ones_like(class_counts)
            scores[1:] = class_counts[1:] / class_counts[:-1]
            weights = scores / scores.sum().clamp_min(_EPS)
        return weights.to(device=device, dtype=dtype)

    node_prob_weights = _get_tensor(
        output,
        "node_prob_weights",
    )
    if not isinstance(node_prob_weights, torch.Tensor) or node_prob_weights.ndim != 1:
        raise ValueError(
            "Hier-COS weight modes 'kl_leaf' and 'kl_coarse' require "
            "`node_prob_weights` with shape [depth]."
        )
    if int(node_prob_weights.numel()) != int(num_levels):
        raise ValueError(
            "Hier-COS `node_prob_weights` depth mismatch for level weighting: "
            f"expected {num_levels}, found {int(node_prob_weights.numel())}."
        )

    weights = node_prob_weights.to(device=device, dtype=dtype)
    if bool((weights < 0).any()):
        raise ValueError("Hier-COS weighting requires non-negative `node_prob_weights`.")
    return torch.flip(weights, dims=[0]) if mode == "kl_coarse" else weights


def resolve_level_weights(
    output: Dict[str, Any],
    cfg: Any,
    level_node_ids: List[torch.Tensor],
    num_levels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Public entry point for `model.weight_mode`.

    Direct subspace supervision reuses this so its level weighting is the one
    the native objective would apply, rather than a second, independent setting.
    """
    return _shared_level_weights(
        output=output,
        cfg=cfg,
        level_node_ids=level_node_ids,
        num_levels=num_levels,
        device=device,
        dtype=dtype,
    )


def compute_loss(
    output: Dict[str, Any],
    targets: HierCosTargets,
    cfg: Any,
    _taxonomy: Optional[Dict[str, Any]] = None,
    return_aux: bool = False,
) -> Union[
    Tuple[torch.Tensor, Dict[str, float]],
    Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]],
]:
    if isinstance(targets, dict) and targets.get("soft_targets_per_level") is not None:
        raise ValueError("Hier-COS does not support mixup/cutmix soft targets. Disable mixup/cutmix.")

    logits_per_level = output.get("logits_per_level")
    if not isinstance(logits_per_level, list) or not logits_per_level:
        raise ValueError("Hier-COS output must contain non-empty `logits_per_level`.")

    num_levels = len(logits_per_level)
    loss_mode = _resolve_loss_mode(cfg)
    hard_targets = _hard_targets_from_input(targets, num_levels=num_levels).to(
        device=logits_per_level[-1].device,
        dtype=torch.long,
    )

    node_logits, level_node_ids, leaf_to_level_local = _validate_output(output=output, num_levels=num_levels)
    node_logits_per_level = _validate_node_logits_per_level(
        output=output,
        level_node_ids=level_node_ids,
        num_levels=num_levels,
    )
    effective_node_logits_per_level = _validate_effective_node_logits_per_level(
        output=output,
        level_node_ids=level_node_ids,
        num_levels=num_levels,
    )
    if effective_node_logits_per_level is not None:
        loss_node_logits_per_level = effective_node_logits_per_level
        loss_node_logits = torch.cat(effective_node_logits_per_level, dim=1)
    else:
        # Preserve the existing no-HCC path exactly: global losses consume the
        # original tensor, and level losses retain their original per-level
        # representation (or None for a dense/global fixed frame).
        loss_node_logits_per_level = node_logits_per_level
        loss_node_logits = node_logits

    hard_targets = hard_targets.to(device=loss_node_logits.device, dtype=torch.long)
    leaf_targets = _resolve_leaf_targets(hard_targets)
    num_leaf = int(leaf_to_level_local.size(0))
    _check_index_range(
        leaf_targets,
        num_leaf,
        f"Hier-COS leaf targets out of range [0, {num_leaf}). "
        "Ensure the finest target level matches dataset leaf ids.",
    )

    leaf_to_level_local = leaf_to_level_local.to(device=loss_node_logits.device, dtype=torch.long)
    level_weights = _shared_level_weights(
        output=output,
        cfg=cfg,
        level_node_ids=level_node_ids,
        num_levels=num_levels,
        device=loss_node_logits.device,
        dtype=loss_node_logits.dtype,
    )
    weight_metrics = {
        f"loss_weight_level_{level}": _to_scalar(level_weight)
        for level, level_weight in enumerate(level_weights)
    }
    abs_node_logits = loss_node_logits.abs()

    alpha = _resolve_alpha(cfg)
    if loss_mode in {"global_softmax_ce_reg", "level_softmax_ce_reg"}:
        if loss_mode == "level_softmax_ce_reg" and loss_node_logits_per_level is not None:
            abs_node_logits_per_level = [level_logits.abs() for level_logits in loss_node_logits_per_level]
            reg, level_reg_losses = _level_regularization_loss_from_level_logits(
                abs_node_logits_per_level=abs_node_logits_per_level,
                leaf_targets=leaf_targets,
                leaf_to_level_local=leaf_to_level_local,
            )
            ce_level_losses = _weighted_target_ce_level_losses_from_level_logits(
                abs_node_logits_per_level=abs_node_logits_per_level,
                leaf_targets=leaf_targets,
                leaf_to_level_local=leaf_to_level_local,
                level_weights=level_weights,
            )
        else:
            reg, level_reg_losses = _level_regularization_loss(
                abs_node_logits=abs_node_logits,
                leaf_targets=leaf_targets,
                level_node_ids=level_node_ids,
                leaf_to_level_local=leaf_to_level_local,
            )
            ce_level_losses = _weighted_target_ce_level_losses(
                abs_node_logits=abs_node_logits,
                level_node_ids=level_node_ids,
                leaf_targets=leaf_targets,
                leaf_to_level_local=leaf_to_level_local,
                level_weights=level_weights,
                softmax_scope="global" if loss_mode == "global_softmax_ce_reg" else "level",
            )
        ce = torch.stack(ce_level_losses).sum()
        level_losses = [
            ce_level_loss + float(alpha) * reg_level_loss
            for ce_level_loss, reg_level_loss in zip(ce_level_losses, level_reg_losses)
        ]
        total = torch.stack(level_losses).sum()
        metrics = {
            "total": _to_scalar(total),
            "ce": _to_scalar(ce),
            "reg": _to_scalar(reg),
            **weight_metrics,
        }
        for level, (ce_level_loss, reg_level_loss, level_loss) in enumerate(
            zip(ce_level_losses, level_reg_losses, level_losses)
        ):
            metrics[f"ce_level_{level}"] = _to_scalar(ce_level_loss)
            metrics[f"reg_level_{level}"] = _to_scalar(reg_level_loss)
            metrics[f"loss_level_{level}"] = _to_scalar(level_loss)

        if not return_aux:
            return total, metrics
        return total, metrics, {
            "level_losses": level_losses,
            "ce_loss": ce,
            "reg_loss": reg,
            "ce_level_losses": ce_level_losses,
            "level_reg_losses": level_reg_losses,
        }

    reg, level_reg_losses = _level_regularization_loss(
        abs_node_logits=abs_node_logits,
        leaf_targets=leaf_targets,
        level_node_ids=level_node_ids,
        leaf_to_level_local=leaf_to_level_local,
    )
    node_targets = _build_node_targets(
        leaf_targets=leaf_targets,
        level_node_ids=level_node_ids,
        leaf_to_level_local=leaf_to_level_local,
        level_weights=level_weights,
        total_nodes=int(loss_node_logits.size(1)),
        dtype=loss_node_logits.dtype,
    )
    log_probs = F.log_softmax(abs_node_logits, dim=1)
    kl = F.kl_div(log_probs, node_targets, reduction="batchmean")
    total = kl + float(alpha) * reg
    metrics = {
        "total": _to_scalar(total),
        "kl": _to_scalar(kl),
        "reg": _to_scalar(reg),
        **weight_metrics,
    }

    if not return_aux:
        return total, metrics
    return total, metrics, {"kl_loss": kl, "reg_loss": reg, "level_reg_losses": level_reg_losses}
