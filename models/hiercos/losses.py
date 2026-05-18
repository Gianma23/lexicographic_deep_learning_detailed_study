from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


HierCosTargets = Union[torch.Tensor, Dict[str, Any]]
_EPS = 1e-12


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
        raise ValueError(f"Hier-COS expected hard targets with shape [B, {num_levels}], got {tuple(hard_targets.shape)}.")
    return hard_targets.long()


def _resolve_model_alpha(cfg: Any, default: float = 1.0) -> float:
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None or not hasattr(model_cfg, "get"):
        return float(default)
    return float(model_cfg.get("alpha", default))


def _resolve_loss_mode(cfg: Any, default: str = "kl_reg") -> str:
    model_cfg = getattr(cfg, "model", None)
    raw_mode = default
    if model_cfg is not None and hasattr(model_cfg, "get"):
        configured = model_cfg.get("loss", default)
        if configured is not None:
            raw_mode = configured

    if not isinstance(raw_mode, str):
        raise ValueError(
            "Hier-COS `model.loss` must be a scalar string: one of ['kl_reg', 'per_level_ce']."
        )

    mode = raw_mode.strip().lower().replace("-", "_")
    if mode in {"kl_reg", "klreg"}:
        return "kl_reg"
    if mode in {"per_level_ce", "perlevel_ce", "level_ce"}:
        return "per_level_ce"
    raise ValueError(
        f"Unsupported Hier-COS model.loss '{raw_mode}'. Expected one of ['kl_reg', 'per_level_ce']."
    )


def _resolve_ce_weight_mode(cfg: Any, default: str = "equal") -> str:
    model_cfg = getattr(cfg, "model", None)
    raw_mode = default
    if model_cfg is not None and hasattr(model_cfg, "get"):
        configured = model_cfg.get("ce_weight_mode", default)
        if configured is not None:
            raw_mode = configured

    if not isinstance(raw_mode, str):
        raise ValueError(
            "Hier-COS `model.ce_weight_mode` must be a scalar string: "
            "one of ['equal', 'kl_leaf', 'kl_coarse']."
        )

    mode = raw_mode.strip().lower().replace("-", "_")
    if mode in {"equal", "uniform", "all_equal"}:
        return "equal"
    if mode in {"kl_leaf", "leaf", "leaf_heavy"}:
        return "kl_leaf"
    if mode in {"kl_coarse", "coarse", "coarse_heavy", "reverse_kl"}:
        return "kl_coarse"
    raise ValueError(
        f"Unsupported Hier-COS model.ce_weight_mode '{raw_mode}'. "
        "Expected one of ['equal', 'kl_leaf', 'kl_coarse']."
    )


def _label_smoothing_from_cfg(cfg: Any) -> float:
    train_cfg = getattr(cfg, "train", None)
    if train_cfg is None or not hasattr(train_cfg, "get"):
        return 0.0
    return min(max(float(train_cfg.get("smoothing", 0.0)), 0.0), 1.0)


def _resolve_leaf_targets(hard_targets: torch.Tensor) -> torch.Tensor:
    if hard_targets.numel() == 0:
        return hard_targets.new_zeros((hard_targets.size(0),), dtype=torch.long)
    return hard_targets[:, -1].long()


def _validate_hiercos_output(output: Dict[str, Any], num_levels: int) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    node_logits = output.get("node_logits")
    if not isinstance(node_logits, torch.Tensor) or node_logits.ndim != 2:
        raise ValueError("Hier-COS output must provide `node_logits` with shape [B, N].")

    level_node_ids = output.get("hiercos_level_node_ids")
    if not isinstance(level_node_ids, list) or len(level_node_ids) != num_levels:
        raise ValueError("Hier-COS output must provide `hiercos_level_node_ids` aligned with hierarchy depth.")
    if not all(isinstance(node_ids, torch.Tensor) and node_ids.ndim == 1 for node_ids in level_node_ids):
        raise ValueError("Hier-COS `hiercos_level_node_ids` entries must be 1D tensors.")

    leaf_to_level_local = output.get("leaf_to_level_local")
    if not isinstance(leaf_to_level_local, torch.Tensor) or leaf_to_level_local.ndim != 2:
        raise ValueError("Hier-COS output must provide `leaf_to_level_local` with shape [num_leaf, depth].")
    if int(leaf_to_level_local.size(1)) != num_levels:
        raise ValueError(
            "Hier-COS `leaf_to_level_local` depth mismatch: "
            f"expected {num_levels}, found {int(leaf_to_level_local.size(1))}."
        )

    node_prob_weights = output.get("node_prob_weights")
    if not isinstance(node_prob_weights, torch.Tensor) or node_prob_weights.ndim != 1:
        raise ValueError("Hier-COS output must provide `node_prob_weights` with shape [depth].")
    if int(node_prob_weights.numel()) != num_levels:
        raise ValueError(
            "Hier-COS `node_prob_weights` depth mismatch: "
            f"expected {num_levels}, found {int(node_prob_weights.numel())}."
        )

    return node_logits, level_node_ids, leaf_to_level_local, node_prob_weights


def _build_node_targets(
    leaf_targets: torch.Tensor,
    level_node_ids: List[torch.Tensor],
    leaf_to_level_local: torch.Tensor,
    node_prob_weights: torch.Tensor,
    total_nodes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = int(leaf_targets.size(0))
    device = leaf_targets.device
    targets = torch.zeros((batch_size, total_nodes), device=device, dtype=dtype)
    row_ids = torch.arange(batch_size, device=device, dtype=torch.long)

    for level, level_nodes in enumerate(level_node_ids):
        level_nodes = level_nodes.to(device=device, dtype=torch.long)
        level_size = int(level_nodes.numel())

        local_ids = leaf_to_level_local[leaf_targets, level].to(device=device, dtype=torch.long)
        if bool((local_ids < 0).any()) or bool((local_ids >= level_size).any()):
            raise ValueError(f"Hier-COS invalid local node indices for level {level}.")

        global_ids = level_nodes[local_ids]
        targets[row_ids, global_ids] = node_prob_weights[level]

    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(_EPS)
    return targets


def _level_regularization_loss(
    node_logits: torch.Tensor,
    leaf_targets: torch.Tensor,
    level_node_ids: List[torch.Tensor],
    leaf_to_level_local: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    abs_logits = node_logits.abs()
    level_losses: List[torch.Tensor] = []

    for level, level_nodes in enumerate(level_node_ids):
        level_nodes = level_nodes.to(device=node_logits.device, dtype=torch.long)
        level_size = int(level_nodes.numel())

        level_logits = abs_logits.index_select(dim=1, index=level_nodes)
        level_logits = level_logits / level_logits.norm(dim=1, keepdim=True).clamp_min(_EPS)

        level_targets = leaf_to_level_local[leaf_targets, level].to(device=node_logits.device, dtype=torch.long)
        if bool((level_targets < 0).any()) or bool((level_targets >= level_size).any()):
            raise ValueError(f"Hier-COS invalid target labels for regularization at level {level}.")

        one_hot = F.one_hot(level_targets, num_classes=level_size).to(dtype=level_logits.dtype)
        level_losses.append((one_hot - level_logits).abs().sum(dim=1).mean())

    reg = torch.stack(level_losses).sum() if level_losses else node_logits.new_zeros(())
    return reg, level_losses


def _per_level_ce_losses(
    logits_per_level: List[torch.Tensor],
    hard_targets: torch.Tensor,
    smoothing: float,
) -> List[torch.Tensor]:
    level_losses: List[torch.Tensor] = []
    for level, logits in enumerate(logits_per_level):
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError(f"Hier-COS logits for level {level} must have shape [B, C].")
        level_targets = hard_targets[:, level].to(device=logits.device, dtype=torch.long)
        level_losses.append(
            F.cross_entropy(
                logits,
                level_targets,
                label_smoothing=float(smoothing),
            )
        )
    return level_losses


def _ce_level_weights(
    output: Dict[str, Any],
    cfg: Any,
    num_levels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mode = _resolve_ce_weight_mode(cfg)
    if mode == "equal":
        return torch.ones((num_levels,), device=device, dtype=dtype)

    node_prob_weights = output.get("node_prob_weights")
    if not isinstance(node_prob_weights, torch.Tensor) or node_prob_weights.ndim != 1:
        raise ValueError(
            "Hier-COS CE weight modes 'kl_leaf' and 'kl_coarse' require "
            "`node_prob_weights` with shape [depth]."
        )
    if int(node_prob_weights.numel()) != int(num_levels):
        raise ValueError(
            "Hier-COS `node_prob_weights` depth mismatch for CE weighting: "
            f"expected {num_levels}, found {int(node_prob_weights.numel())}."
        )

    weights = node_prob_weights.to(device=device, dtype=dtype)
    if bool((weights < 0).any()):
        raise ValueError("Hier-COS CE weighting requires non-negative `node_prob_weights`.")
    if mode == "kl_coarse":
        weights = torch.flip(weights, dims=[0])

    weight_sum = weights.sum().clamp_min(_EPS)
    return weights / weight_sum * float(num_levels)


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

    hard_targets = _hard_targets_from_input(targets, num_levels=num_levels)
    hard_targets = hard_targets.to(device=logits_per_level[-1].device, dtype=torch.long)

    if loss_mode == "per_level_ce":
        raw_level_losses = _per_level_ce_losses(
            logits_per_level=logits_per_level,
            hard_targets=hard_targets,
            smoothing=_label_smoothing_from_cfg(cfg),
        )
        weights = _ce_level_weights(
            output=output,
            cfg=cfg,
            num_levels=num_levels,
            device=raw_level_losses[0].device,
            dtype=raw_level_losses[0].dtype,
        )
        level_losses = [
            level_loss * weights[level]
            for level, level_loss in enumerate(raw_level_losses)
        ]
        total = torch.stack(level_losses).sum()
        metrics = {
            "total": float(total.detach().item()),
            "level_ce": float(total.detach().item()),
        }
        for level, level_loss in enumerate(level_losses):
            metrics[f"loss_level_{level}"] = float(level_loss.detach().item())
        if not return_aux:
            return total, metrics
        return total, metrics, {"level_losses": level_losses}

    node_logits, level_node_ids, leaf_to_level_local, node_prob_weights = _validate_hiercos_output(
        output,
        num_levels=num_levels,
    )

    hard_targets = hard_targets.to(device=node_logits.device, dtype=torch.long)
    leaf_targets = _resolve_leaf_targets(hard_targets)
    num_leaf = int(leaf_to_level_local.size(0))
    if bool((leaf_targets < 0).any()) or bool((leaf_targets >= num_leaf).any()):
        raise ValueError(
            f"Hier-COS leaf targets out of range [0, {num_leaf}). "
            "Ensure the finest target level matches dataset leaf ids."
        )

    node_prob_weights = node_prob_weights.to(device=node_logits.device, dtype=node_logits.dtype)
    node_targets = _build_node_targets(
        leaf_targets=leaf_targets,
        level_node_ids=level_node_ids,
        leaf_to_level_local=leaf_to_level_local.to(device=node_logits.device, dtype=torch.long),
        node_prob_weights=node_prob_weights,
        total_nodes=int(node_logits.size(1)),
        dtype=node_logits.dtype,
    )

    kl = F.kl_div(F.log_softmax(node_logits.abs(), dim=1), node_targets, reduction="batchmean")
    reg, level_reg_losses = _level_regularization_loss(
        node_logits=node_logits,
        leaf_targets=leaf_targets,
        level_node_ids=level_node_ids,
        leaf_to_level_local=leaf_to_level_local.to(device=node_logits.device, dtype=torch.long),
    )

    alpha = _resolve_model_alpha(cfg, default=1.0)
    total = kl + float(alpha) * reg

    metrics = {
        "total": float(total.detach().item()),
        "kl": float(kl.detach().item()),
        "reg": float(reg.detach().item()),
    }
    if not return_aux:
        return total, metrics

    aux_payload: Dict[str, Any] = {
        "kl_loss": kl,
        "reg_loss": reg,
        "level_reg_losses": level_reg_losses,
    }
    return total, metrics, aux_payload
