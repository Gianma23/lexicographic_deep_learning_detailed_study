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


def _resolve_model_alpha(cfg: Any, default: float = 0.1) -> float:
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None or not hasattr(model_cfg, "get"):
        return float(default)
    return float(model_cfg.get("alpha", default))


def _resolve_hafpp_loss_mode(cfg: Any, default: str = "leaf_only") -> str:
    model_cfg = getattr(cfg, "model", None)
    raw_mode = default
    if model_cfg is not None and hasattr(model_cfg, "get"):
        raw_mode = str(model_cfg.get("hafpp_loss_mode", default))
    mode = raw_mode.strip().lower().replace("-", "_")
    if mode in {"leaf_only", "leaf"}:
        return "leaf_only"
    if mode in {"full_node", "fullnode", "node"}:
        return "full_node"
    raise ValueError(
        f"Unsupported model.hafpp_loss_mode '{raw_mode}'. Expected one of ['leaf_only', 'full_node']."
    )


def _resolve_feature_space(cfg: Any) -> str:
    model_cfg = getattr(cfg, "model", None)
    raw = "hier-cos"
    if model_cfg is not None and hasattr(model_cfg, "get"):
        raw = str(model_cfg.get("feature_space", raw))
    mode = raw.strip().lower().replace("_", "-")
    if mode in {"hier-cos", "hiercos"}:
        return "hier-cos"
    if mode in {"haf++", "hafpp", "haf-plus-plus"}:
        return "haf++"
    raise ValueError(f"Unsupported Hier-COS feature_space '{raw}'. Expected 'hier-cos' or 'haf++'.")


def _resolve_leaf_targets(hard_targets: torch.Tensor) -> torch.Tensor:
    if hard_targets.numel() == 0:
        return hard_targets.new_zeros((hard_targets.size(0),), dtype=torch.long)
    return hard_targets[:, -1].long()


def _resolve_hafpp_full_node_targets(
    hard_targets: torch.Tensor,
    node_logits: torch.Tensor,
    level_node_ids: List[torch.Tensor],
    leaf_to_level_local: torch.Tensor,
) -> torch.Tensor:
    leaf_targets = _resolve_leaf_targets(hard_targets)
    num_leaf = int(leaf_to_level_local.size(0))
    if bool((leaf_targets < 0).any()) or bool((leaf_targets >= num_leaf).any()):
        raise ValueError(
            f"Hier-COS leaf targets out of range [0, {num_leaf}) for HAF++ full_node CE."
        )

    depth = int(leaf_to_level_local.size(1))
    leaf_to_level_local = leaf_to_level_local.to(device=node_logits.device, dtype=torch.long)
    finest_local = leaf_to_level_local[leaf_targets, depth - 1]

    finest_node_ids = level_node_ids[-1].to(device=node_logits.device, dtype=torch.long)
    if bool((finest_local < 0).any()) or bool((finest_local >= int(finest_node_ids.numel())).any()):
        raise ValueError("Hier-COS invalid finest-level local indices for HAF++ full_node CE.")

    global_targets = finest_node_ids[finest_local]
    num_nodes = int(node_logits.size(1))
    if bool((global_targets < 0).any()) or bool((global_targets >= num_nodes).any()):
        raise ValueError("Hier-COS invalid global node targets for HAF++ full_node CE.")
    return global_targets


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

    hard_targets = _hard_targets_from_input(targets, num_levels=num_levels)
    hard_targets = hard_targets.to(device=logits_per_level[-1].device, dtype=torch.long)
    feature_space = _resolve_feature_space(cfg)

    if feature_space == "haf++":
        hafpp_loss_mode = _resolve_hafpp_loss_mode(cfg, default="leaf_only")
        if hafpp_loss_mode == "leaf_only":
            leaf_logits = output.get("leaf_logits", logits_per_level[-1])
            if not isinstance(leaf_logits, torch.Tensor) or leaf_logits.ndim != 2:
                raise ValueError("Hier-COS HAF++ output must provide `leaf_logits` with shape [B, num_leaf].")
            leaf_targets = _resolve_leaf_targets(hard_targets).to(device=leaf_logits.device, dtype=torch.long)
            loss = F.cross_entropy(leaf_logits, leaf_targets)
            metrics = {
                "total": float(loss.detach().item()),
                "ce": float(loss.detach().item()),
            }
            if not return_aux:
                return loss, metrics
            return loss, metrics, {"level_losses": []}

        node_logits, level_node_ids, leaf_to_level_local, _node_prob_weights = _validate_hiercos_output(
            output,
            num_levels=num_levels,
        )
        hard_targets = hard_targets.to(device=node_logits.device, dtype=torch.long)
        global_targets = _resolve_hafpp_full_node_targets(
            hard_targets=hard_targets,
            node_logits=node_logits,
            level_node_ids=level_node_ids,
            leaf_to_level_local=leaf_to_level_local,
        )
        loss = F.cross_entropy(node_logits, global_targets)
        metrics = {
            "total": float(loss.detach().item()),
            "ce": float(loss.detach().item()),
            "ce_node": float(loss.detach().item()),
        }
        if not return_aux:
            return loss, metrics
        return loss, metrics, {"level_losses": [], "hafpp_global_targets": global_targets}

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

    alpha = _resolve_model_alpha(cfg, default=0.1)
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
