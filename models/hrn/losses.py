from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


HrnTargets = Union[torch.Tensor, Dict[str, Any]]


def _soft_target_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    probs = target_probs.to(device=logits.device, dtype=logits.dtype)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


def _hard_targets_from_input(targets: HrnTargets) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        return targets
    labels_a = targets.get("labels_a")
    if not isinstance(labels_a, torch.Tensor):
        raise TypeError("Expected hard targets tensor or mixup target dict with `labels_a` tensor.")
    return labels_a


def _mixup_target_distributions(logits_per_level: List[torch.Tensor], targets: HrnTargets) -> Optional[List[torch.Tensor]]:
    if not isinstance(targets, dict):
        return None

    soft_targets = targets.get("soft_targets_per_level")
    if not isinstance(soft_targets, (list, tuple)) or len(soft_targets) != len(logits_per_level):
        return None

    out: List[torch.Tensor] = []
    for level, logits in enumerate(logits_per_level):
        target_level = soft_targets[level]
        if not isinstance(target_level, torch.Tensor):
            return None
        if target_level.ndim != 2 or int(target_level.size(1)) != int(logits.size(-1)):
            return None
        probs = target_level.to(device=logits.device, dtype=logits.dtype)
        out.append(probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return out


def _normalize_parent_of(taxonomy: Optional[Dict[str, Any]]) -> Dict[int, Dict[int, int]]:
    if not taxonomy or "parent_of" not in taxonomy:
        return {}
    out: Dict[int, Dict[int, int]] = {}
    for level_key, mapping in taxonomy["parent_of"].items():
        level = int(level_key)
        out[level] = {int(child): int(parent) for child, parent in mapping.items()}
    return out


def _level_offsets(num_classes_per_level: List[int]) -> List[int]:
    offsets = [0]
    for n in num_classes_per_level[:-1]:
        offsets.append(offsets[-1] + int(n))
    return offsets


def _build_state_space(
    num_classes_per_level: List[int],
    parent_of: Dict[int, Dict[int, int]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if len(num_classes_per_level) != 3:
        return None

    n0, n1, n2 = [int(n) for n in num_classes_per_level]
    map_1 = parent_of.get(1, {})
    map_2 = parent_of.get(2, {})

    required_l1 = set(range(n1))
    required_l2 = set(range(n2))
    if set(map_1.keys()) != required_l1 or set(map_2.keys()) != required_l2:
        return None
    if any(int(parent) not in required_l1 for parent in map_2.values()):
        return None
    if any(int(parent) not in set(range(n0)) for parent in map_1.values()):
        return None

    total_nodes = n0 + n1 + n2
    off0, off1, off2 = _level_offsets(num_classes_per_level)
    rows: List[torch.Tensor] = []

    recorded_l0 = [False] * n0
    recorded_l1 = [False] * n1

    for leaf in range(n2):
        l1 = int(map_2[leaf])
        l0 = int(map_1[l1])

        if not recorded_l0[l0]:
            row0 = torch.zeros(total_nodes, dtype=torch.float32, device=device)
            row0[off0 + l0] = 1.0
            rows.append(row0)
            recorded_l0[l0] = True

        if not recorded_l1[l1]:
            row1 = torch.zeros(total_nodes, dtype=torch.float32, device=device)
            row1[off0 + l0] = 1.0
            row1[off1 + l1] = 1.0
            rows.append(row1)
            recorded_l1[l1] = True

        row2 = torch.zeros(total_nodes, dtype=torch.float32, device=device)
        row2[off0 + l0] = 1.0
        row2[off1 + l1] = 1.0
        row2[off2 + leaf] = 1.0
        rows.append(row2)

    if not rows:
        return None

    state_space = torch.stack(rows, dim=0)  # [S, total_nodes]
    return state_space


def _combinatorial_tree_loss(
    tree_scores_per_level: List[torch.Tensor],
    targets: Optional[torch.Tensor],
    target_probs_per_level: Optional[List[torch.Tensor]],
    num_classes_per_level: List[int],
    taxonomy: Optional[Dict[str, Any]],
    temperature: float,
    eps: float,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    if targets is None and target_probs_per_level is None:
        raise ValueError("Expected either hard targets or soft target distributions.")

    parent_of = _normalize_parent_of(taxonomy)
    device = tree_scores_per_level[0].device
    state_space = _build_state_space(
        num_classes_per_level=num_classes_per_level,
        parent_of=parent_of,
        device=device,
    )
    if state_space is None:
        zero = torch.zeros((), device=device)
        return zero, {0: zero, 1: zero, 2: zero}

    fs = torch.cat(tree_scores_per_level, dim=1).to(dtype=state_space.dtype)  # [B, total_nodes]
    state_logits = torch.matmul(state_space, fs.T) / max(float(temperature), 1e-6)  # [S, B]
    # p(state) for each sample; summing these over states that include a node gives p(node).
    state_probs = torch.softmax(state_logits, dim=0)  # [S, B]
    node_probs = torch.matmul(state_space.T, state_probs).clamp_min(1e-12)  # [total_nodes, B]

    offsets = _level_offsets(num_classes_per_level)
    level_losses: Dict[int, torch.Tensor] = {}
    per_level_tensors: List[torch.Tensor] = []

    for level in range(3):
        start = offsets[level]
        end = start + int(num_classes_per_level[level])
        probs_level = node_probs[start:end].T  # [B, C_level]

        if target_probs_per_level is not None:
            target_probs = target_probs_per_level[level].to(device=device, dtype=state_logits.dtype)
            target_probs = target_probs / target_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
            level_loss = -(target_probs * torch.log(probs_level)).sum(dim=1).mean()
        else:
            idx = targets[:, level].long().clamp(min=0, max=int(num_classes_per_level[level]) - 1)
            picked = probs_level.gather(1, idx.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
            level_loss = -torch.log(picked).mean()

        level_losses[level] = level_loss
        per_level_tensors.append(level_loss)

    tree_loss = torch.stack(per_level_tensors).mean() if per_level_tensors else torch.zeros((), device=device)
    if not torch.isfinite(tree_loss):
        tree_loss = torch.zeros((), device=device)
        level_losses = {0: tree_loss, 1: tree_loss, 2: tree_loss}

    # Keep a tiny floor so log terms stay finite in pathological degenerate states.
    if eps > 0.0:
        tree_loss = tree_loss + 0.0 * eps
    return tree_loss, level_losses


def compute_loss(
    output: Dict[str, Any],
    targets: HrnTargets,
    cfg: Any,
    taxonomy: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits_per_level = output["logits_per_level"]
    if len(logits_per_level) != 3:
        raise ValueError(f"HRN expects 3 logits levels, got {len(logits_per_level)}.")
    mixup_target_probs = _mixup_target_distributions(logits_per_level, targets)
    hard_targets = _hard_targets_from_input(targets) if mixup_target_probs is None else None

    tree_scores_per_level = output.get("tree_scores_per_level")
    if tree_scores_per_level is None:
        tree_scores_per_level = [torch.sigmoid(logits) for logits in logits_per_level]

    num_classes_per_level = [int(logits.size(-1)) for logits in logits_per_level]
    temperature = float(cfg.loss.get("temperature", 1.0))
    eps = float(cfg.loss.get("eps", 1e-8))
    tree_weight = float(cfg.loss.get("tree_weight", 1.0))
    fine_ce_weight = float(cfg.loss.get("fine_ce_weight", 1.0))
    label_smoothing = float(cfg.train.get("smoothing", 0.0))

    tree_loss, tree_loss_levels = _combinatorial_tree_loss(
        tree_scores_per_level=tree_scores_per_level,
        targets=hard_targets,
        target_probs_per_level=mixup_target_probs,
        num_classes_per_level=num_classes_per_level,
        taxonomy=taxonomy,
        temperature=temperature,
        eps=eps,
    )
    if mixup_target_probs is not None:
        fine_ce = _soft_target_cross_entropy(logits_per_level[2], mixup_target_probs[2])
    else:
        fine_ce = F.cross_entropy(logits_per_level[2], hard_targets[:, 2], label_smoothing=label_smoothing)
    total = tree_weight * tree_loss + fine_ce_weight * fine_ce

    metrics = {
        "total": float(total.detach().item()),
        "tree_loss": float(tree_loss.detach().item()),
        "fine_ce": float(fine_ce.detach().item()),
        "loss_level_0": float(tree_loss_levels[0].detach().item()),
        "loss_level_1": float(tree_loss_levels[1].detach().item()),
        "loss_level_2": float(tree_loss_levels[2].detach().item()),
    }
    return total, metrics
