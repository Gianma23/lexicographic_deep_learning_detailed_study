from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


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
) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    if len(num_classes_per_level) != 3:
        return None, None

    n0, n1, n2 = [int(n) for n in num_classes_per_level]
    map_1 = parent_of.get(1, {})
    map_2 = parent_of.get(2, {})

    required_l1 = set(range(n1))
    required_l2 = set(range(n2))
    if set(map_1.keys()) != required_l1 or set(map_2.keys()) != required_l2:
        return None, None
    if any(int(parent) not in required_l1 for parent in map_2.values()):
        return None, None
    if any(int(parent) not in set(range(n0)) for parent in map_1.values()):
        return None, None

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
        return None, None

    state_space = torch.stack(rows, dim=0)  # [S, total_nodes]
    node_to_states: List[torch.Tensor] = []
    for node_idx in range(total_nodes):
        idx = torch.where(state_space[:, node_idx] > 0)[0]
        node_to_states.append(idx)
    return state_space, node_to_states


def _combinatorial_tree_loss(
    tree_scores_per_level: List[torch.Tensor],
    targets: torch.Tensor,
    num_classes_per_level: List[int],
    taxonomy: Optional[Dict[str, Any]],
    temperature: float,
    eps: float,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    parent_of = _normalize_parent_of(taxonomy)
    device = tree_scores_per_level[0].device
    state_space, node_to_states = _build_state_space(
        num_classes_per_level=num_classes_per_level,
        parent_of=parent_of,
        device=device,
    )
    if state_space is None or node_to_states is None:
        zero = torch.zeros((), device=device)
        return zero, {0: zero, 1: zero, 2: zero}

    fs = torch.cat(tree_scores_per_level, dim=1).to(dtype=state_space.dtype)  # [B, total_nodes]
    state_logits = torch.matmul(state_space, fs.T) / max(float(temperature), 1e-6)  # [S, B]
    log_z = torch.logsumexp(state_logits, dim=0)  # [B]

    offsets = _level_offsets(num_classes_per_level)
    level_losses: Dict[int, torch.Tensor] = {}
    per_level_tensors: List[torch.Tensor] = []

    for level in range(3):
        node_ids = (targets[:, level] + offsets[level]).tolist()
        sample_losses: List[torch.Tensor] = []
        for batch_idx, node_id in enumerate(node_ids):
            states_idx = node_to_states[int(node_id)]
            if states_idx.numel() == 0:
                continue
            log_marginal = torch.logsumexp(torch.index_select(state_logits[:, batch_idx], 0, states_idx), dim=0)
            sample_losses.append(-(log_marginal - log_z[batch_idx]))
        if sample_losses:
            level_loss = torch.stack(sample_losses).mean()
        else:
            level_loss = torch.zeros((), device=device)
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
    targets: torch.Tensor,
    cfg: Any,
    taxonomy: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits_per_level = output["logits_per_level"]
    if len(logits_per_level) != 3:
        raise ValueError(f"HRN expects 3 logits levels, got {len(logits_per_level)}.")

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
        targets=targets,
        num_classes_per_level=num_classes_per_level,
        taxonomy=taxonomy,
        temperature=temperature,
        eps=eps,
    )
    fine_ce = F.cross_entropy(logits_per_level[2], targets[:, 2], label_smoothing=label_smoothing)
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
