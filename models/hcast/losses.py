from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _normalize_parent_of(taxonomy: Optional[Dict[str, Any]]) -> Dict[int, Dict[int, int]]:
    if not taxonomy or "parent_of" not in taxonomy:
        return {}
    raw = taxonomy["parent_of"]
    out: Dict[int, Dict[int, int]] = {}
    for k, v in raw.items():
        lk = int(k)
        out[lk] = {int(ck): int(pk) for ck, pk in v.items()}
    return out


def _project_children_to_parent(child_probs: torch.Tensor, parent_dim: int, mapping: Dict[int, int]) -> torch.Tensor:
    proj = torch.zeros((child_probs.size(0), parent_dim), device=child_probs.device, dtype=child_probs.dtype)
    for child_idx, parent_idx in mapping.items():
        if child_idx < child_probs.size(-1) and parent_idx < parent_dim:
            proj[:, parent_idx] = proj[:, parent_idx] + child_probs[:, child_idx]
    return proj / proj.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _hierarchy_violation_loss(probs_per_level: List[torch.Tensor], parent_of: Dict[int, Dict[int, int]]) -> torch.Tensor:
    if not parent_of:
        return torch.zeros((), device=probs_per_level[0].device)

    penalties = []
    for level in range(1, len(probs_per_level)):
        mapping = parent_of.get(level)
        if not mapping:
            continue

        child_probs = probs_per_level[level]
        parent_probs = probs_per_level[level - 1]

        gather_parent = torch.zeros_like(child_probs)
        for child_idx, parent_idx in mapping.items():
            if child_idx < child_probs.size(-1) and parent_idx < parent_probs.size(-1):
                gather_parent[:, child_idx] = parent_probs[:, parent_idx]

        # Expected mismatch probability mass across invalid parent-child links.
        mismatch = 1.0 - (child_probs * gather_parent).sum(dim=-1)
        penalties.append(mismatch.mean())

    if not penalties:
        return torch.zeros((), device=probs_per_level[0].device)
    return torch.stack(penalties).mean()


def _tree_path_kl_loss(probs_per_level: List[torch.Tensor], parent_of: Dict[int, Dict[int, int]]) -> torch.Tensor:
    if not parent_of:
        return torch.zeros((), device=probs_per_level[0].device)

    kls = []
    for level in range(1, len(probs_per_level)):
        mapping = parent_of.get(level)
        if not mapping:
            continue

        child_probs = probs_per_level[level]
        parent_probs = probs_per_level[level - 1]
        projected = _project_children_to_parent(child_probs, parent_probs.size(-1), mapping)
        kls.append(F.kl_div(projected.log(), parent_probs, reduction="batchmean"))

    if not kls:
        return torch.zeros((), device=probs_per_level[0].device)
    return torch.stack(kls).mean()


def _global_kl_loss(logits_per_level: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    # Upstream H-CAST option (`globalkl`): KL between concatenated logits and one-hot targets.
    probs = [F.log_softmax(logits, dim=-1) for logits in logits_per_level]
    all_outputs = torch.cat(probs, dim=1)

    onehots = []
    for level, logits in enumerate(logits_per_level):
        onehots.append(F.one_hot(targets[:, level], num_classes=logits.size(-1)).float())
    all_targets = torch.cat(onehots, dim=1)
    all_targets = all_targets / all_targets.sum(dim=1, keepdim=True).clamp_min(1e-8)

    return F.kl_div(all_outputs, all_targets, reduction="batchmean")


def compute_loss(
    output: Dict[str, Any],
    targets: torch.Tensor,
    cfg: Any,
    taxonomy: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits_per_level = output["logits_per_level"]
    probs_per_level = output.get("probs_per_level") or [F.softmax(logits, dim=-1) for logits in logits_per_level]

    level_losses = [F.cross_entropy(logits, targets[:, level]) for level, logits in enumerate(logits_per_level)]
    ce_loss = torch.stack(level_losses).mean()

    parent_of = _normalize_parent_of(taxonomy)
    hv_loss = _hierarchy_violation_loss(probs_per_level, parent_of)
    tree_kl = _tree_path_kl_loss(probs_per_level, parent_of)

    hv_w = float(cfg.loss.get("beta_hv", 0.1))
    tree_kl_w = float(cfg.loss.get("alpha_tree_kl", 0.1))

    total = ce_loss + hv_w * hv_loss + tree_kl_w * tree_kl

    # Preserve upstream-style optional global KL term.
    if bool(cfg.loss.get("globalkl", False)):
        gk_w = float(cfg.loss.get("gk_weight", 1.0))
        gk_loss = _global_kl_loss(logits_per_level, targets)
        total = total + gk_w * gk_loss
    else:
        gk_loss = torch.zeros((), device=total.device)

    return total, {
        "total": float(total.detach().item()),
        "level_ce": float(ce_loss.detach().item()),
        "hv_loss": float(hv_loss.detach().item()),
        "tree_kl": float(tree_kl.detach().item()),
        "gk_loss": float(gk_loss.detach().item()),
    }
