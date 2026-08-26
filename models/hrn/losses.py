from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ..common.hcc import select_effective_logits

HrnTargets = Union[torch.Tensor, Dict[str, Any]]


def _hard_targets_from_input(targets: HrnTargets) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        return targets
    if not isinstance(targets, dict):
        raise TypeError("HRN targets must be a tensor or dict.")

    hard_targets = targets.get("hard_targets")
    if isinstance(hard_targets, torch.Tensor):
        return hard_targets

    labels_a = targets.get("labels_a")
    if isinstance(labels_a, torch.Tensor):
        return labels_a

    raise TypeError("HRN expected `hard_targets` (or legacy `labels_a`) tensor in target dict.")


def _resolve_loss_mode(cfg: Any) -> str:
    model_cfg = getattr(cfg, "model", None)
    raw_mode = model_cfg.get("loss", "native") if hasattr(model_cfg, "get") else "native"
    if raw_mode is None:
        raw_mode = "native"
    if not isinstance(raw_mode, str) or raw_mode not in {"native", "level_conditional"}:
        raise ValueError("HRN model.loss must be one of ['native', 'level_conditional'].")
    return raw_mode


def _normalize_parent_of(taxonomy: Optional[Dict[str, Any]]) -> Dict[int, Dict[int, int]]:
    if not taxonomy or "parent_of" not in taxonomy:
        return {}
    normalized: Dict[int, Dict[int, int]] = {}
    for level_key, mapping in taxonomy["parent_of"].items():
        level = int(level_key)
        normalized[level] = {int(child): int(parent) for child, parent in mapping.items()}
    return normalized


def _level_offsets(num_classes_per_level: List[int]) -> List[int]:
    offsets = [0]
    for n in num_classes_per_level[:-1]:
        offsets.append(offsets[-1] + int(n))
    return offsets


def _build_state_space(
    num_classes_per_level: List[int],
    parent_of: Dict[int, Dict[int, int]],
    device: torch.device,
    dtype: torch.dtype,
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
    if any(parent not in required_l1 for parent in map_2.values()):
        return None
    if any(parent not in set(range(n0)) for parent in map_1.values()):
        return None

    total_nodes = n0 + n1 + n2
    off0, off1, off2 = _level_offsets(num_classes_per_level)

    # Upstream `TreeLoss` includes a leading all-zero row.
    rows: List[torch.Tensor] = [torch.zeros(total_nodes, device=device, dtype=dtype)]

    recorded_l0 = [False] * n0
    recorded_l1 = [False] * n1
    for leaf in range(n2):
        l1 = int(map_2[leaf])
        l0 = int(map_1[l1])

        if not recorded_l0[l0]:
            row0 = torch.zeros(total_nodes, device=device, dtype=dtype)
            row0[off0 + l0] = 1.0
            rows.append(row0)
            recorded_l0[l0] = True

        if not recorded_l1[l1]:
            row1 = torch.zeros(total_nodes, device=device, dtype=dtype)
            row1[off0 + l0] = 1.0
            row1[off1 + l1] = 1.0
            rows.append(row1)
            recorded_l1[l1] = True

        row2 = torch.zeros(total_nodes, device=device, dtype=dtype)
        row2[off0 + l0] = 1.0
        row2[off1 + l1] = 1.0
        row2[off2 + leaf] = 1.0
        rows.append(row2)

    return torch.stack(rows, dim=0)


def _resolve_observed_indices(
    hard_targets: torch.Tensor,
    num_classes_per_level: List[int],
    targets: HrnTargets,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(hard_targets.size(0))
    device = hard_targets.device
    offsets = _level_offsets(num_classes_per_level)

    if isinstance(targets, dict):
        observed_global = targets.get("observed_global_index")
        if isinstance(observed_global, torch.Tensor):
            observed_global = observed_global.to(device=device, dtype=torch.long).view(-1)
            if observed_global.numel() != batch_size:
                raise ValueError("`observed_global_index` must have one entry per sample.")
            observed_level = targets.get("observed_level")
            if isinstance(observed_level, torch.Tensor):
                observed_level = observed_level.to(device=device, dtype=torch.long).view(-1)
                if observed_level.numel() != batch_size:
                    raise ValueError("`observed_level` must have one entry per sample.")
            else:
                # Backward-compatible default for current full-path targets.
                observed_level = torch.full((batch_size,), 2, device=device, dtype=torch.long)
            return observed_global, observed_level

        observed_level = targets.get("observed_level")
        if isinstance(observed_level, torch.Tensor):
            observed_level = observed_level.to(device=device, dtype=torch.long).view(-1)
            if observed_level.numel() != batch_size:
                raise ValueError("`observed_level` must have one entry per sample.")
            gathered = hard_targets.gather(1, observed_level.unsqueeze(1)).squeeze(1).long()
            observed_global = gathered + torch.tensor(offsets, device=device, dtype=torch.long)[observed_level]
            return observed_global, observed_level

    # Default behavior: treat current labels as leaf-observed samples.
    observed_level = torch.full((batch_size,), 2, device=device, dtype=torch.long)
    observed_global = hard_targets[:, 2].long() + int(offsets[2])
    return observed_global, observed_level


def _hierarchical_loss(tree_scores_per_level: List[torch.Tensor], observed_global: torch.Tensor, state_space: torch.Tensor) -> torch.Tensor:
    fs = torch.cat(tree_scores_per_level, dim=1).to(dtype=state_space.dtype)  # [B, N]
    joint = torch.exp(torch.matmul(state_space, fs.T))  # [S, B]
    z = joint.sum(dim=0).clamp_min(1e-12)  # [B]
    eligible = state_space[:, observed_global.long()] > 0  # [S, B]
    marginal = (joint * eligible.to(dtype=joint.dtype)).sum(dim=0).clamp_min(1e-12)
    return (-torch.log(marginal / z)).to(dtype=torch.float64).mean()


def _level_conditional_tree_losses(
    tree_scores_per_level: List[torch.Tensor],
    hard_targets: torch.Tensor,
    state_space: torch.Tensor,
    fine_tree_loss: torch.Tensor,
) -> List[torch.Tensor]:
    """Split the native tree loss into three conditional negative log-likelihoods.

    `_hierarchical_loss` returns `-log P(state in subtree(u))` for an observed
    node `u`. The observed nodes of one sample are nested,
    `subtree(coarse) ⊇ subtree(middle) ⊇ {leaf}`, so the chain rule gives

        -log P(leaf) = -log P(sub(coarse))
                       - log P(sub(middle)) / P(sub(coarse))
                       - log P(leaf)        / P(sub(middle)),

    that is, the successive differences of the three marginal terms. The three
    returned tensors therefore sum exactly to `fine_tree_loss`: with projection
    disabled the decomposed mode reproduces the `native` objective, gradients
    included, and each term is a proper conditional NLL of its own level.

    Each marginal is evaluated with the levels finer than its own detached. They
    enter that term only through the state-space normaliser and through the deep
    states inside the observed subtree, so without the detach every term reads
    every level's scores and `-log P(sub(coarse))` carries gradient into the fine
    head at the same order as into its own. Detaching leaves all three values
    untouched, so the telescoping identity and the native-gradient equality both
    survive, and makes the level gradients triangular: term `l` is supported on
    the parameters of levels `0..l` only. That is what gives lexicographic
    ordering a well-defined subject -- without it `grad(l_0)` spans every branch,
    including a fine head that cannot move the coarse logits at all.

    The detach acts on the gradient, not on the value: the normaliser still sums
    over states of every depth, so a fine-only step still moves the coarse term's
    value. The guarantee is that `l_0` is stationary along its own coordinates,
    not that the coarse objective is unchanged.
    """
    offsets = _level_offsets([int(scores.size(-1)) for scores in tree_scores_per_level])

    def _marginal_at(level: int) -> torch.Tensor:
        detached_scores = [
            scores if score_level <= level else scores.detach()
            for score_level, scores in enumerate(tree_scores_per_level)
        ]
        return _hierarchical_loss(
            tree_scores_per_level=detached_scores,
            observed_global=hard_targets[:, level].long() + int(offsets[level]),
            state_space=state_space,
        )

    # `fine_tree_loss` is the deepest marginal, which has nothing to detach.
    marginal_losses = [_marginal_at(level) for level in range(2)]
    marginal_losses.append(fine_tree_loss)
    return [
        marginal_losses[0],
        marginal_losses[1] - marginal_losses[0],
        marginal_losses[2] - marginal_losses[1],
    ]


def compute_loss(
    output: Dict[str, Any],
    targets: HrnTargets,
    cfg: Any,
    taxonomy: Optional[Dict[str, Any]] = None,
    return_aux: bool = False,
) -> Union[
    Tuple[torch.Tensor, Dict[str, float]],
    Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]],
]:
    if isinstance(targets, dict):
        raise ValueError(
            "HRN paper-reproduction mode supports full leaf-labeled samples only; "
            "partial/relabeling targets and mixup/cutmix target dictionaries are unsupported."
        )

    loss_mode = _resolve_loss_mode(cfg)
    hard_targets = _hard_targets_from_input(targets)
    if hard_targets.ndim != 2 or int(hard_targets.size(1)) != 3:
        raise ValueError(f"HRN expects hard targets of shape [B, 3], got {tuple(hard_targets.shape)}.")

    logits_per_level = output.get("logits_per_level")
    has_logits_per_level = isinstance(logits_per_level, list) and len(logits_per_level) == 3
    if logits_per_level is not None and not has_logits_per_level:
        raise ValueError("HRN output `logits_per_level` must be a 3-level list when provided.")

    raw_tree_scores_per_level = output.get("tree_scores_per_level")
    if not isinstance(raw_tree_scores_per_level, list) or len(raw_tree_scores_per_level) != 3:
        raise ValueError("HRN output must provide `tree_scores_per_level` with 3 tensors.")

    # If effective_tree_scores_per_level is present, the hierarchical marginal
    # loss trains on the HCC-corrected coarse and middle activations; the fine
    # tree head is not part of the constrained triple and keeps its raw score.
    _, tree_scores_per_level = select_effective_logits(
        output,
        raw_key="tree_scores_per_level",
        effective_key="effective_tree_scores_per_level",
    )

    species_ce_logits = output.get("species_ce_logits")
    if not isinstance(species_ce_logits, torch.Tensor):
        if not has_logits_per_level:
            raise ValueError("HRN output must provide `species_ce_logits` or a 3-level `logits_per_level`.")
        species_ce_logits = logits_per_level[2]

    # HCC constrains the emitted triple, whose fine entry is this head, so the
    # leaf cross-entropy trains on the corrected logits whenever it is active.
    effective_logits_per_level = output.get("effective_logits_per_level")
    if isinstance(effective_logits_per_level, list) and len(effective_logits_per_level) == 3:
        species_ce_logits = effective_logits_per_level[2]

    num_classes_per_level = [int(x.size(-1)) for x in tree_scores_per_level]
    parent_of = _normalize_parent_of(taxonomy)
    state_space = _build_state_space(
        num_classes_per_level=num_classes_per_level,
        parent_of=parent_of,
        device=tree_scores_per_level[0].device,
        dtype=torch.float32,
    )
    if state_space is None:
        raise ValueError("HRN combinatorial loss requires a complete 3-level taxonomy (`parent_of` maps for levels 1 and 2).")

    observed_global, observed_level = _resolve_observed_indices(hard_targets=hard_targets, num_classes_per_level=num_classes_per_level, targets=targets)
    fine_tree_loss = _hierarchical_loss(
        tree_scores_per_level=tree_scores_per_level,
        observed_global=observed_global,
        state_space=state_space,
    )

    leaf_mask = observed_level == 2
    if bool(leaf_mask.any()):
        ce_loss = F.cross_entropy(
            species_ce_logits[leaf_mask].to(dtype=torch.float64),
            hard_targets[leaf_mask, 2].long(),
        )
    else:
        ce_loss = fine_tree_loss.new_zeros(())

    if loss_mode == "level_conditional":
        tree_level_losses = _level_conditional_tree_losses(
            tree_scores_per_level=tree_scores_per_level,
            hard_targets=hard_targets,
            state_space=state_space,
            fine_tree_loss=fine_tree_loss,
        )
        level_losses = [
            tree_level_losses[0],
            tree_level_losses[1],
            tree_level_losses[2] + ce_loss,
        ]
        # The conditional terms telescope, so `hier_loss` and `total` here are
        # numerically the `native` tree loss and the `native` total.
        hier_loss = torch.stack(tree_level_losses).sum()
        total = torch.stack(level_losses).sum()

        metrics = {
            "total": float(total.detach().item()),
            "hier_loss": float(hier_loss.detach().item()),
            "ce_loss_leaf": float(ce_loss.detach().item()),
            "tree_loss": float(hier_loss.detach().item()),
            "fine_ce": float(ce_loss.detach().item()),
        }
        for level, (tree_level_loss, level_loss) in enumerate(zip(tree_level_losses, level_losses)):
            metrics[f"tree_loss_level_{level}"] = float(tree_level_loss.detach().item())
            metrics[f"loss_level_{level}"] = float(level_loss.detach().item())

        if not return_aux:
            return total, metrics
        return total, metrics, {
            "observed_global_index": observed_global,
            "observed_level": observed_level,
            "leaf_mask": leaf_mask,
            "level_losses": level_losses,
            "tree_level_losses": tree_level_losses,
        }

    hier_loss = fine_tree_loss
    total = hier_loss + ce_loss

    metrics = {
        "total": float(total.detach().item()),
        "hier_loss": float(hier_loss.detach().item()),
        "ce_loss_leaf": float(ce_loss.detach().item()),
        # Backward-compatible aliases used by existing logs.
        "tree_loss": float(hier_loss.detach().item()),
        "fine_ce": float(ce_loss.detach().item()),
    }
    if not return_aux:
        return total, metrics

    aux_payload: Dict[str, Any] = {
        "observed_global_index": observed_global,
        "observed_level": observed_level,
        "leaf_mask": leaf_mask,
    }
    return total, metrics, aux_payload
