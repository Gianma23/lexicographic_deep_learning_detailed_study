from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


HcastTargets = Union[torch.Tensor, Dict[str, Any]]
_EPS = 1e-12


def _section_to_dict(section: Any) -> Dict[str, Any]:
    """Convert a config section into a plain dictionary when possible."""
    if isinstance(section, dict):
        return dict(section)
    if hasattr(section, "items"):
        return {k: v for k, v in section.items()}
    return {}


def _normalize_rows(probs: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Normalize each row to sum to one with epsilon protection."""
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)


def _resolve_hcc_cfg(cfg: Any) -> Dict[str, Any]:
    """Read HCC config with backward-compatible fallback locations."""
    hcc_cfg = _section_to_dict(getattr(cfg, "hcc", None))
    if not hcc_cfg and hasattr(cfg, "get"):
        hcc_cfg = _section_to_dict(cfg.get("hcc", None))
    if not hcc_cfg:
        model_cfg = _section_to_dict(getattr(cfg, "model", None))
        hcc_cfg = _section_to_dict(model_cfg.get("hcc", None))
    return hcc_cfg


def _resolve_hcc_eps(cfg: Any, default: float = _EPS) -> float:
    """Read the HCC numerical epsilon from config, falling back to a safe default."""
    hcc_cfg = _resolve_hcc_cfg(cfg)

    raw_eps = hcc_cfg.get("eps", default) if hcc_cfg else default
    eps = float(raw_eps) if raw_eps is not None else float(default)
    return eps if eps > 0.0 else float(default)


def _resolve_hcc_use_conditional_prob(cfg: Any, default: bool = False) -> bool:
    """Read whether HCC probability-path losses should be parent-conditioned."""
    hcc_cfg = _resolve_hcc_cfg(cfg)
    if not hcc_cfg:
        return bool(default)

    raw_value = hcc_cfg.get("use_conditional_prob", default)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(raw_value)


def _resolve_hcc_conditioned_levels(cfg: Any, default: Tuple[int, ...] = (1, 2)) -> Tuple[int, ...]:
    """Read which hierarchy levels should use conditional log-probabilities."""
    hcc_cfg = _resolve_hcc_cfg(cfg)
    if not hcc_cfg:
        return tuple(default)

    raw_levels = hcc_cfg.get("conditional_prob_levels", None)
    if raw_levels is None:
        return tuple(default)

    parsed_levels: List[int] = []
    if isinstance(raw_levels, bool):
        raise ValueError("hcc.conditional_prob_levels must not be a boolean.")
    if isinstance(raw_levels, (int, float)):
        parsed_levels = [int(raw_levels)]
    elif isinstance(raw_levels, str):
        normalized = raw_levels.strip().lower()
        if normalized in {"", "none", "null"}:
            return tuple()
        cleaned = normalized
        for ch in "[]()":
            cleaned = cleaned.replace(ch, " ")
        tokens = [token for token in cleaned.replace(",", " ").split() if token]
        try:
            parsed_levels = [int(token) for token in tokens]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "hcc.conditional_prob_levels string must contain integer level ids, "
                "for example '1', '2', or '1,2'."
            ) from exc
    elif hasattr(raw_levels, "items"):
        raise ValueError("hcc.conditional_prob_levels must not be a mapping/object.")
    elif hasattr(raw_levels, "__iter__"):
        try:
            parsed_levels = [int(level) for level in raw_levels]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "hcc.conditional_prob_levels iterable must contain integer level ids."
            ) from exc
    else:
        raise ValueError(
            "hcc.conditional_prob_levels must be int, iterable of ints, or comma-separated string."
        )

    deduped: List[int] = []
    seen = set()
    for level in parsed_levels:
        if level <= 0:
            continue
        if level in seen:
            continue
        deduped.append(level)
        seen.add(level)
    return tuple(deduped)


def _label_smoothing_from_cfg(cfg: Any) -> float:
    """Extract label smoothing from the training config and clamp it to [0, 1]."""
    train_cfg = _section_to_dict(getattr(cfg, "train", None))
    if not train_cfg and hasattr(cfg, "get"):
        train_cfg = _section_to_dict(cfg.get("train", None))
    return min(max(float(train_cfg.get("smoothing", 0.0)), 0.0), 1.0)


def _soft_targets_from_input(
    scores_per_level: List[torch.Tensor],
    targets: HcastTargets,
) -> Optional[List[torch.Tensor]]:
    """Validate and normalize per-level soft targets, typically from MixUp/CutMix."""
    if not isinstance(targets, dict):
        return None

    soft_targets = targets.get("soft_targets_per_level")
    if not isinstance(soft_targets, (list, tuple)) or len(soft_targets) != len(scores_per_level):
        return None

    out: List[torch.Tensor] = []
    for scores, target_level in zip(scores_per_level, soft_targets):
        if not isinstance(target_level, torch.Tensor) or target_level.ndim != 2:
            return None
        if int(target_level.size(1)) != int(scores.size(-1)):
            return None
        # MixUp/CutMix provide dense targets per level. We move them onto the
        # same device/dtype as the selected score source and re-normalize in
        # case interpolation introduced small row-sum drift.
        probs = target_level.to(device=scores.device, dtype=scores.dtype)
        out.append(_normalize_rows(probs))
    return out


def _hard_targets_from_input(targets: HcastTargets, num_levels: int) -> Optional[torch.Tensor]:
    """Extract hard class indices with shape [batch, num_levels] from the input targets."""
    hard_targets = (
        targets
        if isinstance(targets, torch.Tensor)
        else targets.get("hard_targets")
        if isinstance(targets, dict)
        else None
    )
    if not isinstance(hard_targets, torch.Tensor):
        return None
    if hard_targets.ndim != 2 or int(hard_targets.size(1)) != int(num_levels):
        return None
    return hard_targets.long()


def _soft_cross_entropy_from_log_probs(log_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy when both predictions and targets are distributions."""
    probs = target_probs.to(device=log_probs.device, dtype=log_probs.dtype)
    probs = _normalize_rows(probs)
    return -(probs * log_probs).sum(dim=-1).mean()


def _label_smoothed_target_probs(
    target: torch.Tensor,
    num_classes: int,
    smoothing: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a label-smoothed target distribution from hard labels."""
    off_value = smoothing / max(float(num_classes), 1.0)
    on_value = 1.0 - smoothing + off_value
    out = torch.full(
        (target.size(0), num_classes),
        fill_value=off_value,
        dtype=dtype,
        device=target.device,
    )
    return out.scatter_(1, target.unsqueeze(1), on_value)


def _dynamic_level_weights(
    scores_per_level: List[torch.Tensor],
    hard_targets: torch.Tensor,
    gamma: float,
    eps: float,
) -> List[float]:
    """Estimate per-level loss weights from current accuracy and class-count priors."""
    num_levels = len(scores_per_level)
    if num_levels == 0:
        return []

    num_classes_per_level = [max(int(scores.size(-1)), 0) for scores in scores_per_level]
    total_classes = float(sum(num_classes_per_level))
    if total_classes <= eps:
        omega_init = [1.0 / num_levels for _ in range(num_levels)]
    else:
        omega_init = [1.0 - (float(num_classes) / total_classes) for num_classes in num_classes_per_level]
        omega_sum = float(sum(omega_init))
        omega_init = [value / omega_sum for value in omega_init] if omega_sum > eps else [1.0 / num_levels] * num_levels

    rho: List[float] = []
    for level, scores in enumerate(scores_per_level):
        acc = (scores.argmax(dim=-1) == hard_targets[:, level]).float().mean().item()
        rho.append((1.0 - float(acc)) * omega_init[level])

    rho_sum = float(sum(rho))
    base = omega_init if rho_sum <= eps else [value / rho_sum for value in rho]
    scale = 1.0 - min(max(float(gamma), 0.0), 1.0)
    return [scale * value for value in base]


def _global_kl_loss(
    scores_per_level: List[torch.Tensor],
    hard_targets: Optional[torch.Tensor],
    target_probs_per_level: Optional[List[torch.Tensor]],
) -> torch.Tensor:
    """Compute the optional global KL regularizer over concatenated hierarchy outputs."""
    # Global KL treats the whole hierarchy as one concatenated categorical
    # vector. This regularizer therefore depends on the same score source
    # chosen for the per-level losses below.
    all_outputs = F.log_softmax(torch.cat(scores_per_level, dim=1), dim=1)

    if target_probs_per_level is not None:
        all_targets = torch.cat(
            [target_probs.to(dtype=all_outputs.dtype) for target_probs in target_probs_per_level],
            dim=1,
        )
    else:
        if hard_targets is None:
            raise ValueError("Either `hard_targets` or `target_probs_per_level` must be provided for global KL.")
        onehots = [
            F.one_hot(hard_targets[:, level], num_classes=scores.size(-1)).float()
            for level, scores in enumerate(scores_per_level)
        ]
        all_targets = torch.cat(onehots, dim=1)

    all_targets = all_targets.to(device=all_outputs.device, dtype=all_outputs.dtype)
    all_targets = _normalize_rows(all_targets)
    return F.kl_div(all_outputs, all_targets, reduction="batchmean")


def _level_losses_from_scores(
    scores_per_level: List[torch.Tensor],
    hard_targets: Optional[torch.Tensor],
    target_probs_per_level: Optional[List[torch.Tensor]],
    smoothing: float,
    scores_are_log_probs: bool,
) -> List[torch.Tensor]:
    """Compute one loss value per hierarchy level from logits or log-probabilities."""
    if target_probs_per_level is not None:
        # Soft-target path used for MixUp/CutMix. If the caller already passed
        # log-probabilities we reuse them directly; otherwise we derive
        # log-softmax from logits first.
        log_probs_per_level = (
            scores_per_level
            if scores_are_log_probs
            else [F.log_softmax(scores, dim=-1) for scores in scores_per_level]
        )
        return [
            _soft_cross_entropy_from_log_probs(log_probs, target_probs_per_level[level])
            for level, log_probs in enumerate(log_probs_per_level)
        ]

    if hard_targets is None:
        raise TypeError("Expected hard targets tensor of shape [B, L].")

    if not scores_are_log_probs:
        # Standard H-CAST path: raw logits go straight into cross_entropy.
        return [
            F.cross_entropy(scores, hard_targets[:, level], label_smoothing=smoothing)
            for level, scores in enumerate(scores_per_level)
        ]

    if smoothing <= 0.0:
        # Probability path: the caller already converted probabilities into
        # log-probabilities, so the matching hard-label loss is NLL.
        return [
            F.nll_loss(log_probs, hard_targets[:, level])
            for level, log_probs in enumerate(scores_per_level)
        ]

    # Same probability path as above, but label smoothing requires building
    # smoothed target distributions and taking explicit soft cross-entropy.
    return [
        _soft_cross_entropy_from_log_probs(
            log_probs,
            _label_smoothed_target_probs(
                hard_targets[:, level],
                num_classes=log_probs.size(-1),
                smoothing=smoothing,
                dtype=log_probs.dtype,
            ),
        )
        for level, log_probs in enumerate(scores_per_level)
    ]


def _resolve_score_source(output: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[torch.Tensor], bool]:
    """Choose whether losses should read from raw logits or HCC effective probabilities."""
    logits_per_level = output["logits_per_level"]
    if not isinstance(logits_per_level, list) or not logits_per_level:
        raise ValueError("H-CAST output must contain non-empty `logits_per_level`.")

    effective_probs = output.get("effective_probs_per_level")
    if effective_probs is None:
        # Before HCC activates, or when HCC is disabled, downstream loss stays
        # on the original logits exactly like baseline H-CAST.
        return logits_per_level, logits_per_level, False
    if not isinstance(effective_probs, list) or len(effective_probs) != len(logits_per_level):
        raise ValueError("`effective_probs_per_level` must be None or a list aligned with `logits_per_level`.")
    # Once HCC emits final probabilities, the loss switches to those values
    # rather than the raw logits. The caller must then treat them as
    # probabilities, not as logits.
    return logits_per_level, effective_probs, True


def _resolve_level_weights(
    loss_cfg: Dict[str, Any],
    score_source_per_level: List[torch.Tensor],
    hard_targets: Optional[torch.Tensor],
    num_levels: int,
) -> List[float]:
    """Return static or dynamic per-level weights according to the loss config."""
    level_weight_cfg = _section_to_dict(loss_cfg.get("level_weighting", {}))
    mode = str(level_weight_cfg.get("mode", "static")).strip().lower()
    if mode != "dynamic":
        return [1.0 for _ in range(num_levels)]

    if hard_targets is None:
        raise ValueError(
            "Dynamic level weighting requires hard targets. "
            "Pass `hard_targets` in target dict when using soft targets."
        )

    gamma = float(level_weight_cfg.get("gamma", 0.0))
    eps = max(float(level_weight_cfg.get("eps", _EPS)), _EPS)
    # Dynamic weights are computed from the same score source used by the loss,
    # so when HCC is active they react to constrained probabilities, not raw
    # logits.
    return _dynamic_level_weights(
        scores_per_level=score_source_per_level,
        hard_targets=hard_targets,
        gamma=gamma,
        eps=eps,
    )


def _parent_index_for_level(
    taxonomy: Optional[Dict[str, Any]],
    level: int,
    num_children: int,
    num_parents: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build child->parent indices for transition (level-1 -> level)."""
    if not isinstance(taxonomy, dict):
        return None

    parent_of = taxonomy.get("parent_of")
    if not isinstance(parent_of, dict):
        return None

    raw_mapping = parent_of.get(level, parent_of.get(str(level)))
    if not isinstance(raw_mapping, dict):
        return None

    parent_index = [-1 for _ in range(num_children)]
    for child_raw, parent_raw in raw_mapping.items():
        try:
            child = int(child_raw)
            parent = int(parent_raw)
        except (TypeError, ValueError):
            return None
        if child < 0 or child >= num_children:
            return None
        if parent < 0 or parent >= num_parents:
            return None
        parent_index[child] = parent

    if any(parent < 0 for parent in parent_index):
        return None
    return torch.tensor(parent_index, dtype=torch.long, device=device)


def _condition_selected_levels_log_probs(
    probs_per_level: List[torch.Tensor],
    base_log_probs_per_level: List[torch.Tensor],
    taxonomy: Optional[Dict[str, Any]],
    eps: float,
    conditioned_levels: Tuple[int, ...] = (1, 2),
) -> List[torch.Tensor]:
    """Convert selected levels from marginal log p(child) to log p(child|parent)."""
    conditioned = list(base_log_probs_per_level)
    num_levels = len(probs_per_level)

    for level in conditioned_levels:
        if level <= 0 or level >= num_levels:
            continue

        child_probs = probs_per_level[level]
        parent_probs = probs_per_level[level - 1]
        parent_index = _parent_index_for_level(
            taxonomy=taxonomy,
            level=level,
            num_children=int(child_probs.size(1)),
            num_parents=int(parent_probs.size(1)),
            device=child_probs.device,
        )
        if parent_index is None:
            raise ValueError(
                f"Conditioned loss for level {level} requires taxonomy['parent_of'][{level}] "
                "with a complete child->parent mapping."
            )

        parent_mass_for_child = parent_probs[:, parent_index].clamp_min(eps)
        conditioned[level] = conditioned[level] - torch.log(parent_mass_for_child)

    return conditioned


def compute_loss(
    output: Dict[str, Any],
    targets: HcastTargets,
    cfg: Any,
    _taxonomy: Optional[Dict[str, Any]] = None,
    return_aux: bool = False,
) -> Union[
    Tuple[torch.Tensor, Dict[str, float]],
    Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]],
]:
    loss_cfg = _section_to_dict(getattr(cfg.model, "loss", {}))

    # - if effective_probs_per_level is absent, training uses raw logits
    # - if effective_probs_per_level is present, training uses HCC outputs
    logits_per_level, score_source_per_level, use_probability_scores = _resolve_score_source(output)
    target_probs_per_level = _soft_targets_from_input(score_source_per_level, targets)
    hard_targets = _hard_targets_from_input(targets, num_levels=len(logits_per_level))

    if target_probs_per_level is None and hard_targets is None:
        raise TypeError("Expected hard targets tensor of shape [B, L].")
    if hard_targets is not None:
        hard_targets = hard_targets.to(device=score_source_per_level[0].device, dtype=torch.long)

    smoothing = _label_smoothing_from_cfg(cfg)
    if use_probability_scores:
        eps = _resolve_hcc_eps(cfg, default=_EPS)
        use_conditional_prob = _resolve_hcc_use_conditional_prob(cfg, default=False)
        # effective_probs_per_level are already normalized probabilities, so we
        # convert them into log-probabilities for NLL / soft cross-entropy.
        # Passing them into cross_entropy directly would be wrong because that
        # function expects logits and would apply log_softmax again.
        base_log_probs_per_level = [torch.log(probs.clamp_min(eps)) for probs in score_source_per_level]
        if use_conditional_prob:
            conditioned_levels = _resolve_hcc_conditioned_levels(cfg, default=(1, 2))
            # Optional conditioned loss path:
            # log p(child|parent) = log p(child) - log p(parent).
            loss_scores_per_level = _condition_selected_levels_log_probs(
                probs_per_level=score_source_per_level,
                base_log_probs_per_level=base_log_probs_per_level,
                taxonomy=_taxonomy,
                eps=eps,
                conditioned_levels=conditioned_levels,
            )
        else:
            # Default behavior: keep marginal log-probabilities per level.
            loss_scores_per_level = base_log_probs_per_level
        gk_scores_per_level = base_log_probs_per_level
        scores_are_log_probs = True
    else:
        # Baseline path: keep raw logits and let PyTorch handle log-softmax
        # internally inside cross_entropy.
        loss_scores_per_level = logits_per_level
        gk_scores_per_level = logits_per_level
        scores_are_log_probs = False

    level_losses = _level_losses_from_scores(
        scores_per_level=loss_scores_per_level,
        hard_targets=hard_targets,
        target_probs_per_level=target_probs_per_level,
        smoothing=smoothing,
        scores_are_log_probs=scores_are_log_probs,
    )

    level_weights = _resolve_level_weights(
        loss_cfg=loss_cfg,
        score_source_per_level=score_source_per_level,
        hard_targets=hard_targets,
        num_levels=len(level_losses),
    )

    # Main task loss is the weighted sum of per-level losses.
    weighted_level_losses = [
        level_loss * float(level_weights[level])
        for level, level_loss in enumerate(level_losses)
    ]
    weighted_level_ce = torch.stack(weighted_level_losses).sum()
    total = weighted_level_ce

    if bool(loss_cfg.get("globalkl", False)):
        global_kl_weight = float(loss_cfg.get("gk_weight", 1.0))
        # Keep global KL on the unconditioned score path. Conditioning is
        # applied only to level-1/2 task losses above.
        gk_loss = _global_kl_loss(
            scores_per_level=gk_scores_per_level,
            hard_targets=hard_targets,
            target_probs_per_level=target_probs_per_level,
        )
        total = total + global_kl_weight * gk_loss
    else:
        gk_loss = torch.zeros((), device=total.device, dtype=total.dtype)

    metrics: Dict[str, float] = {
        "total": float(total.detach().item()),
        "level_ce": float(weighted_level_ce.detach().item()),
        "gk_loss": float(gk_loss.detach().item()),
    }
    for level, level_loss in enumerate(level_losses):
        metrics[f"loss_level_{level}"] = float(level_loss.detach().item())
        metrics[f"loss_weight_level_{level}"] = float(level_weights[level])

    if not return_aux:
        return total, metrics

    aux_payload: Dict[str, Any] = {"level_losses": list(level_losses)}
    return total, metrics, aux_payload
