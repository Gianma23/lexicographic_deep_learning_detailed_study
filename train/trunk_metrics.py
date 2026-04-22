from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

_GRAD_EPS = 1e-12
_GRAD_LEVEL_NAMES = ("coarse", "mid", "fine")


def _trainable_named_params(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _capture_trainable_param_snapshot(params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
    """Capture trainable parameter values for epoch-to-epoch delta norms."""
    return [param.detach().to(device="cpu", dtype=torch.float32).clone() for param in params]


def _param_norm_from_values(
    params: Sequence[torch.nn.Parameter],
    include_mask: Sequence[bool],
) -> float:
    norm_sq = None
    for param, include in zip(params, include_mask):
        if not include:
            continue
        value_fp = param.detach().float()
        sq_value = torch.sum(value_fp * value_fp)
        norm_sq = sq_value if norm_sq is None else norm_sq + sq_value
    if norm_sq is None:
        return 0.0
    return float(torch.sqrt(norm_sq.clamp_min(0.0)).item())


def _delta_param_norm_from_snapshot(
    params: Sequence[torch.nn.Parameter],
    start_snapshot: Sequence[torch.Tensor],
    include_mask: Sequence[bool],
) -> float:
    norm_sq = 0.0
    used = 0
    for param, start_tensor, include in zip(params, start_snapshot, include_mask):
        if not include:
            continue
        current_tensor = param.detach().to(device="cpu", dtype=torch.float32)
        delta_tensor = current_tensor - start_tensor
        norm_sq += float(torch.sum(delta_tensor * delta_tensor).item())
        used += 1
    if used <= 0:
        return 0.0
    return float(norm_sq**0.5)


def _extract_level_losses(loss_aux: Any) -> List[torch.Tensor]:
    if not isinstance(loss_aux, dict):
        return []
    raw_level_losses = loss_aux.get("level_losses")
    if not isinstance(raw_level_losses, (list, tuple)):
        return []

    level_losses: List[torch.Tensor] = []
    for level_loss in raw_level_losses[: len(_GRAD_LEVEL_NAMES)]:
        if (
            isinstance(level_loss, torch.Tensor)
            and int(level_loss.ndim) == 0
            and bool(level_loss.requires_grad)
        ):
            level_losses.append(level_loss)
    return level_losses


def _grad_norm_from_autograd_grads(
    grads: Sequence[Optional[torch.Tensor]],
    include_mask: Sequence[bool],
) -> float:
    norm_sq = None
    for grad, include in zip(grads, include_mask):
        if not include or grad is None:
            continue
        grad_fp = grad.detach().float()
        value = torch.sum(grad_fp * grad_fp)
        norm_sq = value if norm_sq is None else norm_sq + value

    if norm_sq is None:
        return 0.0
    return float(torch.sqrt(norm_sq.clamp_min(0.0)).item())


def _grad_cosine_from_autograd_grads(
    grads_a: Sequence[Optional[torch.Tensor]],
    grads_b: Sequence[Optional[torch.Tensor]],
    include_mask: Sequence[bool],
    eps: float = _GRAD_EPS,
) -> float:
    dot = None
    norm_a_sq = None
    norm_b_sq = None
    for grad_a, grad_b, include in zip(grads_a, grads_b, include_mask):
        if not include or grad_a is None or grad_b is None:
            continue
        grad_a_fp = grad_a.detach().float()
        grad_b_fp = grad_b.detach().float()

        dot_term = torch.sum(grad_a_fp * grad_b_fp)
        norm_a_term = torch.sum(grad_a_fp * grad_a_fp)
        norm_b_term = torch.sum(grad_b_fp * grad_b_fp)

        dot = dot_term if dot is None else dot + dot_term
        norm_a_sq = norm_a_term if norm_a_sq is None else norm_a_sq + norm_a_term
        norm_b_sq = norm_b_term if norm_b_sq is None else norm_b_sq + norm_b_term

    if dot is None or norm_a_sq is None or norm_b_sq is None:
        return 0.0

    denom = torch.sqrt(norm_a_sq.clamp_min(0.0)) * torch.sqrt(norm_b_sq.clamp_min(0.0))
    cosine = dot / (denom + float(eps))
    return float(cosine.clamp(min=-1.0, max=1.0).item())


def _resolve_trunk_masks(
    coarse_grads: Sequence[Optional[torch.Tensor]],
    mid_grads: Sequence[Optional[torch.Tensor]],
    fine_grads: Sequence[Optional[torch.Tensor]],
) -> Dict[str, List[bool]]:
    t1_mask = []
    t2_mask = []
    t3_mask = []
    for coarse_grad, mid_grad, fine_grad in zip(coarse_grads, mid_grads, fine_grads):
        coarse_active = coarse_grad is not None
        mid_active = mid_grad is not None
        fine_active = fine_grad is not None
        t1_mask.append(coarse_active and mid_active and fine_active)
        t2_mask.append(coarse_active and mid_active and (not fine_active))
        t3_mask.append(coarse_active and (not mid_active) and (not fine_active))
    return {"t1": t1_mask, "t2": t2_mask, "t3": t3_mask}


def _compute_level_grad_map(
    trainable_named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    level_losses: Sequence[torch.Tensor],
    retain_graph: bool,
) -> Optional[Dict[str, Tuple[Optional[torch.Tensor], ...]]]:
    if not trainable_named_params or len(level_losses) < 3:
        return None

    selected_level_losses = list(level_losses[:3])
    trainable_params = [param for _, param in trainable_named_params]

    level_grad_map: Dict[str, Tuple[Optional[torch.Tensor], ...]] = {}
    num_levels = len(selected_level_losses)
    for idx, (level_name, level_loss) in enumerate(zip(_GRAD_LEVEL_NAMES, selected_level_losses)):
        keep_graph = bool(retain_graph or (idx < (num_levels - 1)))
        level_grad_map[level_name] = torch.autograd.grad(
            level_loss,
            trainable_params,
            retain_graph=keep_graph,
            allow_unused=True,
        )
    return level_grad_map


def _merge_masks(mask_a: Sequence[bool], mask_b: Sequence[bool]) -> List[bool]:
    return [bool(a or b) for a, b in zip(mask_a, mask_b)]


def _mask_views_from_trunk_masks(trunk_masks: Mapping[str, Sequence[bool]]) -> Dict[str, List[bool]]:
    t1_mask = list(trunk_masks.get("t1", []))
    t2_mask = list(trunk_masks.get("t2", []))
    t3_mask = list(trunk_masks.get("t3", []))
    t2t1_mask = _merge_masks(t2_mask, t1_mask)
    t3t2t1_mask = _merge_masks(t3_mask, t2t1_mask)
    return {
        "t1": t1_mask,
        "t2": t2_mask,
        "t3": t3_mask,
        "t2t1": t2t1_mask,
        "t3t2t1": t3t2t1_mask,
    }


def _dot_from_autograd_grads(
    grads_a: Sequence[Optional[torch.Tensor]],
    grads_b: Sequence[Optional[torch.Tensor]],
    include_mask: Sequence[bool],
) -> torch.Tensor:
    dot = None
    for grad_a, grad_b, include in zip(grads_a, grads_b, include_mask):
        if not include or grad_a is None or grad_b is None:
            continue
        grad_a_fp = grad_a.detach().float()
        grad_b_fp = grad_b.detach().float()
        term = torch.sum(grad_a_fp * grad_b_fp)
        dot = term if dot is None else dot + term
    if dot is None:
        return torch.zeros((), dtype=torch.float32)
    return dot


def _project_onto_reference(
    target_grads: Sequence[Optional[torch.Tensor]],
    reference_grads: Sequence[Optional[torch.Tensor]],
    include_mask: Sequence[bool],
    eps: float,
) -> Tuple[Tuple[Optional[torch.Tensor], ...], float, bool]:
    denom = _dot_from_autograd_grads(reference_grads, reference_grads, include_mask)
    denom_value = float(denom.item())

    if denom_value <= float(eps):
        return tuple(target_grads), 0.0, False

    numer = _dot_from_autograd_grads(target_grads, reference_grads, include_mask)
    coeff = float((numer / denom).item())

    projected: List[Optional[torch.Tensor]] = []
    for grad_target, grad_ref, include in zip(target_grads, reference_grads, include_mask):
        if grad_target is None:
            projected.append(None)
            continue
        if (not include) or grad_ref is None:
            projected.append(grad_target)
            continue
        projected.append(grad_target - (coeff * grad_ref))
    return tuple(projected), coeff, True


def _sum_grad_tuples(
    grads_a: Sequence[Optional[torch.Tensor]],
    grads_b: Sequence[Optional[torch.Tensor]],
    grads_c: Sequence[Optional[torch.Tensor]],
) -> Tuple[Optional[torch.Tensor], ...]:
    summed: List[Optional[torch.Tensor]] = []
    for grad_a, grad_b, grad_c in zip(grads_a, grads_b, grads_c):
        pieces = [grad for grad in (grad_a, grad_b, grad_c) if grad is not None]
        if not pieces:
            summed.append(None)
            continue
        total = pieces[0]
        for piece in pieces[1:]:
            total = total + piece
        summed.append(total)
    return tuple(summed)


def _scale_grad_tuple(
    grads: Sequence[Optional[torch.Tensor]],
    factor: float,
) -> Tuple[Optional[torch.Tensor], ...]:
    factor_f = float(factor)
    if abs(factor_f - 1.0) <= 1e-12:
        return tuple(grads)

    scaled: List[Optional[torch.Tensor]] = []
    for grad in grads:
        if grad is None:
            scaled.append(None)
        else:
            scaled.append(grad * factor_f)
    return tuple(scaled)


def _trunk_grad_norm_metrics(
    coarse_grads: Sequence[Optional[torch.Tensor]],
    mid_grads: Sequence[Optional[torch.Tensor]],
    fine_grads: Sequence[Optional[torch.Tensor]],
    mask_views: Mapping[str, Sequence[bool]],
    prefix: str = "",
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    if any(mask_views["t3t2t1"]):
        out[f"{prefix}grad_norm_t3t2t1_coarse"] = _grad_norm_from_autograd_grads(
            coarse_grads,
            mask_views["t3t2t1"],
        )

    if any(mask_views["t3"]):
        out[f"{prefix}grad_norm_t3_coarse"] = _grad_norm_from_autograd_grads(
            coarse_grads,
            mask_views["t3"],
        )

    if any(mask_views["t2t1"]):
        out[f"{prefix}grad_norm_t2t1_coarse"] = _grad_norm_from_autograd_grads(
            coarse_grads,
            mask_views["t2t1"],
        )
        out[f"{prefix}grad_norm_t2t1_mid"] = _grad_norm_from_autograd_grads(
            mid_grads,
            mask_views["t2t1"],
        )

    if any(mask_views["t2"]):
        out[f"{prefix}grad_norm_t2_coarse"] = _grad_norm_from_autograd_grads(
            coarse_grads,
            mask_views["t2"],
        )
        out[f"{prefix}grad_norm_t2_mid"] = _grad_norm_from_autograd_grads(
            mid_grads,
            mask_views["t2"],
        )

    if any(mask_views["t1"]):
        out[f"{prefix}grad_norm_t1_coarse"] = _grad_norm_from_autograd_grads(
            coarse_grads,
            mask_views["t1"],
        )
        out[f"{prefix}grad_norm_t1_mid"] = _grad_norm_from_autograd_grads(
            mid_grads,
            mask_views["t1"],
        )
        out[f"{prefix}grad_norm_t1_fine"] = _grad_norm_from_autograd_grads(
            fine_grads,
            mask_views["t1"],
        )

    return out


def _trunk_param_norm_metrics(
    params: Sequence[torch.nn.Parameter],
    start_snapshot: Sequence[torch.Tensor],
    trunk_masks: Optional[Mapping[str, Sequence[bool]]],
) -> Dict[str, float]:
    if trunk_masks is None:
        return {}

    mask_views = _mask_views_from_trunk_masks(trunk_masks)
    metrics: Dict[str, float] = {}
    trunk_order = ("t3t2t1", "t3", "t2t1", "t2", "t1")

    for trunk_name in trunk_order:
        mask = list(mask_views.get(trunk_name, []))
        if not mask or not any(mask):
            continue
        metrics[f"param_norm_{trunk_name}"] = _param_norm_from_values(params, mask)
        metrics[f"delta_param_norm_{trunk_name}"] = _delta_param_norm_from_snapshot(
            params=params,
            start_snapshot=start_snapshot,
            include_mask=mask,
        )
    return metrics


def _build_lexicographic_grads(
    coarse_grads: Sequence[Optional[torch.Tensor]],
    mid_grads: Sequence[Optional[torch.Tensor]],
    fine_grads: Sequence[Optional[torch.Tensor]],
    trunk_masks: Mapping[str, Sequence[bool]],
    eps: float = _GRAD_EPS,
    include_metrics: bool = True,
) -> Tuple[Dict[str, Tuple[Optional[torch.Tensor], ...]], Dict[str, float]]:
    t1_mask = list(trunk_masks.get("t1", []))
    t2_mask = list(trunk_masks.get("t2", []))
    coarse_mid_mask = _merge_masks(t1_mask, t2_mask)

    mid_projected, _mid_coeff, mid_applied = _project_onto_reference(
        target_grads=mid_grads,
        reference_grads=coarse_grads,
        include_mask=coarse_mid_mask,
        eps=eps,
    )

    mid_basis_t1, _fine_mid_basis_coeff, fine_mid_basis_applied = _project_onto_reference(
        target_grads=mid_projected,
        reference_grads=coarse_grads,
        include_mask=t1_mask,
        eps=eps,
    )

    fine_after_coarse, _fine_coeff_coarse, fine_applied_coarse = _project_onto_reference(
        target_grads=fine_grads,
        reference_grads=coarse_grads,
        include_mask=t1_mask,
        eps=eps,
    )
    fine_projected, _fine_coeff_mid_basis, fine_applied_mid_basis = _project_onto_reference(
        target_grads=fine_after_coarse,
        reference_grads=mid_basis_t1,
        include_mask=t1_mask,
        eps=eps,
    )

    total_grads = _sum_grad_tuples(coarse_grads, mid_projected, fine_projected)
    grad_pack: Dict[str, Tuple[Optional[torch.Tensor], ...]] = {
        "coarse": tuple(coarse_grads),
        "mid_projected": mid_projected,
        "fine_projected": fine_projected,
        "total": total_grads,
    }

    if not include_metrics:
        return grad_pack, {}

    metrics: Dict[str, float] = {
        "post_projection_applied_t2t1_mid_coarse": 1.0 if mid_applied else 0.0,
        "post_projection_applied_t1_mid_proj_coarse": 1.0 if fine_mid_basis_applied else 0.0,
        "post_projection_applied_t1_fine_coarse": 1.0 if fine_applied_coarse else 0.0,
        "post_projection_applied_t1_fine_mid_proj": 1.0 if fine_applied_mid_basis else 0.0,
    }
    all_params_mask = [True] * len(coarse_grads)

    metrics["cos_mid_coarse"] = _grad_cosine_from_autograd_grads(
        mid_grads,
        coarse_grads,
        all_params_mask,
        eps=eps,
    )
    metrics["post_cos_mid_coarse"] = _grad_cosine_from_autograd_grads(
        mid_projected,
        coarse_grads,
        all_params_mask,
        eps=eps,
    )
    metrics["cos_fine_coarse"] = _grad_cosine_from_autograd_grads(
        fine_grads,
        coarse_grads,
        all_params_mask,
        eps=eps,
    )
    metrics["post_cos_fine_coarse"] = _grad_cosine_from_autograd_grads(
        fine_projected,
        coarse_grads,
        all_params_mask,
        eps=eps,
    )
    metrics["cos_fine_mid"] = _grad_cosine_from_autograd_grads(
        fine_grads,
        mid_grads,
        all_params_mask,
        eps=eps,
    )
    metrics["post_cos_fine_mid"] = _grad_cosine_from_autograd_grads(
        fine_projected,
        mid_projected,
        all_params_mask,
        eps=eps,
    )

    return grad_pack, metrics


def _prepare_lexicographic_update(
    trainable_named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    level_losses: Sequence[torch.Tensor],
    eps: float = _GRAD_EPS,
    include_metrics: bool = True,
    grad_scale: float = 1.0,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:
    level_grad_map = _compute_level_grad_map(
        trainable_named_params=trainable_named_params,
        level_losses=level_losses,
        retain_graph=False,
    )
    if level_grad_map is None:
        return None, {}

    coarse_grads = level_grad_map["coarse"]
    mid_grads = level_grad_map["mid"]
    fine_grads = level_grad_map["fine"]
    trunk_masks = _resolve_trunk_masks(
        coarse_grads=coarse_grads,
        mid_grads=mid_grads,
        fine_grads=fine_grads,
    )

    projected_grads, _ = _build_lexicographic_grads(
        coarse_grads=coarse_grads,
        mid_grads=mid_grads,
        fine_grads=fine_grads,
        trunk_masks=trunk_masks,
        eps=eps,
        include_metrics=False,
    )

    metrics: Dict[str, float] = {}
    if include_metrics:
        safe_scale = float(grad_scale) if float(grad_scale) > 0.0 else 1.0
        unscale = 1.0 / safe_scale

        coarse_for_log = _scale_grad_tuple(coarse_grads, unscale)
        mid_for_log = _scale_grad_tuple(mid_grads, unscale)
        fine_for_log = _scale_grad_tuple(fine_grads, unscale)

        projected_for_log, lex_metrics = _build_lexicographic_grads(
            coarse_grads=coarse_for_log,
            mid_grads=mid_for_log,
            fine_grads=fine_for_log,
            trunk_masks=trunk_masks,
            eps=eps,
            include_metrics=True,
        )
        metrics.update(lex_metrics)

        mask_views = _mask_views_from_trunk_masks(trunk_masks)
        post_metrics = _trunk_grad_norm_metrics(
            coarse_grads=projected_for_log["coarse"],
            mid_grads=projected_for_log["mid_projected"],
            fine_grads=projected_for_log["fine_projected"],
            mask_views=mask_views,
            prefix="post_",
        )
        metrics.update(post_metrics)

    state: Dict[str, Any] = {
        "trunk_masks": trunk_masks,
        "level_grad_map": level_grad_map,
        "projected_grads": projected_grads,
    }
    return state, metrics


def _prepare_trunk_grad_metrics(
    trainable_named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    level_losses: Sequence[torch.Tensor],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:
    level_grad_map = _compute_level_grad_map(
        trainable_named_params=trainable_named_params,
        level_losses=level_losses,
        retain_graph=True,
    )
    if level_grad_map is None:
        return None, {}

    coarse_grads = level_grad_map["coarse"]
    mid_grads = level_grad_map["mid"]
    fine_grads = level_grad_map["fine"]
    trunk_masks = _resolve_trunk_masks(
        coarse_grads=coarse_grads,
        mid_grads=mid_grads,
        fine_grads=fine_grads,
    )
    mask_views = _mask_views_from_trunk_masks(trunk_masks)

    metrics: Dict[str, float] = _trunk_grad_norm_metrics(
        coarse_grads=coarse_grads,
        mid_grads=mid_grads,
        fine_grads=fine_grads,
        mask_views=mask_views,
    )

    all_params_mask = [True] * len(coarse_grads)
    metrics["cos_mid_coarse"] = _grad_cosine_from_autograd_grads(
        mid_grads,
        coarse_grads,
        all_params_mask,
        eps=_GRAD_EPS,
    )
    metrics["cos_fine_coarse"] = _grad_cosine_from_autograd_grads(
        fine_grads,
        coarse_grads,
        all_params_mask,
        eps=_GRAD_EPS,
    )
    metrics["cos_fine_mid"] = _grad_cosine_from_autograd_grads(
        fine_grads,
        mid_grads,
        all_params_mask,
        eps=_GRAD_EPS,
    )

    if not metrics:
        return None, {}

    state: Dict[str, Any] = {"trunk_masks": trunk_masks}
    return state, metrics
