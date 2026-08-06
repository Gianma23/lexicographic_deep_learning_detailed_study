from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .types import GradTuple, LevelGradMap, LexicographicUpdateState, TrunkGradState

_GRAD_EPS = 1e-12
_GRAD_LEVEL_NAMES = ("coarse", "mid", "fine")
_LEX_PROJECTION_MODES = ("coarse_first", "fine_first")
_LEX_PROJECTION_RULES = ("orthogonalize_all", "conflict_only")

# Internal projection-flag key -> logged metric name. Flags are produced as
# on-device 0-dim tensors and reduced to floats in the batched metric transfer.
_PROJECTION_FLAG_METRICS = {
    "mid_off_coarse_t2": "post_projection_applied_t2_mid_coarse",
    "mid_off_coarse_t1": "post_projection_applied_t1_mid_coarse",
    "fine_off_higher_t1": "post_projection_applied_t1_fine_higher",
    "mid_off_fine_t1": "post_projection_applied_t1_mid_fine",
    "coarse_off_mid_t2": "post_projection_applied_t2_coarse_mid",
    "coarse_off_higher_t1": "post_projection_applied_t1_coarse_higher",
}


def get_trainable_named_params(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def capture_trainable_param_snapshot(params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
    """Capture trainable parameter values for epoch-to-epoch delta norms."""
    return [param.detach().to(device="cpu", dtype=torch.float32).clone() for param in params]


def extract_level_losses(loss_aux: Any) -> List[torch.Tensor]:
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


def assign_grads_to_params(
    params: Sequence[torch.nn.Parameter],
    grads: Sequence[Optional[torch.Tensor]],
) -> None:
    for param, grad in zip(params, grads):
        if grad is None:
            param.grad = None
            continue
        param.grad = grad.detach()


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


def _batched_grad_metrics(
    norm_specs: Sequence[
        Tuple[str, Sequence[Optional[torch.Tensor]], Sequence[bool]]
    ],
    cosine_specs: Sequence[
        Tuple[
            str,
            Sequence[Optional[torch.Tensor]],
            Sequence[Optional[torch.Tensor]],
            Sequence[bool],
            float,
        ]
    ],
    scalar_specs: Sequence[Tuple[str, torch.Tensor]] = (),
) -> Dict[str, float]:
    """Reduce several gradient diagnostics in one parameter scan.

    Each accumulator retains the previous implementation's parameter order.
    Per-parameter FP32 squares and dot products are shared across metrics, and
    the final scalar tensors are transferred to the CPU together to avoid one
    synchronization per logged value.

    ``scalar_specs`` carries already-reduced 0-dim tensors (for example the
    projection applied flags) so they join that same batched transfer instead of
    each forcing its own synchronization.
    """
    norm_sums: Dict[str, Optional[torch.Tensor]] = {
        name: None for name, _, _ in norm_specs
    }
    cosine_sums: Dict[
        str,
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
    ] = {name: (None, None, None) for name, _, _, _, _ in cosine_specs}

    sequence_lengths = [len(grads) for _, grads, _ in norm_specs]
    sequence_lengths.extend(len(grads_a) for _, grads_a, _, _, _ in cosine_specs)
    sequence_lengths.extend(len(grads_b) for _, _, grads_b, _, _ in cosine_specs)
    num_params = max(sequence_lengths, default=0)

    for param_idx in range(num_params):
        fp32_cache: Dict[int, torch.Tensor] = {}
        square_cache: Dict[int, torch.Tensor] = {}
        dot_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        def as_fp32(grad: torch.Tensor) -> torch.Tensor:
            cache_key = id(grad)
            value = fp32_cache.get(cache_key)
            if value is None:
                value = grad.detach().float()
                fp32_cache[cache_key] = value
            return value

        def square_sum(grad: torch.Tensor) -> torch.Tensor:
            cache_key = id(grad)
            value = square_cache.get(cache_key)
            if value is None:
                grad_fp = as_fp32(grad)
                value = torch.sum(grad_fp * grad_fp)
                square_cache[cache_key] = value
            return value

        def dot_sum(grad_a: torch.Tensor, grad_b: torch.Tensor) -> torch.Tensor:
            cache_key = (id(grad_a), id(grad_b))
            value = dot_cache.get(cache_key)
            if value is None:
                value = torch.sum(as_fp32(grad_a) * as_fp32(grad_b))
                dot_cache[cache_key] = value
            return value

        for name, grads, include_mask in norm_specs:
            if param_idx >= len(grads) or param_idx >= len(include_mask):
                continue
            grad = grads[param_idx]
            if not include_mask[param_idx] or grad is None:
                continue
            term = square_sum(grad)
            current = norm_sums[name]
            norm_sums[name] = term if current is None else current + term

        for name, grads_a, grads_b, include_mask, _ in cosine_specs:
            if (
                param_idx >= len(grads_a)
                or param_idx >= len(grads_b)
                or param_idx >= len(include_mask)
            ):
                continue
            grad_a = grads_a[param_idx]
            grad_b = grads_b[param_idx]
            if not include_mask[param_idx] or grad_a is None or grad_b is None:
                continue

            dot, norm_a_sq, norm_b_sq = cosine_sums[name]
            dot_term = dot_sum(grad_a, grad_b)
            norm_a_term = square_sum(grad_a)
            norm_b_term = square_sum(grad_b)
            cosine_sums[name] = (
                dot_term if dot is None else dot + dot_term,
                norm_a_term if norm_a_sq is None else norm_a_sq + norm_a_term,
                norm_b_term if norm_b_sq is None else norm_b_sq + norm_b_term,
            )

    scalar_tensors: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, float] = {}
    for name, _, _ in norm_specs:
        norm_sq = norm_sums[name]
        if norm_sq is None:
            metrics[name] = 0.0
        else:
            scalar_tensors[name] = torch.sqrt(norm_sq.clamp_min(0.0))

    cosine_eps = {name: float(eps) for name, _, _, _, eps in cosine_specs}
    for name, _, _, _, _ in cosine_specs:
        dot, norm_a_sq, norm_b_sq = cosine_sums[name]
        if dot is None or norm_a_sq is None or norm_b_sq is None:
            metrics[name] = 0.0
            continue
        denom = torch.sqrt(norm_a_sq.clamp_min(0.0)) * torch.sqrt(
            norm_b_sq.clamp_min(0.0)
        )
        cosine = dot / (denom + cosine_eps[name])
        scalar_tensors[name] = cosine.clamp(min=-1.0, max=1.0)

    for name, value in scalar_specs:
        scalar_tensors[name] = value.detach().to(dtype=torch.float32)

    # A model normally keeps all trainable parameters on one device. Grouping
    # by device also preserves correct behavior for the less common case where
    # diagnostic tensors span multiple devices.
    device_groups: Dict[torch.device, List[Tuple[str, torch.Tensor]]] = {}
    for name, value in scalar_tensors.items():
        device_groups.setdefault(value.device, []).append((name, value))
    for items in device_groups.values():
        values = torch.stack([value.detach().reshape(()) for _, value in items])
        cpu_values = values.to(device="cpu", dtype=torch.float32).tolist()
        for (name, _), value in zip(items, cpu_values):
            metrics[name] = float(value)

    return metrics


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
) -> Optional[LevelGradMap]:
    if not trainable_named_params or len(level_losses) < 3:
        return None

    selected_level_losses = list(level_losses[:3])
    trainable_params = [param for _, param in trainable_named_params]

    level_grad_map: LevelGradMap = {}
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


def _coerce_level_grad_map(
    level_grad_map: Any,
    num_params: int,
) -> Optional[LevelGradMap]:
    if not isinstance(level_grad_map, Mapping):
        return None

    coerced: LevelGradMap = {}
    for level_name in _GRAD_LEVEL_NAMES:
        grads = level_grad_map.get(level_name)
        if not isinstance(grads, (list, tuple)) or len(grads) != int(num_params):
            return None
        coerced_level: List[Optional[torch.Tensor]] = []
        for grad in grads:
            if grad is not None and not isinstance(grad, torch.Tensor):
                return None
            coerced_level.append(grad)
        coerced[level_name] = tuple(coerced_level)
    return coerced


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
    projection_rule: str = "orthogonalize_all",
) -> Tuple[GradTuple, torch.Tensor, torch.Tensor]:
    """Project ``target_grads`` off ``reference_grads`` without host synchronization.

    The coefficient and the applied flag are returned as 0-dim tensors and stay
    on the gradient device, so the projection never blocks on ``.item()``. A
    projection that does not apply resolves to a zero coefficient, and
    ``grad_target - 0 * grad_ref`` is bitwise equal to ``grad_target`` for finite
    references. The trade-off is that a non-applied projection now allocates a
    tuple instead of aliasing ``target_grads``; this matters mainly for
    ``conflict_only``, where non-application is the common case.
    """
    if projection_rule not in _LEX_PROJECTION_RULES:
        raise ValueError(
            f"Unsupported lex projection rule '{projection_rule}'. "
            f"Expected one of {list(_LEX_PROJECTION_RULES)}."
        )

    eps_value = float(eps)
    denom = _dot_from_autograd_grads(reference_grads, reference_grads, include_mask)
    # Both dots skip the same entries, but an empty reduction falls back to a CPU
    # scalar, so align the pair before combining them.
    numer = _dot_from_autograd_grads(target_grads, reference_grads, include_mask).to(
        device=denom.device,
        dtype=denom.dtype,
    )

    applied = denom > eps_value
    if projection_rule == "conflict_only":
        applied = applied & (numer < -eps_value)
    coeff = torch.where(
        applied,
        numer / denom.clamp_min(eps_value),
        torch.zeros((), dtype=denom.dtype, device=denom.device),
    )

    projected: List[Optional[torch.Tensor]] = []
    for grad_target, grad_ref, include in zip(target_grads, reference_grads, include_mask):
        if grad_target is None:
            projected.append(None)
            continue
        if (not include) or grad_ref is None:
            projected.append(grad_target)
            continue
        projected.append(grad_target - (coeff * grad_ref))
    return tuple(projected), coeff, applied


def _sum_grad_tuples(
    grads_a: Sequence[Optional[torch.Tensor]],
    grads_b: Sequence[Optional[torch.Tensor]],
    grads_c: Sequence[Optional[torch.Tensor]],
) -> GradTuple:
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


def _compose_mid_projected_grads(
    mid_grads: Sequence[Optional[torch.Tensor]],
    mid_projected_t2: Sequence[Optional[torch.Tensor]],
    mid_projected_t1: Sequence[Optional[torch.Tensor]],
    t2_mask: Sequence[bool],
    t1_mask: Sequence[bool],
) -> GradTuple:
    composed: List[Optional[torch.Tensor]] = []
    for mid_grad, mid_t2, mid_t1, is_t2, is_t1 in zip(
        mid_grads,
        mid_projected_t2,
        mid_projected_t1,
        t2_mask,
        t1_mask,
    ):
        if is_t2:
            composed.append(mid_t2)
        elif is_t1:
            composed.append(mid_t1)
        else:
            composed.append(mid_grad)
    return tuple(composed)


def _compose_coarse_projected_grads(
    coarse_grads: Sequence[Optional[torch.Tensor]],
    coarse_projected_t2: Sequence[Optional[torch.Tensor]],
    coarse_projected_t1: Sequence[Optional[torch.Tensor]],
    t2_mask: Sequence[bool],
    t1_mask: Sequence[bool],
) -> GradTuple:
    composed: List[Optional[torch.Tensor]] = []
    for coarse_grad, coarse_t2, coarse_t1, is_t2, is_t1 in zip(
        coarse_grads,
        coarse_projected_t2,
        coarse_projected_t1,
        t2_mask,
        t1_mask,
    ):
        if is_t2:
            composed.append(coarse_t2)
        elif is_t1:
            composed.append(coarse_t1)
        else:
            composed.append(coarse_grad)
    return tuple(composed)


def _scale_grad_tuple(
    grads: Sequence[Optional[torch.Tensor]],
    factor: float,
) -> GradTuple:
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


def _trunk_grad_norm_specs(
    coarse_grads: Sequence[Optional[torch.Tensor]],
    mid_grads: Sequence[Optional[torch.Tensor]],
    fine_grads: Sequence[Optional[torch.Tensor]],
    mask_views: Mapping[str, Sequence[bool]],
    prefix: str = "",
) -> List[Tuple[str, Sequence[Optional[torch.Tensor]], Sequence[bool]]]:
    norm_specs: List[
        Tuple[str, Sequence[Optional[torch.Tensor]], Sequence[bool]]
    ] = []

    if any(mask_views["t3t2t1"]):
        norm_specs.append(
            (f"{prefix}grad_norm_t3t2t1_coarse", coarse_grads, mask_views["t3t2t1"])
        )

    if any(mask_views["t3"]):
        norm_specs.append((f"{prefix}grad_norm_t3_coarse", coarse_grads, mask_views["t3"]))

    if any(mask_views["t2t1"]):
        norm_specs.extend(
            [
                (f"{prefix}grad_norm_t2t1_coarse", coarse_grads, mask_views["t2t1"]),
                (f"{prefix}grad_norm_t2t1_mid", mid_grads, mask_views["t2t1"]),
            ]
        )

    if any(mask_views["t2"]):
        norm_specs.extend(
            [
                (f"{prefix}grad_norm_t2_coarse", coarse_grads, mask_views["t2"]),
                (f"{prefix}grad_norm_t2_mid", mid_grads, mask_views["t2"]),
            ]
        )

    if any(mask_views["t1"]):
        norm_specs.extend(
            [
                (f"{prefix}grad_norm_t1_coarse", coarse_grads, mask_views["t1"]),
                (f"{prefix}grad_norm_t1_mid", mid_grads, mask_views["t1"]),
                (f"{prefix}grad_norm_t1_fine", fine_grads, mask_views["t1"]),
            ]
        )

    return norm_specs


def _build_lexicographic_grads(
    coarse_grads: Sequence[Optional[torch.Tensor]],
    mid_grads: Sequence[Optional[torch.Tensor]],
    fine_grads: Sequence[Optional[torch.Tensor]],
    trunk_masks: Mapping[str, Sequence[bool]],
    projection_mode: str = "coarse_first",
    projection_rule: str = "orthogonalize_all",
    eps: float = _GRAD_EPS,
    include_metrics: bool = True,
) -> Tuple[Dict[str, GradTuple], Dict[str, float]]:
    if projection_mode not in _LEX_PROJECTION_MODES:
        raise ValueError(
            f"Unsupported lex projection mode '{projection_mode}'. "
            f"Expected one of {list(_LEX_PROJECTION_MODES)}."
        )
    if projection_rule not in _LEX_PROJECTION_RULES:
        raise ValueError(
            f"Unsupported lex projection rule '{projection_rule}'. "
            f"Expected one of {list(_LEX_PROJECTION_RULES)}."
        )

    t1_mask = list(trunk_masks.get("t1", []))
    t2_mask = list(trunk_masks.get("t2", []))

    # `None` means the projection was never attempted in this mode; anything set
    # below is an on-device 0-dim bool tensor, kept unread to avoid a sync.
    projection_flags: Dict[str, Optional[torch.Tensor]] = {
        key: None for key in _PROJECTION_FLAG_METRICS
    }

    if projection_mode == "coarse_first":
        mid_projected_t2, _mid_coeff_t2, mid_applied_t2 = _project_onto_reference(
            target_grads=mid_grads,
            reference_grads=coarse_grads,
            include_mask=t2_mask,
            eps=eps,
            projection_rule=projection_rule,
        )
        mid_projected_t1, _mid_coeff_t1, mid_applied_t1 = _project_onto_reference(
            target_grads=mid_grads,
            reference_grads=coarse_grads,
            include_mask=t1_mask,
            eps=eps,
            projection_rule=projection_rule,
        )
        mid_projected = _compose_mid_projected_grads(
            mid_grads=mid_grads,
            mid_projected_t2=mid_projected_t2,
            mid_projected_t1=mid_projected_t1,
            t2_mask=t2_mask,
            t1_mask=t1_mask,
        )
        higher_t1 = _sum_grad_tuples(
            coarse_grads,
            mid_projected_t1,
            (None,) * len(coarse_grads),
        )
        fine_projected, _fine_coeff_higher, fine_applied_higher = _project_onto_reference(
            target_grads=fine_grads,
            reference_grads=higher_t1,
            include_mask=t1_mask,
            eps=eps,
            projection_rule=projection_rule,
        )
        coarse_projected = tuple(coarse_grads)
        projection_flags["mid_off_coarse_t2"] = mid_applied_t2
        projection_flags["mid_off_coarse_t1"] = mid_applied_t1
        projection_flags["fine_off_higher_t1"] = fine_applied_higher
    else:
        mid_projected_t1, _mid_coeff_t1, mid_applied_t1 = _project_onto_reference(
            target_grads=mid_grads,
            reference_grads=fine_grads,
            include_mask=t1_mask,
            eps=eps,
            projection_rule=projection_rule,
        )
        mid_projected = _compose_mid_projected_grads(
            mid_grads=mid_grads,
            mid_projected_t2=mid_grads,
            mid_projected_t1=mid_projected_t1,
            t2_mask=t2_mask,
            t1_mask=t1_mask,
        )

        coarse_projected_t2, _coarse_coeff_t2, coarse_applied_t2 = _project_onto_reference(
            target_grads=coarse_grads,
            reference_grads=mid_grads,
            include_mask=t2_mask,
            eps=eps,
            projection_rule=projection_rule,
        )
        higher_t1 = _sum_grad_tuples(
            fine_grads,
            mid_projected_t1,
            (None,) * len(coarse_grads),
        )
        coarse_projected_t1, _coarse_coeff_t1, coarse_applied_t1 = _project_onto_reference(
            target_grads=coarse_grads,
            reference_grads=higher_t1,
            include_mask=t1_mask,
            eps=eps,
            projection_rule=projection_rule,
        )
        coarse_projected = _compose_coarse_projected_grads(
            coarse_grads=coarse_grads,
            coarse_projected_t2=coarse_projected_t2,
            coarse_projected_t1=coarse_projected_t1,
            t2_mask=t2_mask,
            t1_mask=t1_mask,
        )
        fine_projected = tuple(fine_grads)

        projection_flags["mid_off_fine_t1"] = mid_applied_t1
        projection_flags["coarse_off_mid_t2"] = coarse_applied_t2
        projection_flags["coarse_off_higher_t1"] = coarse_applied_t1

    total_grads = _sum_grad_tuples(coarse_projected, mid_projected, fine_projected)
    grad_pack: Dict[str, GradTuple] = {
        "coarse": coarse_projected,
        "mid_projected": mid_projected,
        "fine_projected": fine_projected,
        "total": total_grads,
    }

    if not include_metrics:
        return grad_pack, {}

    t2t1_mask = _merge_masks(t2_mask, t1_mask)
    higher_t1 = _sum_grad_tuples(
        coarse_projected,
        mid_projected,
        (None,) * len(coarse_projected),
    )
    metrics: Dict[str, float] = {
        "lex_projection_mode_coarse_first": 1.0 if projection_mode == "coarse_first" else 0.0,
        "lex_projection_mode_fine_first": 1.0 if projection_mode == "fine_first" else 0.0,
    }
    # Modes that never attempt a projection report 0.0 directly; the rest ride
    # along in the single batched device-to-host transfer below.
    flag_specs: List[Tuple[str, torch.Tensor]] = []
    for flag_key, metric_name in _PROJECTION_FLAG_METRICS.items():
        flag = projection_flags[flag_key]
        if flag is None:
            metrics[metric_name] = 0.0
        else:
            flag_specs.append((metric_name, flag))
    mask_views = _mask_views_from_trunk_masks(trunk_masks)
    metrics.update(
        _batched_grad_metrics(
            norm_specs=_trunk_grad_norm_specs(
                coarse_grads=coarse_projected,
                mid_grads=mid_projected,
                fine_grads=fine_projected,
                mask_views=mask_views,
                prefix="post_",
            ),
            cosine_specs=[
                ("post_cos_t2_mid_proj_coarse", mid_projected, coarse_projected, t2_mask, eps),
                ("post_cos_t1_mid_proj_coarse", mid_projected, coarse_projected, t1_mask, eps),
                (
                    "post_cos_t2t1_mid_proj_coarse",
                    mid_projected,
                    coarse_projected,
                    t2t1_mask,
                    eps,
                ),
                ("post_cos_t1_fine_proj_higher", fine_projected, higher_t1, t1_mask, eps),
                ("post_cos_t1_fine_proj_coarse", fine_projected, coarse_projected, t1_mask, eps),
                ("post_cos_t1_fine_proj_mid_proj", fine_projected, mid_projected, t1_mask, eps),
            ],
            scalar_specs=flag_specs,
        )
    )

    return grad_pack, metrics


def compute_trunk_param_norm_metrics(
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


def prepare_lexicographic_update(
    trainable_named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    level_losses: Sequence[torch.Tensor],
    eps: float = _GRAD_EPS,
    include_metrics: bool = True,
    grad_scale: float = 1.0,
    projection_mode: str = "coarse_first",
    projection_rule: str = "orthogonalize_all",
    precomputed_level_grad_map: Optional[Any] = None,
) -> Tuple[Optional[LexicographicUpdateState], Dict[str, float]]:
    """Build lexicographic grads in unscaled units, optionally scaling returned grads.

    When AMP is active the caller passes the current GradScaler scale as
    ``grad_scale``. Projection coefficients and metrics stay in unscaled units,
    and only the ``"total"`` entry is multiplied by ``grad_scale`` so
    ``GradScaler.step`` can unscale and check it normally. ``"total"`` is the
    only entry the training loop assigns to ``param.grad``; the per-level
    entries are diagnostic and stay in unscaled units.
    """
    level_grad_map = _coerce_level_grad_map(
        level_grad_map=precomputed_level_grad_map,
        num_params=len(trainable_named_params),
    )
    if level_grad_map is None:
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

    projected_grads, metrics = _build_lexicographic_grads(
        coarse_grads=coarse_grads,
        mid_grads=mid_grads,
        fine_grads=fine_grads,
        trunk_masks=trunk_masks,
        projection_mode=projection_mode,
        projection_rule=projection_rule,
        eps=eps,
        include_metrics=include_metrics,
    )

    safe_scale = float(grad_scale) if float(grad_scale) > 0.0 else 1.0
    projected_grads_for_step = dict(projected_grads)
    projected_grads_for_step["total"] = _scale_grad_tuple(
        projected_grads["total"],
        safe_scale,
    )

    state = LexicographicUpdateState(
        trunk_masks=trunk_masks,
        level_grad_map=level_grad_map,
        projected_grads=projected_grads_for_step,
    )
    return state, metrics


def compute_trunk_grad_metrics(
    trainable_named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    level_losses: Sequence[torch.Tensor],
    retain_graph: bool = True,
) -> Tuple[Optional[TrunkGradState], Dict[str, float]]:
    level_grad_map = _compute_level_grad_map(
        trainable_named_params=trainable_named_params,
        level_losses=level_losses,
        retain_graph=bool(retain_graph),
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

    raw_higher_t1 = _sum_grad_tuples(
        coarse_grads,
        mid_grads,
        (None,) * len(coarse_grads),
    )
    metrics = _batched_grad_metrics(
        norm_specs=_trunk_grad_norm_specs(
            coarse_grads=coarse_grads,
            mid_grads=mid_grads,
            fine_grads=fine_grads,
            mask_views=mask_views,
        ),
        cosine_specs=[
            ("cos_t2_mid_coarse", mid_grads, coarse_grads, mask_views["t2"], _GRAD_EPS),
            ("cos_t1_mid_coarse", mid_grads, coarse_grads, mask_views["t1"], _GRAD_EPS),
            (
                "cos_t2t1_mid_coarse",
                mid_grads,
                coarse_grads,
                mask_views["t2t1"],
                _GRAD_EPS,
            ),
            ("cos_t1_fine_higher", fine_grads, raw_higher_t1, mask_views["t1"], _GRAD_EPS),
            ("cos_t1_fine_coarse", fine_grads, coarse_grads, mask_views["t1"], _GRAD_EPS),
            ("cos_t1_fine_mid", fine_grads, mid_grads, mask_views["t1"], _GRAD_EPS),
        ],
    )

    if not metrics:
        return None, {}

    state = TrunkGradState(
        trunk_masks=trunk_masks,
        level_grad_map=level_grad_map,
    )
    return state, metrics
