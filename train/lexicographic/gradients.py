from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .types import GradTuple, LevelGradMap, LexicographicUpdateState, TrunkGradState

_GRAD_EPS = 1e-12
_GRAD_LEVEL_NAMES = ("coarse", "mid", "fine")
_LEX_PROJECTION_MODES = ("coarse_first", "fine_first", "pairwise_orthogonal")
_LEX_PROJECTION_RULES = ("orthogonalize_all", "conflict_only")


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
) -> Tuple[GradTuple, float, bool]:
    if projection_rule not in _LEX_PROJECTION_RULES:
        raise ValueError(
            f"Unsupported lex projection rule '{projection_rule}'. "
            f"Expected one of {list(_LEX_PROJECTION_RULES)}."
        )

    denom = _dot_from_autograd_grads(reference_grads, reference_grads, include_mask)
    denom_value = float(denom.item())

    if denom_value <= float(eps):
        return tuple(target_grads), 0.0, False

    numer = _dot_from_autograd_grads(target_grads, reference_grads, include_mask)
    numer_value = float(numer.item())
    conflict = numer_value < -float(eps)
    if projection_rule == "conflict_only" and not conflict:
        return tuple(target_grads), 0.0, False

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


def _scale_grad_pack(
    grad_pack: Mapping[str, Sequence[Optional[torch.Tensor]]],
    factor: float,
) -> Dict[str, GradTuple]:
    factor_f = float(factor)
    return {str(key): _scale_grad_tuple(grads, factor_f) for key, grads in grad_pack.items()}


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

    projection_flags: Dict[str, bool] = {
        "mid_off_coarse_t2": False,
        "mid_off_coarse_t1": False,
        "fine_off_higher_t1": False,
        "mid_off_fine_t1": False,
        "coarse_off_mid_t2": False,
        "coarse_off_higher_t1": False,
        "fine_off_coarse_t1": False,
        "fine_off_mid_t1": False,
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
        projection_flags["mid_off_coarse_t2"] = bool(mid_applied_t2)
        projection_flags["mid_off_coarse_t1"] = bool(mid_applied_t1)
        projection_flags["fine_off_higher_t1"] = bool(fine_applied_higher)
    elif projection_mode == "fine_first":
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

        projection_flags["mid_off_fine_t1"] = bool(mid_applied_t1)
        projection_flags["coarse_off_mid_t2"] = bool(coarse_applied_t2)
        projection_flags["coarse_off_higher_t1"] = bool(coarse_applied_t1)
    else:
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

        fine_projected_coarse_t1, _fine_coeff_coarse, fine_applied_coarse = _project_onto_reference(
            target_grads=fine_grads,
            reference_grads=coarse_grads,
            include_mask=t1_mask,
            eps=eps,
            projection_rule=projection_rule,
        )
        fine_projected, _fine_coeff_mid, fine_applied_mid = _project_onto_reference(
            target_grads=fine_projected_coarse_t1,
            reference_grads=mid_projected_t1,
            include_mask=t1_mask,
            eps=eps,
            projection_rule=projection_rule,
        )
        coarse_projected = tuple(coarse_grads)

        projection_flags["mid_off_coarse_t2"] = bool(mid_applied_t2)
        projection_flags["mid_off_coarse_t1"] = bool(mid_applied_t1)
        projection_flags["fine_off_coarse_t1"] = bool(fine_applied_coarse)
        projection_flags["fine_off_mid_t1"] = bool(fine_applied_mid)

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
        "lex_projection_mode_pairwise_orthogonal": 1.0 if projection_mode == "pairwise_orthogonal" else 0.0,
        "post_projection_applied_t2_mid_coarse": 1.0 if projection_flags["mid_off_coarse_t2"] else 0.0,
        "post_projection_applied_t1_mid_coarse": 1.0 if projection_flags["mid_off_coarse_t1"] else 0.0,
        "post_projection_applied_t1_fine_higher": 1.0 if projection_flags["fine_off_higher_t1"] else 0.0,
        "post_projection_applied_t1_mid_fine": 1.0 if projection_flags["mid_off_fine_t1"] else 0.0,
        "post_projection_applied_t2_coarse_mid": 1.0 if projection_flags["coarse_off_mid_t2"] else 0.0,
        "post_projection_applied_t1_coarse_higher": 1.0 if projection_flags["coarse_off_higher_t1"] else 0.0,
        "post_projection_applied_t1_fine_coarse": 1.0 if projection_flags["fine_off_coarse_t1"] else 0.0,
        "post_projection_applied_t1_fine_mid_proj": 1.0 if projection_flags["fine_off_mid_t1"] else 0.0,
    }
    metrics["post_cos_t2_mid_proj_coarse"] = _grad_cosine_from_autograd_grads(
        mid_projected,
        coarse_projected,
        t2_mask,
        eps=eps,
    )
    metrics["post_cos_t1_mid_proj_coarse"] = _grad_cosine_from_autograd_grads(
        mid_projected,
        coarse_projected,
        t1_mask,
        eps=eps,
    )
    metrics["post_cos_t2t1_mid_proj_coarse"] = _grad_cosine_from_autograd_grads(
        mid_projected,
        coarse_projected,
        t2t1_mask,
        eps=eps,
    )
    metrics["post_cos_t1_fine_proj_higher"] = _grad_cosine_from_autograd_grads(
        fine_projected,
        higher_t1,
        t1_mask,
        eps=eps,
    )
    metrics["post_cos_t1_fine_proj_coarse"] = _grad_cosine_from_autograd_grads(
        fine_projected,
        coarse_projected,
        t1_mask,
        eps=eps,
    )
    metrics["post_cos_t1_fine_proj_mid_proj"] = _grad_cosine_from_autograd_grads(
        fine_projected,
        mid_projected,
        t1_mask,
        eps=eps,
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
    while returned gradients are multiplied by ``grad_scale`` so
    ``GradScaler.step`` can unscale and check them normally.
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

    metrics: Dict[str, float] = {}
    projected_grads, _ = _build_lexicographic_grads(
        coarse_grads=coarse_grads,
        mid_grads=mid_grads,
        fine_grads=fine_grads,
        trunk_masks=trunk_masks,
        projection_mode=projection_mode,
        projection_rule=projection_rule,
        eps=eps,
        include_metrics=False,
    )

    if include_metrics:
        projected_for_log, lex_metrics = _build_lexicographic_grads(
            coarse_grads=coarse_grads,
            mid_grads=mid_grads,
            fine_grads=fine_grads,
            trunk_masks=trunk_masks,
            projection_mode=projection_mode,
            projection_rule=projection_rule,
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

    safe_scale = float(grad_scale) if float(grad_scale) > 0.0 else 1.0
    projected_grads_for_step = _scale_grad_pack(projected_grads, safe_scale)

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

    metrics: Dict[str, float] = _trunk_grad_norm_metrics(
        coarse_grads=coarse_grads,
        mid_grads=mid_grads,
        fine_grads=fine_grads,
        mask_views=mask_views,
    )

    metrics["cos_t2_mid_coarse"] = _grad_cosine_from_autograd_grads(
        mid_grads,
        coarse_grads,
        mask_views["t2"],
        eps=_GRAD_EPS,
    )
    metrics["cos_t1_mid_coarse"] = _grad_cosine_from_autograd_grads(
        mid_grads,
        coarse_grads,
        mask_views["t1"],
        eps=_GRAD_EPS,
    )
    metrics["cos_t2t1_mid_coarse"] = _grad_cosine_from_autograd_grads(
        mid_grads,
        coarse_grads,
        mask_views["t2t1"],
        eps=_GRAD_EPS,
    )
    raw_higher_t1 = _sum_grad_tuples(
        coarse_grads,
        mid_grads,
        (None,) * len(coarse_grads),
    )
    metrics["cos_t1_fine_higher"] = _grad_cosine_from_autograd_grads(
        fine_grads,
        raw_higher_t1,
        mask_views["t1"],
        eps=_GRAD_EPS,
    )
    metrics["cos_t1_fine_coarse"] = _grad_cosine_from_autograd_grads(
        fine_grads,
        coarse_grads,
        mask_views["t1"],
        eps=_GRAD_EPS,
    )
    metrics["cos_t1_fine_mid"] = _grad_cosine_from_autograd_grads(
        fine_grads,
        mid_grads,
        mask_views["t1"],
        eps=_GRAD_EPS,
    )

    if not metrics:
        return None, {}

    state = TrunkGradState(
        trunk_masks=trunk_masks,
        level_grad_map=level_grad_map,
    )
    return state, metrics
