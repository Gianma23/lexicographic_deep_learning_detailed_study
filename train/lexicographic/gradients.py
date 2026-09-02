from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .types import (
    DEFAULT_GRADIENT_BLOCKS,
    GRADIENT_BLOCK_NAMES,
    GradTuple,
    LevelGradMap,
    LexicographicUpdateState,
    TrunkGradState,
)

_GRAD_EPS = 1e-12
_GRAD_LEVEL_NAMES = ("coarse", "mid", "fine")
_LEX_PROJECTION_MODES = ("coarse_first", "fine_first")
_BLOCK_LEVELS = {
    block: tuple(_GRAD_LEVEL_NAMES[int(index) - 1] for index in block[1:])
    for block in GRADIENT_BLOCK_NAMES
}
_LEGACY_EXACT_BLOCK_ALIASES = {"p123": "t1", "p12": "t2", "p1": "t3"}
_CANONICAL_PAIR_ORDER = (
    ("mid", "coarse"),
    ("fine", "coarse"),
    ("fine", "mid"),
)
_LEGACY_PROJECTION_FLAG_NAMES = (
    "post_projection_applied_t2_mid_coarse",
    "post_projection_applied_t1_mid_coarse",
    "post_projection_applied_t1_fine_higher",
    "post_projection_applied_t1_mid_fine",
    "post_projection_applied_t2_coarse_mid",
    "post_projection_applied_t1_coarse_higher",
)


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
    blocks: Sequence[str] = DEFAULT_GRADIENT_BLOCKS,
) -> Dict[str, List[bool]]:
    selected_blocks = tuple(blocks)
    masks = {block: [] for block in selected_blocks}
    for coarse_grad, mid_grad, fine_grad in zip(coarse_grads, mid_grads, fine_grads):
        active = (coarse_grad is not None, mid_grad is not None, fine_grad is not None)
        support = "p" + "".join(
            str(level_index + 1)
            for level_index, is_active in enumerate(active)
            if is_active
        )
        for block in selected_blocks:
            masks[block].append(support == block)

    # Keep the historical exact-block names available to internal callers and
    # mixed old/new analysis code. New metrics use only the canonical p-names.
    for block, legacy_name in _LEGACY_EXACT_BLOCK_ALIASES.items():
        if block in masks:
            masks[legacy_name] = list(masks[block])
    return masks


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
    num_params = max((len(mask) for mask in trunk_masks.values()), default=0)

    def exact_mask(canonical_name: str, legacy_name: str) -> List[bool]:
        mask = list(trunk_masks.get(canonical_name, trunk_masks.get(legacy_name, [])))
        return mask if mask else [False] * num_params

    t1_mask = exact_mask("p123", "t1")
    t2_mask = exact_mask("p12", "t2")
    t3_mask = exact_mask("p1", "t3")
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
) -> Tuple[GradTuple, torch.Tensor, torch.Tensor]:
    """Project ``target_grads`` off ``reference_grads`` without host synchronization.

    The coefficient and the applied flag are returned as 0-dim tensors and stay
    on the gradient device, so the projection never blocks on ``.item()``. A
    projection that does not apply resolves to a zero coefficient, and
    ``grad_target - 0 * grad_ref`` is bitwise equal to ``grad_target`` for finite
    references.
    """
    eps_value = float(eps)
    denom = _dot_from_autograd_grads(reference_grads, reference_grads, include_mask)
    # Both dots skip the same entries, but an empty reduction falls back to a CPU
    # scalar, so align the pair before combining them.
    numer = _dot_from_autograd_grads(target_grads, reference_grads, include_mask).to(
        device=denom.device,
        dtype=denom.dtype,
    )

    applied = denom > eps_value
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


def _sum_grad_sequences(
    grad_sequences: Sequence[Sequence[Optional[torch.Tensor]]],
) -> GradTuple:
    if not grad_sequences:
        return tuple()
    num_params = len(grad_sequences[0])
    summed: List[Optional[torch.Tensor]] = []
    for param_index in range(num_params):
        pieces = [
            grads[param_index]
            for grads in grad_sequences
            if grads[param_index] is not None
        ]
        if not pieces:
            summed.append(None)
            continue
        total = pieces[0]
        for piece in pieces[1:]:
            total = total + piece
        summed.append(total)
    return tuple(summed)


def _canonical_grad_norm_specs(
    level_grads: Mapping[str, Sequence[Optional[torch.Tensor]]],
    block_masks: Mapping[str, Sequence[bool]],
    blocks: Sequence[str],
    prefix: str = "",
) -> List[Tuple[str, Sequence[Optional[torch.Tensor]], Sequence[bool]]]:
    specs: List[Tuple[str, Sequence[Optional[torch.Tensor]], Sequence[bool]]] = []
    for block in blocks:
        mask = list(block_masks.get(block, []))
        if not mask or not any(mask):
            continue
        for level_name in _BLOCK_LEVELS[block]:
            specs.append(
                (f"{prefix}grad_norm_{block}_{level_name}", level_grads[level_name], mask)
            )
    return specs


def _canonical_pairwise_cosine_specs(
    level_grads: Mapping[str, Sequence[Optional[torch.Tensor]]],
    block_masks: Mapping[str, Sequence[bool]],
    blocks: Sequence[str],
    eps: float,
    prefix: str = "",
) -> List[
    Tuple[
        str,
        Sequence[Optional[torch.Tensor]],
        Sequence[Optional[torch.Tensor]],
        Sequence[bool],
        float,
    ]
]:
    specs = []
    for block in blocks:
        mask = list(block_masks.get(block, []))
        if not mask or not any(mask):
            continue
        active_levels = set(_BLOCK_LEVELS[block])
        for target_name, reference_name in _CANONICAL_PAIR_ORDER:
            if target_name not in active_levels or reference_name not in active_levels:
                continue
            specs.append(
                (
                    f"{prefix}cos_{block}_{target_name}_{reference_name}",
                    level_grads[target_name],
                    level_grads[reference_name],
                    mask,
                    eps,
                )
            )
    return specs


def _legacy_projection_flag_alias(
    block: str,
    target_name: str,
    reference_name: str,
) -> Optional[str]:
    aliases = {
        ("p12", "mid", "coarse"): "post_projection_applied_t2_mid_coarse",
        ("p123", "mid", "coarse"): "post_projection_applied_t1_mid_coarse",
        ("p123", "fine", "higher"): "post_projection_applied_t1_fine_higher",
        ("p123", "mid", "fine"): "post_projection_applied_t1_mid_fine",
        ("p12", "coarse", "mid"): "post_projection_applied_t2_coarse_mid",
        ("p123", "coarse", "higher"): "post_projection_applied_t1_coarse_higher",
    }
    return aliases.get((block, target_name, reference_name))


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
    eps: float = _GRAD_EPS,
    include_metrics: bool = True,
    blocks: Sequence[str] = DEFAULT_GRADIENT_BLOCKS,
) -> Tuple[Dict[str, GradTuple], Dict[str, float]]:
    if projection_mode not in _LEX_PROJECTION_MODES:
        raise ValueError(
            f"Unsupported lex projection mode '{projection_mode}'. "
            f"Expected one of {list(_LEX_PROJECTION_MODES)}."
        )
    selected_blocks = tuple(blocks)
    projected_by_level: Dict[str, GradTuple] = {
        "coarse": tuple(coarse_grads),
        "mid": tuple(mid_grads),
        "fine": tuple(fine_grads),
    }
    priority = (
        ("coarse", "mid", "fine")
        if projection_mode == "coarse_first"
        else ("fine", "mid", "coarse")
    )
    # block, target, reference label, mask, reference gradients, applied flag
    projection_records: List[
        Tuple[str, str, str, Sequence[bool], GradTuple, torch.Tensor]
    ] = []

    for block in selected_blocks:
        mask = list(trunk_masks.get(block, []))
        if not mask or not any(mask):
            continue
        active_levels = set(_BLOCK_LEVELS[block])
        ordered_active = [name for name in priority if name in active_levels]
        for target_index in range(1, len(ordered_active)):
            target_name = ordered_active[target_index]
            higher_names = ordered_active[:target_index]
            # Gram-Schmidt: remove the target's component along each
            # higher-priority gradient in turn. Earlier iterations already made
            # those gradients mutually orthogonal on this mask, so the target
            # lands in the orthogonal complement of their span and
            # `<total, g_h> == ||g_h||^2` holds for every higher-priority level
            # `h`, which is the lexicographic guarantee.
            #
            # Projecting off their resultant instead only enforces orthogonality
            # to the sum. That leaves `<target_proj, g_coarse>` equal to
            # `-<target_proj, g_mid_proj>`, free to be negative, so the step can
            # ascend the priority objective -- observed on HRN CIFAR-100, where
            # `|g_fine| / |g_coarse| ~ 5` made the leak large enough to invert
            # coarse-first training for its first ~45 epochs.
            step_applied: List[torch.Tensor] = []
            for reference_name in higher_names:
                reference_grads = projected_by_level[reference_name]
                projected_target, _coefficient, applied = _project_onto_reference(
                    target_grads=projected_by_level[target_name],
                    reference_grads=reference_grads,
                    include_mask=mask,
                    eps=eps,
                )
                projected_by_level[target_name] = projected_target
                step_applied.append(applied)
                projection_records.append(
                    (block, target_name, reference_name, mask, reference_grads, applied)
                )
            if include_metrics and len(higher_names) > 1:
                # Aggregate `higher` record kept so readers of the pre-Gram-Schmidt
                # logs keep a flag and a resultant cosine for the last target.
                combined_applied = step_applied[0]
                for applied in step_applied[1:]:
                    combined_applied = combined_applied & applied
                projection_records.append(
                    (
                        block,
                        target_name,
                        "higher",
                        mask,
                        _sum_grad_sequences(
                            [projected_by_level[name] for name in higher_names]
                        ),
                        combined_applied,
                    )
                )

    coarse_projected = projected_by_level["coarse"]
    mid_projected = projected_by_level["mid"]
    fine_projected = projected_by_level["fine"]
    total_grads = _sum_grad_tuples(coarse_projected, mid_projected, fine_projected)
    grad_pack: Dict[str, GradTuple] = {
        "coarse": coarse_projected,
        "mid_projected": mid_projected,
        "fine_projected": fine_projected,
        "total": total_grads,
    }

    if not include_metrics:
        return grad_pack, {}

    metrics: Dict[str, float] = {
        "lex_projection_mode_coarse_first": 1.0 if projection_mode == "coarse_first" else 0.0,
        "lex_projection_mode_fine_first": 1.0 if projection_mode == "fine_first" else 0.0,
    }
    flag_specs: List[Tuple[str, torch.Tensor]] = []
    higher_cosine_specs = []
    emitted_legacy_flags = set()
    for block, target_name, reference_name, mask, reference_grads, applied in projection_records:
        canonical_flag = f"post_projection_applied_{block}_{target_name}_{reference_name}"
        flag_specs.append((canonical_flag, applied))
        legacy_flag = _legacy_projection_flag_alias(block, target_name, reference_name)
        if legacy_flag is not None:
            flag_specs.append((legacy_flag, applied))
            emitted_legacy_flags.add(legacy_flag)
        if reference_name == "higher":
            higher_cosine_specs.append(
                (
                    f"post_cos_{block}_{target_name}_higher",
                    projected_by_level[target_name],
                    reference_grads,
                    mask,
                    eps,
                )
            )

    for legacy_flag in _LEGACY_PROJECTION_FLAG_NAMES:
        if legacy_flag not in emitted_legacy_flags:
            metrics[legacy_flag] = 0.0

    mask_views = _mask_views_from_trunk_masks(trunk_masks)
    t1_mask = mask_views["t1"]
    t2_mask = mask_views["t2"]
    t2t1_mask = mask_views["t2t1"]
    legacy_higher_t1 = _sum_grad_tuples(
        coarse_projected,
        mid_projected,
        (None,) * len(coarse_projected),
    )
    projected_level_grads = {
        "coarse": coarse_projected,
        "mid": mid_projected,
        "fine": fine_projected,
    }
    metrics.update(
        _batched_grad_metrics(
            norm_specs=(
                _canonical_grad_norm_specs(
                    level_grads=projected_level_grads,
                    block_masks=trunk_masks,
                    blocks=selected_blocks,
                    prefix="post_",
                )
                + _trunk_grad_norm_specs(
                    coarse_grads=coarse_projected,
                    mid_grads=mid_projected,
                    fine_grads=fine_projected,
                    mask_views=mask_views,
                    prefix="post_",
                )
            ),
            cosine_specs=(
                _canonical_pairwise_cosine_specs(
                    level_grads=projected_level_grads,
                    block_masks=trunk_masks,
                    blocks=selected_blocks,
                    eps=eps,
                    prefix="post_",
                )
                + higher_cosine_specs
                + [
                    (
                        "post_cos_t2_mid_proj_coarse",
                        mid_projected,
                        coarse_projected,
                        t2_mask,
                        eps,
                    ),
                    (
                        "post_cos_t1_mid_proj_coarse",
                        mid_projected,
                        coarse_projected,
                        t1_mask,
                        eps,
                    ),
                    (
                        "post_cos_t2t1_mid_proj_coarse",
                        mid_projected,
                        coarse_projected,
                        t2t1_mask,
                        eps,
                    ),
                    (
                        "post_cos_t1_fine_proj_higher",
                        fine_projected,
                        legacy_higher_t1,
                        t1_mask,
                        eps,
                    ),
                    (
                        "post_cos_t1_fine_proj_coarse",
                        fine_projected,
                        coarse_projected,
                        t1_mask,
                        eps,
                    ),
                    (
                        "post_cos_t1_fine_proj_mid_proj",
                        fine_projected,
                        mid_projected,
                        t1_mask,
                        eps,
                    ),
                ]
            ),
            scalar_specs=flag_specs,
        )
    )

    return grad_pack, metrics


def compute_trunk_param_norm_metrics(
    params: Sequence[torch.nn.Parameter],
    start_snapshot: Sequence[torch.Tensor],
    trunk_masks: Optional[Mapping[str, Sequence[bool]]],
    blocks: Sequence[str] = DEFAULT_GRADIENT_BLOCKS,
) -> Dict[str, float]:
    if trunk_masks is None:
        return {}

    mask_views = _mask_views_from_trunk_masks(trunk_masks)
    metrics: Dict[str, float] = {}
    for block in blocks:
        mask = list(trunk_masks.get(block, []))
        if not mask or not any(mask):
            continue
        metrics[f"param_norm_{block}"] = _param_norm_from_values(params, mask)
        metrics[f"delta_param_norm_{block}"] = _delta_param_norm_from_snapshot(
            params=params,
            start_snapshot=start_snapshot,
            include_mask=mask,
        )

    # Deprecated aliases remain emitted so existing notebooks can consume new
    # runs while historical run_log.jsonl files remain untouched.
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
    blocks: Sequence[str] = DEFAULT_GRADIENT_BLOCKS,
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
        blocks=blocks,
    )

    projected_grads, metrics = _build_lexicographic_grads(
        coarse_grads=coarse_grads,
        mid_grads=mid_grads,
        fine_grads=fine_grads,
        trunk_masks=trunk_masks,
        projection_mode=projection_mode,
        eps=eps,
        include_metrics=include_metrics,
        blocks=blocks,
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
    blocks: Sequence[str] = DEFAULT_GRADIENT_BLOCKS,
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
        blocks=blocks,
    )
    mask_views = _mask_views_from_trunk_masks(trunk_masks)

    raw_higher_t1 = _sum_grad_tuples(
        coarse_grads,
        mid_grads,
        (None,) * len(coarse_grads),
    )
    level_grads = {"coarse": coarse_grads, "mid": mid_grads, "fine": fine_grads}
    metrics = _batched_grad_metrics(
        norm_specs=(
            _canonical_grad_norm_specs(
                level_grads=level_grads,
                block_masks=trunk_masks,
                blocks=blocks,
            )
            + _trunk_grad_norm_specs(
                coarse_grads=coarse_grads,
                mid_grads=mid_grads,
                fine_grads=fine_grads,
                mask_views=mask_views,
            )
        ),
        cosine_specs=(
            _canonical_pairwise_cosine_specs(
                level_grads=level_grads,
                block_masks=trunk_masks,
                blocks=blocks,
                eps=_GRAD_EPS,
            )
            + [
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
            ]
        ),
    )

    if not metrics:
        return None, {}

    state = TrunkGradState(
        trunk_masks=trunk_masks,
        level_grad_map=level_grad_map,
    )
    return state, metrics
