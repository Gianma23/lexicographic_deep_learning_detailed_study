"""Shared subspace-norm scoring and direct supervision of the decoded scores.

The loss is capability based: any model can opt in by exposing
``subspace_scores_per_level`` and ``subspace_path_overlap_by_level`` in its
forward output. No model-family check is performed here.

The supervised quantity is the exact per-level taxonomy-subspace norm the
decoder ranks. Its ground truth is not a one-hot label. A subspace contains the
ancestors, the node itself and its descendants, so two classes sharing an
ancestor necessarily share the corresponding coordinates: the score of a sibling
of the correct class is bounded below by the energy on the shared ancestors, and
a one-hot target is therefore infeasible. The attainable target is the score
profile induced by the intended node geometry -- unit energy spread evenly over
the ground-truth path -- which assigns class ``c`` the square root of the
fraction of that path its subspace contains. Both the score
and that profile are put on one common scale -- the norm of the node coordinate
vector, shared by every level -- and compared through a tempered soft
cross-entropy, so the objective is minimized exactly at the intended geometry
rather than at a degenerate one. The scale must be shared: normalizing each
level separately frees one scale per level and leaves the node energies
unidentified, which shows up as the recovered geometry drifting off the
ground-truth path whenever the target is not sharply peaked.

The target carries no level weighting. A class's target is the square root of the
fraction of the ground-truth path its subspace contains, which is fixed by the
taxonomy alone: the intended geometry spreads unit energy evenly over the path,
so a competitor's score is bounded below by the share of the path it shares. The
level weights are spent where the model family's own objective spends them --
`model.weight_mode` resolves the per-level coefficients of the scalarisation,
exactly as it does for the native Hier-COS softmax losses -- and they are
normalized to sum to one, so the loss keeps the scale of the previous uniform
mean and learning rates transfer from the Hier-COS baselines unchanged.

Separating the two roles is what keeps the arm interpretable. The temperature is
then the only control over the target's margins, and `model.weight_mode` means
the same thing here as in the baseline the arm is measured against.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


SUBSPACE_SCORES_KEY = "subspace_scores_per_level"
SUBSPACE_PATH_OVERLAP_KEY = "subspace_path_overlap_by_level"
SUBSPACE_COORDINATES_KEY = "node_logits"
DEFAULT_TAU = 0.25


@dataclass(frozen=True)
class SubspaceSupervisionConfig:
    enabled: bool = False
    tau: float = DEFAULT_TAU
    eps: float = 1.0e-12


def _section_to_dict(section: Any) -> Dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, Mapping):
        return dict(section)
    if hasattr(section, "items"):
        return {key: value for key, value in section.items()}
    return {}


def resolve_subspace_supervision_config(cfg: Any) -> SubspaceSupervisionConfig:
    train_cfg = _section_to_dict(getattr(cfg, "train", None))
    if not train_cfg and hasattr(cfg, "get"):
        train_cfg = _section_to_dict(cfg.get("train", None))
    raw_cfg = _section_to_dict(train_cfg.get("subspace_supervision", None))

    enabled = raw_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("train.subspace_supervision.enabled must be a boolean.")

    raw_tau = raw_cfg.get("tau", DEFAULT_TAU)
    if isinstance(raw_tau, bool):
        raise ValueError("train.subspace_supervision.tau must be a finite number > 0.")
    try:
        tau = float(raw_tau)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "train.subspace_supervision.tau must be a finite number > 0."
        ) from exc
    if not isfinite(tau) or tau <= 0.0:
        raise ValueError("train.subspace_supervision.tau must be a finite number > 0.")

    raw_eps = raw_cfg.get("eps", 1.0e-12)
    if isinstance(raw_eps, bool):
        raise ValueError("train.subspace_supervision.eps must be a finite number > 0.")
    try:
        eps = float(raw_eps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "train.subspace_supervision.eps must be a finite number > 0."
        ) from exc
    if not isfinite(eps) or eps <= 0.0:
        raise ValueError("train.subspace_supervision.eps must be a finite number > 0.")

    return SubspaceSupervisionConfig(enabled=enabled, tau=tau, eps=eps)


def subspace_supervision_enabled(cfg: Any) -> bool:
    return resolve_subspace_supervision_config(cfg).enabled


class _SqrtWithZeroSubgradient(torch.autograd.Function):
    """Square root with the exact forward value and a zero derivative at zero."""

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        nonnegative = value.clamp_min(0.0)
        root = torch.sqrt(nonnegative)
        ctx.save_for_backward(root, value > 0.0)
        return root

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        root, positive = ctx.saved_tensors
        denominator = torch.where(positive, 2.0 * root, torch.ones_like(root))
        grad_input = torch.where(
            positive,
            grad_output / denominator,
            torch.zeros_like(grad_output),
        )
        return (grad_input,)


def subspace_norms(
    node_coordinates: torch.Tensor,
    level_subspace_masks: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Compute taxonomy-subspace L2 norms with stable training gradients.

    Half and bfloat16 coordinates are accumulated in float32. The forward
    values remain the exact ``sqrt(sum(coordinate**2))`` scores; only the
    undefined derivative at an exactly zero norm is defined as zero.
    """
    if not isinstance(node_coordinates, torch.Tensor) or node_coordinates.ndim != 2:
        raise ValueError("Subspace norms require node coordinates with shape [B, N].")
    if not isinstance(level_subspace_masks, (list, tuple)) or not level_subspace_masks:
        raise ValueError("Subspace norms require a non-empty list of level masks.")

    compute_dtype = (
        torch.float32
        if node_coordinates.dtype in {torch.float16, torch.bfloat16}
        else node_coordinates.dtype
    )
    autocast_off = nullcontext()
    if node_coordinates.device.type in {"cpu", "cuda", "xpu", "mps"}:
        autocast_off = torch.autocast(
            device_type=node_coordinates.device.type,
            enabled=False,
        )

    scores_per_level: List[torch.Tensor] = []
    with autocast_off:
        coordinates = node_coordinates.to(dtype=compute_dtype)
        squared = coordinates.pow(2)
        for level, mask in enumerate(level_subspace_masks):
            if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
                raise ValueError(f"Subspace mask for level {level} must have shape [C, N].")
            if int(mask.size(1)) != int(node_coordinates.size(1)):
                raise ValueError(
                    f"Subspace mask for level {level} has width {int(mask.size(1))}; "
                    f"expected {int(node_coordinates.size(1))}."
                )
            device_mask = mask.to(device=node_coordinates.device, dtype=compute_dtype)
            score_sq = torch.matmul(
                squared,
                device_mask.transpose(0, 1).contiguous(),
            )
            scores_per_level.append(_SqrtWithZeroSubgradient.apply(score_sq))
    return scores_per_level


def subspace_target_profile(
    path_overlap: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    """Attainable subspace-score target for one level.

    ``path_overlap[leaf, c]`` counts how many of the ``depth`` levels of
    ``leaf``'s ground-truth path fall inside class ``c``'s taxonomy subspace: the
    whole path for the correct class, and only the levels shared down to the
    lowest common ancestor for any other. Spreading unit energy evenly over the
    path, the induced score is the square root of that shared fraction, so the
    correct class scores 1 and every other class scores the square root of the
    fraction of the path its subspace contains. The target therefore depends on
    the taxonomy alone. Nodes on a single-child chain span one subspace and
    receive the same target, which is the one distinction the subspace map
    cannot express.
    """
    if not isinstance(path_overlap, torch.Tensor) or path_overlap.ndim != 2:
        raise ValueError("Subspace path overlap must have shape [num_leaf, C].")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise ValueError("Subspace target depth must be a positive integer.")
    overlap = path_overlap.to(dtype=torch.long)
    if bool((overlap < 0).any()) or bool((overlap > depth).any()):
        raise ValueError(
            f"Subspace path overlap counts must lie in [0, {depth}]."
        )

    shared_fraction = torch.arange(
        depth + 1,
        device=overlap.device,
        dtype=torch.float32,
    ) / float(depth)
    return torch.sqrt(shared_fraction[overlap])


def _hard_targets(targets: Any, depth: int) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        hard_targets = targets
    elif isinstance(targets, Mapping):
        soft_target_keys = ("soft_targets_per_level", "labels_b", "lam", "lambda")
        if any(targets.get(key) is not None for key in soft_target_keys):
            raise ValueError(
                "Direct subspace supervision does not support soft targets; "
                "disable mixup and cutmix."
            )
        hard_targets = targets.get("hard_targets")
    else:
        hard_targets = None

    if not isinstance(hard_targets, torch.Tensor):
        raise TypeError("Direct subspace supervision requires hard targets with shape [B, L].")
    if hard_targets.ndim != 2 or int(hard_targets.size(1)) != int(depth):
        raise ValueError(
            f"Direct subspace supervision expected targets [B, {depth}], "
            f"got {tuple(hard_targets.shape)}."
        )
    return hard_targets.long()


def _validate_scores(output: Mapping[str, Any]) -> List[torch.Tensor]:
    if not isinstance(output, Mapping):
        raise TypeError("Direct subspace supervision requires model output as a mapping.")
    scores = output.get(SUBSPACE_SCORES_KEY)
    if not isinstance(scores, (list, tuple)) or not scores:
        raise ValueError(
            "Enabled direct subspace supervision requires model output "
            f"`{SUBSPACE_SCORES_KEY}` as a non-empty level-aligned list."
        )

    validated_scores: List[torch.Tensor] = []
    batch_size: Optional[int] = None
    for level, level_scores in enumerate(scores):
        if not isinstance(level_scores, torch.Tensor) or level_scores.ndim != 2:
            raise ValueError(f"Subspace scores for level {level} must have shape [B, C].")
        if not level_scores.is_floating_point():
            raise ValueError(f"Subspace scores for level {level} must be floating point.")
        if int(level_scores.size(0)) <= 0 or int(level_scores.size(1)) <= 0:
            raise ValueError(f"Subspace scores for level {level} cannot be empty.")
        if batch_size is None:
            batch_size = int(level_scores.size(0))
        elif int(level_scores.size(0)) != batch_size:
            raise ValueError("Subspace score levels must share one batch size.")
        if not bool(torch.isfinite(level_scores).all()):
            raise ValueError(f"Subspace scores for level {level} contain non-finite values.")
        validated_scores.append(level_scores)
    return validated_scores


def _validate_path_overlap(
    output: Mapping[str, Any],
    scores: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    path_overlap = output.get(SUBSPACE_PATH_OVERLAP_KEY)
    if not isinstance(path_overlap, (list, tuple)) or len(path_overlap) != len(scores):
        raise ValueError(
            "Enabled direct subspace supervision requires model output "
            f"`{SUBSPACE_PATH_OVERLAP_KEY}` aligned with score levels."
        )

    validated: List[torch.Tensor] = []
    num_leaf: Optional[int] = None
    for level, (level_scores, level_overlap) in enumerate(zip(scores, path_overlap)):
        if not isinstance(level_overlap, torch.Tensor) or level_overlap.ndim != 2:
            raise ValueError(
                f"Subspace path overlap for level {level} must have shape [num_leaf, C]."
            )
        if level_overlap.is_floating_point():
            raise ValueError(
                f"Subspace path overlap for level {level} must be an integer count."
            )
        if int(level_overlap.size(0)) <= 0 or int(level_overlap.size(1)) <= 0:
            raise ValueError(f"Subspace path overlap for level {level} cannot be empty.")
        if int(level_scores.size(1)) != int(level_overlap.size(1)):
            raise ValueError(
                f"Subspace score/target width mismatch at level {level}: "
                f"{int(level_scores.size(1))} versus {int(level_overlap.size(1))}."
            )
        if num_leaf is None:
            num_leaf = int(level_overlap.size(0))
        elif int(level_overlap.size(0)) != num_leaf:
            raise ValueError("Subspace path overlap tables must share one leaf dimension.")
        validated.append(level_overlap)
    return validated


def _resolve_level_weights(
    output: Mapping[str, Any],
    cfg: Any,
    depth: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the model's own level coefficients, normalized to sum to one.

    The subspace arm deliberately does not define a separate weighting. It reads
    the level weights the model family's native objective would use and spends
    them where that objective spends them: as the per-level coefficients of the
    scalarisation. The target geometry is weight-free, so a level's importance is
    expressed once, in the loss, and `model.weight_mode` keeps the same meaning
    here as in the baseline the arm is measured against.
    """
    level_node_ids = output.get("hiercos_level_node_ids")
    if not isinstance(level_node_ids, list) or len(level_node_ids) != int(depth):
        raise ValueError(
            "Direct subspace supervision resolves its level weights from the "
            "model's native weighting and therefore requires model output "
            "`hiercos_level_node_ids` aligned with hierarchy depth."
        )
    from models.hiercos.losses import resolve_level_weights

    weights = resolve_level_weights(
        output=output,
        cfg=cfg,
        level_node_ids=level_node_ids,
        num_levels=int(depth),
        device=device,
        dtype=dtype,
    )
    if bool((weights <= 0).any()):
        raise ValueError(
            "Direct subspace supervision requires strictly positive level weights; "
            "a zero weight would drop that level from the objective entirely."
        )
    return weights / weights.sum()


def _validate_coordinates(
    output: Mapping[str, Any],
    batch_size: int,
) -> torch.Tensor:
    coordinates = output.get(SUBSPACE_COORDINATES_KEY)
    if not isinstance(coordinates, torch.Tensor) or coordinates.ndim != 2:
        raise ValueError(
            "Enabled direct subspace supervision requires model output "
            f"`{SUBSPACE_COORDINATES_KEY}` with shape [B, N]."
        )
    if int(coordinates.size(0)) != int(batch_size):
        raise ValueError(
            "Direct subspace supervision coordinate batch size must match the score batch size."
        )
    return coordinates


def _compute_soft_cross_entropy_loss(
    scores: Sequence[torch.Tensor],
    path_overlap: Sequence[torch.Tensor],
    coordinates: torch.Tensor,
    hard_targets: torch.Tensor,
    resolved: SubspaceSupervisionConfig,
    level_weights: torch.Tensor,
    return_aux: bool,
):
    depth = len(scores)
    leaf_targets = hard_targets[:, -1]
    coordinate_norm = (
        coordinates.to(dtype=torch.float32)
        .norm(p=2, dim=1, keepdim=True)
        .clamp_min(resolved.eps)
    )
    num_leaf = int(path_overlap[0].size(0))
    if bool((leaf_targets < 0).any()) or bool((leaf_targets >= num_leaf).any()):
        raise ValueError(
            f"Direct subspace supervision leaf targets must be in [0, {num_leaf})."
        )

    raw_level_losses: List[torch.Tensor] = []
    weighted_level_losses: List[torch.Tensor] = []
    level_divergences: List[torch.Tensor] = []
    score_norm_means: List[torch.Tensor] = []
    metrics: Dict[str, float] = {}

    for level, (level_scores, overlap) in enumerate(zip(scores, path_overlap)):
        compute_dtype = (
            torch.float32
            if level_scores.dtype in {torch.float16, torch.bfloat16}
            else level_scores.dtype
        )
        # The target is rebuilt rather than tabulated so it stays valid for any
        # taxonomy the model exposes; it depends on the hierarchy alone.
        lookup = subspace_target_profile(
            path_overlap=overlap.to(device=level_scores.device),
            depth=depth,
        )
        level_targets = lookup.to(dtype=compute_dtype).index_select(
            0,
            leaf_targets.to(device=level_scores.device),
        )
        expected_classes = hard_targets[:, level].to(device=level_scores.device)
        if not torch.equal(level_targets.argmax(dim=1), expected_classes):
            raise ValueError(
                "Direct subspace target profiles are inconsistent with hard targets "
                f"at hierarchy level {level}."
            )

        autocast_off = nullcontext()
        if level_scores.device.type in {"cpu", "cuda", "xpu", "mps"}:
            autocast_off = torch.autocast(
                device_type=level_scores.device.type,
                enabled=False,
            )
        with autocast_off:
            score_work = level_scores.to(dtype=compute_dtype)
            # One scale shared by every level, not one per level. Under the
            # intended geometry z_l[y_l] = ||u|| at every level and the target
            # already has t_l[y_l] = 1, so dividing by ||u|| puts both sides in
            # the same units while giving up only the single global degree of
            # freedom the score map is genuinely homogeneous in. Normalizing per
            # level instead would free one scale per level and leave the node
            # energies unidentified.
            scale = coordinate_norm.to(device=score_work.device, dtype=compute_dtype)
            normalized_scores = score_work / scale
            target_probabilities = F.softmax(level_targets / resolved.tau, dim=1)
            score_log_probabilities = F.log_softmax(
                normalized_scores / resolved.tau,
                dim=1,
            )
            raw_loss = -(target_probabilities * score_log_probabilities).sum(dim=1).mean()
            # The target entropy is the attainable floor of `raw_loss`; the
            # divergence is what the optimizer can actually remove and is zero
            # exactly at the intended node geometry.
            target_entropy = -(
                target_probabilities
                * torch.log(target_probabilities.clamp_min(resolved.eps))
            ).sum(dim=1).mean()
            divergence = raw_loss - target_entropy
            score_norm_mean = score_work.norm(p=2, dim=1).mean()

        level_weight = level_weights[level].to(
            device=raw_loss.device,
            dtype=raw_loss.dtype,
        )
        weighted_loss = level_weight * raw_loss
        raw_level_losses.append(raw_loss)
        weighted_level_losses.append(weighted_loss)
        level_divergences.append(divergence)
        score_norm_means.append(score_norm_mean)

        metrics[f"subspace_soft_cross_entropy_level_{level}"] = float(raw_loss.detach().item())
        metrics[f"subspace_target_kl_level_{level}"] = float(divergence.detach().item())
        # `loss_level_*` is this level's contribution to the total, matching the
        # native Hier-COS losses; `subspace_*_level_*` stay unweighted so the
        # geometry diagnostics remain comparable across weight modes.
        metrics[f"loss_level_{level}"] = float(weighted_loss.detach().item())
        metrics[f"loss_weight_level_{level}"] = float(level_weight.detach().item())
        metrics[f"subspace_score_l2_level_{level}"] = float(
            score_norm_mean.detach().item()
        )

    # The levels enter the objective through `model.weight_mode`, the same
    # coefficients the native Hier-COS softmax losses apply. They sum to one, so
    # this is a convex combination and reduces to the previous uniform mean under
    # `equal`; the loss scale is unchanged and learning rates transfer from the
    # Hier-COS baselines. The aggregates below follow the same weighting so that
    # `total` is exactly the sum of the reported per-level contributions.
    stacked_weights = level_weights.to(device=raw_level_losses[0].device)
    total = torch.stack(weighted_level_losses).sum()
    metrics["total"] = float(total.detach().item())
    metrics["subspace_soft_cross_entropy"] = float(total.detach().item())
    metrics["subspace_target_kl"] = float(
        (torch.stack(level_divergences) * stacked_weights).sum().detach().item()
    )
    metrics["subspace_score_l2"] = float(
        torch.stack(score_norm_means).mean().detach().item()
    )

    if not return_aux:
        return total, metrics
    return total, metrics, {
        "level_losses": weighted_level_losses,
        "subspace_soft_cross_entropy_losses": raw_level_losses,
        "subspace_target_divergences": level_divergences,
    }


def compute_subspace_supervision_loss(
    output: Mapping[str, Any],
    targets: Any,
    cfg: Any,
    _taxonomy: Optional[Dict[str, Any]] = None,
    return_aux: bool = False,
) -> Union[
    Tuple[torch.Tensor, Dict[str, float]],
    Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]],
]:
    """Train the model's per-level taxonomy-subspace scores directly."""
    resolved = resolve_subspace_supervision_config(cfg)
    if not resolved.enabled:
        raise ValueError("Direct subspace supervision loss called while the mechanism is disabled.")

    scores = _validate_scores(output)
    depth = len(scores)
    hard_targets = _hard_targets(targets, depth=depth).to(device=scores[0].device)
    if int(hard_targets.size(0)) != int(scores[0].size(0)):
        raise ValueError(
            "Direct subspace supervision target batch size must match the score batch size."
        )

    path_overlap = _validate_path_overlap(output, scores=scores)
    coordinates = _validate_coordinates(output, batch_size=int(scores[0].size(0)))
    level_weights = _resolve_level_weights(
        output=output,
        cfg=cfg,
        depth=depth,
        device=scores[0].device,
        dtype=(
            torch.float32
            if scores[0].dtype in {torch.float16, torch.bfloat16}
            else scores[0].dtype
        ),
    )
    return _compute_soft_cross_entropy_loss(
        scores=scores,
        path_overlap=path_overlap,
        coordinates=coordinates,
        hard_targets=hard_targets,
        resolved=resolved,
        level_weights=level_weights,
        return_aux=return_aux,
    )


__all__ = [
    "DEFAULT_TAU",
    "SUBSPACE_COORDINATES_KEY",
    "SUBSPACE_PATH_OVERLAP_KEY",
    "SUBSPACE_SCORES_KEY",
    "SubspaceSupervisionConfig",
    "compute_subspace_supervision_loss",
    "resolve_subspace_supervision_config",
    "subspace_norms",
    "subspace_target_profile",
    "subspace_supervision_enabled",
]
