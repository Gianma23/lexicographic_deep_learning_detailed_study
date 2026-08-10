"""Shared subspace-norm scoring and direct-supervision loss.

The loss is capability based: any model can opt in by exposing
``subspace_scores_per_level`` and ``subspace_target_profiles_by_level`` in its
forward output.  No model-family check is performed here.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


SUBSPACE_SCORES_KEY = "subspace_scores_per_level"
SUBSPACE_TARGET_PROFILES_KEY = "subspace_target_profiles_by_level"
TARGET_MODE = "sqrt_path_weights"
LOSS_MODE = "normalized_mse"


@dataclass(frozen=True)
class SubspaceSupervisionConfig:
    enabled: bool = False
    target_mode: str = TARGET_MODE
    loss: str = LOSS_MODE
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

    enabled = bool(raw_cfg.get("enabled", False))
    target_mode = raw_cfg.get("target_mode", TARGET_MODE)
    loss_mode = raw_cfg.get("loss", LOSS_MODE)
    raw_eps = raw_cfg.get("eps", 1.0e-12)

    if not isinstance(target_mode, str) or target_mode != TARGET_MODE:
        raise ValueError(
            "train.subspace_supervision.target_mode must be "
            f"'{TARGET_MODE}', got {target_mode!r}."
        )
    if not isinstance(loss_mode, str) or loss_mode != LOSS_MODE:
        raise ValueError(
            "train.subspace_supervision.loss must be "
            f"'{LOSS_MODE}', got {loss_mode!r}."
        )
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

    return SubspaceSupervisionConfig(
        enabled=enabled,
        target_mode=target_mode,
        loss=loss_mode,
        eps=eps,
    )


def subspace_supervision_enabled(cfg: Any) -> bool:
    return bool(resolve_subspace_supervision_config(cfg).enabled)


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


@torch.no_grad()
def build_sqrt_path_target_profiles(
    level_node_ids: Sequence[torch.Tensor],
    level_subspace_masks: Sequence[torch.Tensor],
    leaf_to_level_local: torch.Tensor,
    path_weights: torch.Tensor,
) -> List[torch.Tensor]:
    """Precompute dense subspace-norm targets for every leaf class."""
    if not isinstance(leaf_to_level_local, torch.Tensor) or leaf_to_level_local.ndim != 2:
        raise ValueError("Subspace target construction requires leaf_to_level_local [num_leaf, L].")
    depth = int(leaf_to_level_local.size(1))
    if len(level_node_ids) != depth or len(level_subspace_masks) != depth:
        raise ValueError("Subspace target topology must be aligned with hierarchy depth.")
    if not isinstance(path_weights, torch.Tensor) or path_weights.ndim != 1:
        raise ValueError("Subspace path weights must have shape [L].")
    if int(path_weights.numel()) != depth:
        raise ValueError("Subspace path weights must be aligned with hierarchy depth.")

    weights = path_weights.to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("Subspace path weights must be finite and non-negative.")
    weight_sum = weights.sum()
    if float(weight_sum.item()) <= 0.0:
        raise ValueError("Subspace path weights must have positive total mass.")
    weights = weights / weight_sum

    parsed_node_ids = [node_ids.to(device="cpu", dtype=torch.long) for node_ids in level_node_ids]
    total_nodes = sum(int(node_ids.numel()) for node_ids in parsed_node_ids)
    num_leaf = int(leaf_to_level_local.size(0))
    target_coordinates = torch.zeros((num_leaf, total_nodes), dtype=torch.float64)
    row_ids = torch.arange(num_leaf, dtype=torch.long)
    local_paths = leaf_to_level_local.to(device="cpu", dtype=torch.long)

    for level, node_ids in enumerate(parsed_node_ids):
        local_ids = local_paths[:, level]
        if bool((local_ids < 0).any()) or bool((local_ids >= int(node_ids.numel())).any()):
            raise ValueError(f"Invalid path indices while building level {level} targets.")
        target_coordinates[row_ids, node_ids[local_ids]] = torch.sqrt(weights[level])

    profiles = subspace_norms(
        target_coordinates,
        [mask.to(device="cpu") for mask in level_subspace_masks],
    )
    return [profile.to(dtype=torch.float32).detach() for profile in profiles]


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


def _validate_contract(
    output: Mapping[str, Any],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if not isinstance(output, Mapping):
        raise TypeError("Direct subspace supervision requires model output as a mapping.")
    scores = output.get(SUBSPACE_SCORES_KEY)
    target_profiles = output.get(SUBSPACE_TARGET_PROFILES_KEY)
    if not isinstance(scores, (list, tuple)) or not scores:
        raise ValueError(
            "Enabled direct subspace supervision requires model output "
            f"`{SUBSPACE_SCORES_KEY}` as a non-empty level-aligned list."
        )
    if not isinstance(target_profiles, (list, tuple)) or len(target_profiles) != len(scores):
        raise ValueError(
            "Enabled direct subspace supervision requires model output "
            f"`{SUBSPACE_TARGET_PROFILES_KEY}` aligned with score levels."
        )

    validated_scores: List[torch.Tensor] = []
    validated_profiles: List[torch.Tensor] = []
    batch_size: Optional[int] = None
    num_leaf: Optional[int] = None
    for level, (level_scores, level_profiles) in enumerate(zip(scores, target_profiles)):
        if not isinstance(level_scores, torch.Tensor) or level_scores.ndim != 2:
            raise ValueError(f"Subspace scores for level {level} must have shape [B, C].")
        if not isinstance(level_profiles, torch.Tensor) or level_profiles.ndim != 2:
            raise ValueError(
                f"Subspace target profiles for level {level} must have shape [num_leaf, C]."
            )
        if not level_scores.is_floating_point():
            raise ValueError(f"Subspace scores for level {level} must be floating point.")
        if not level_profiles.is_floating_point():
            raise ValueError(f"Subspace targets for level {level} must be floating point.")
        if int(level_scores.size(0)) <= 0 or int(level_scores.size(1)) <= 0:
            raise ValueError(f"Subspace scores for level {level} cannot be empty.")
        if int(level_profiles.size(0)) <= 0 or int(level_profiles.size(1)) <= 0:
            raise ValueError(f"Subspace target profiles for level {level} cannot be empty.")
        if tuple(level_scores.shape[1:]) != tuple(level_profiles.shape[1:]):
            raise ValueError(
                f"Subspace score/target width mismatch at level {level}: "
                f"{int(level_scores.size(1))} versus {int(level_profiles.size(1))}."
            )
        if batch_size is None:
            batch_size = int(level_scores.size(0))
        elif int(level_scores.size(0)) != batch_size:
            raise ValueError("Subspace score levels must share one batch size.")
        if num_leaf is None:
            num_leaf = int(level_profiles.size(0))
        elif int(level_profiles.size(0)) != num_leaf:
            raise ValueError("Subspace target lookup tables must share one leaf dimension.")
        if not bool(torch.isfinite(level_scores).all()):
            raise ValueError(f"Subspace scores for level {level} contain non-finite values.")
        if not bool(torch.isfinite(level_profiles).all()):
            raise ValueError(f"Subspace targets for level {level} contain non-finite values.")
        validated_scores.append(level_scores)
        validated_profiles.append(level_profiles)
    return validated_scores, validated_profiles


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
    """Match predicted and ground-truth subspace profiles at every level."""
    resolved = resolve_subspace_supervision_config(cfg)
    if not resolved.enabled:
        raise ValueError("Direct subspace supervision loss called while the mechanism is disabled.")

    scores, profile_lookup = _validate_contract(output)
    depth = len(scores)
    hard_targets = _hard_targets(targets, depth=depth).to(device=scores[0].device)
    leaf_targets = hard_targets[:, -1]
    num_leaf = int(profile_lookup[0].size(0))
    if bool((leaf_targets < 0).any()) or bool((leaf_targets >= num_leaf).any()):
        raise ValueError(
            f"Direct subspace supervision leaf targets must be in [0, {num_leaf})."
        )

    raw_level_losses: List[torch.Tensor] = []
    weighted_level_losses: List[torch.Tensor] = []
    score_norm_means: List[torch.Tensor] = []
    metrics: Dict[str, float] = {}
    for level, (level_scores, lookup) in enumerate(zip(scores, profile_lookup)):
        compute_dtype = (
            torch.float32
            if level_scores.dtype in {torch.float16, torch.bfloat16}
            else level_scores.dtype
        )
        level_targets = lookup.to(device=level_scores.device, dtype=compute_dtype).index_select(
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
            normalized_scores = F.normalize(
                score_work,
                p=2.0,
                dim=1,
                eps=resolved.eps,
            )
            normalized_targets = F.normalize(
                level_targets,
                p=2.0,
                dim=1,
                eps=resolved.eps,
            )
            raw_loss = (normalized_scores - normalized_targets).pow(2).sum(dim=1).mean()
            score_norm_mean = score_work.norm(p=2, dim=1).mean()
        weighted_loss = raw_loss / float(depth)
        raw_level_losses.append(raw_loss)
        weighted_level_losses.append(weighted_loss)
        score_norm_means.append(score_norm_mean)

        metrics[f"subspace_profile_mse_level_{level}"] = float(raw_loss.detach().item())
        metrics[f"loss_level_{level}"] = float(weighted_loss.detach().item())
        metrics[f"subspace_score_l2_level_{level}"] = float(
            score_norm_mean.detach().item()
        )

    total = torch.stack(weighted_level_losses).sum()
    metrics["total"] = float(total.detach().item())
    metrics["subspace_profile_mse"] = float(total.detach().item())
    metrics["subspace_score_l2"] = float(
        torch.stack(score_norm_means).mean().detach().item()
    )

    if not return_aux:
        return total, metrics
    return total, metrics, {
        "level_losses": weighted_level_losses,
        "subspace_profile_losses": raw_level_losses,
    }


__all__ = [
    "LOSS_MODE",
    "SUBSPACE_SCORES_KEY",
    "SUBSPACE_TARGET_PROFILES_KEY",
    "TARGET_MODE",
    "SubspaceSupervisionConfig",
    "build_sqrt_path_target_profiles",
    "compute_subspace_supervision_loss",
    "resolve_subspace_supervision_config",
    "subspace_norms",
    "subspace_supervision_enabled",
]
