from typing import Optional

import torch


def squash(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Capsule squash non-linearity."""
    sq_norm = (x * x).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * x / torch.sqrt(sq_norm + eps)


def safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Numerically stable L2 norm used by taxonomy weighting."""
    sq_norm = (x * x).sum(dim=dim)
    return torch.sqrt(sq_norm + eps)


def taxonomy_guided_routing_weights(
    raw_weights: torch.Tensor,
    level: int,
    taxonomy_matrix: Optional[torch.Tensor] = None,
    prev_predictions: Optional[torch.Tensor] = None,
    taxonomy_temperature: float = 5.0,
    mask_threshold_high: float = 0.9,
    mask_threshold_low: float = 0.1,
    mask_temperature: float = 0.5,
    mask_center: float = 0.5,
) -> torch.Tensor:
    """
    Compute routing weights using upstream-style taxonomy-aware masking.

    Args:
        raw_weights: [B, N_in, N_out]
        level: hierarchy level index
        taxonomy_matrix: [N_parent, N_out] binary parent-child map
        prev_predictions: [B, N_parent, D_parent] previous level capsules
    """
    if level == 0:
        return torch.softmax(raw_weights, dim=-1)

    if taxonomy_matrix is None:
        return torch.softmax(raw_weights * mask_temperature, dim=-1)

    bsz, n_in, _ = raw_weights.shape
    parent_classes = int(taxonomy_matrix.size(0))
    if parent_classes <= 0:
        return torch.softmax(raw_weights * mask_temperature, dim=-1)

    repeats = n_in // parent_classes

    if prev_predictions is not None:
        prev_activations = torch.softmax(safe_norm(prev_predictions, dim=-1), dim=-1)  # [B, N_parent]
        weighted_taxonomy = taxonomy_matrix.unsqueeze(0) * prev_activations.unsqueeze(2)  # [B, N_parent, N_out]
    else:
        weighted_taxonomy = taxonomy_matrix.unsqueeze(0).expand(bsz, -1, -1)

    mask_range = float(mask_threshold_high - mask_threshold_low)
    soft_taxonomy = mask_range * torch.sigmoid(
        taxonomy_temperature * (weighted_taxonomy - mask_center)
    ) + mask_threshold_low  # [B, N_parent, N_out]

    replicated = soft_taxonomy.repeat(1, max(1, repeats), 1)
    remaining = n_in - (repeats * parent_classes)
    if remaining > 0:
        remainder = soft_taxonomy[:, :1, :].repeat(1, remaining, 1)
        extended_mask = torch.cat([replicated, remainder], dim=1)
    else:
        extended_mask = replicated

    extended_mask = extended_mask[:, :n_in, :]
    masked_weights = raw_weights * extended_mask
    return torch.softmax(masked_weights * mask_temperature, dim=-1)


def hierarchical_agreement(
    votes: torch.Tensor,
    prev_predictions: torch.Tensor,
    dim_transform: torch.Tensor,
    hierarchical_gate: torch.Tensor,
) -> torch.Tensor:
    """
    Apply upstream-style hierarchical agreement gating to votes.

    Args:
        votes: [B, N_in, N_out, D_out]
        prev_predictions: [B, N_parent, D_prev]
        dim_transform: [D_prev, D_out]
        hierarchical_gate: [1, N_out, N_parent]
    """
    bsz, n_in, n_out, d_out = votes.shape
    prev_transformed = torch.matmul(prev_predictions, dim_transform)  # [B, N_parent, D_out]

    votes_reshaped = votes.reshape(bsz, n_in * n_out, d_out)
    agreement = torch.matmul(votes_reshaped, prev_transformed.transpose(1, 2))
    agreement = agreement.reshape(bsz, n_in, n_out, prev_predictions.size(1))

    gate = hierarchical_gate.unsqueeze(1)  # [1, 1, N_out, N_parent]
    gated = agreement * gate
    consistency = torch.sigmoid(gated.sum(dim=-1, keepdim=True))  # [B, N_in, N_out, 1]
    return votes * consistency
