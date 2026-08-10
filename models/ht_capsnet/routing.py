from typing import Optional

import torch


def squash(x: torch.Tensor, dim: int = -1, eps: float = 1e-7) -> torch.Tensor:
    """Apply the standard capsule squash non-linearity."""
    sq_norm = (x * x).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * x / torch.sqrt(sq_norm + eps)


def safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-7) -> torch.Tensor:
    """Compute a numerically stable L2 norm."""
    sq_norm = (x * x).sum(dim=dim)
    return torch.sqrt(sq_norm + eps)


def taxonomy_guided_routing_weights(
    raw_weights: torch.Tensor,
    level: int,
    taxonomy_matrix: Optional[torch.Tensor] = None,
    prev_predictions: Optional[torch.Tensor] = None,
    taxonomy_temperature: float = 0.5,
    mask_threshold_high: float = 0.9,
    mask_threshold_low: float = 0.1,
    mask_temperature: float = 0.5,
    mask_center: float = 0.5,
) -> torch.Tensor:
    """Compute routing weights with optional taxonomy-aware masking.

    Args:
        raw_weights: Unnormalized routing logits with shape ``[B, N_in, N_out]``.
        level: Hierarchy level index.
        taxonomy_matrix: Parent-child map with shape ``[N_parent, N_out]``.
        prev_predictions: Previous-level capsules with shape ``[B, N_parent, D_parent]``.

    Returns:
        Routing coefficients with shape ``[B, N_in, N_out]``.
    """
    if raw_weights.ndim != 3:
        raise ValueError(
            f"`raw_weights` must have shape [B, N_in, N_out], got {tuple(raw_weights.shape)}."
        )

    if level == 0:
        return torch.softmax(raw_weights, dim=-1)

    if taxonomy_matrix is None:
        return torch.softmax(raw_weights * mask_temperature, dim=-1)

    taxonomy_matrix = taxonomy_matrix.to(device=raw_weights.device, dtype=raw_weights.dtype)
    if taxonomy_matrix.ndim != 2:
        raise ValueError(
            f"`taxonomy_matrix` must have shape [N_parent, N_out], got {tuple(taxonomy_matrix.shape)}."
        )

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

    replicated = soft_taxonomy.repeat(1, repeats, 1)
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
    """Apply hierarchical agreement gating to votes.

    Args:
        votes: Vote tensors with shape ``[B, N_in, N_out, D_out]``.
        prev_predictions: Previous-level capsules with shape ``[B, N_parent, D_prev]``.
        dim_transform: Projection matrix with shape ``[D_prev, D_out]``.
        hierarchical_gate: Learnable gate with shape ``[1, N_out, N_parent]``.

    Returns:
        Gated votes with the same shape as ``votes``.
    """
    if votes.ndim != 4:
        raise ValueError(f"`votes` must have shape [B, N_in, N_out, D_out], got {tuple(votes.shape)}.")
    if prev_predictions.ndim != 3:
        raise ValueError(
            f"`prev_predictions` must have shape [B, N_parent, D_prev], got {tuple(prev_predictions.shape)}."
        )
    if dim_transform.ndim != 2:
        raise ValueError(
            f"`dim_transform` must have shape [D_prev, D_out], got {tuple(dim_transform.shape)}."
        )
    if hierarchical_gate.ndim != 3:
        raise ValueError(
            f"`hierarchical_gate` must have shape [1, N_out, N_parent], got {tuple(hierarchical_gate.shape)}."
        )

    bsz, n_in, n_out, d_out = votes.shape
    if int(prev_predictions.size(-1)) != int(dim_transform.size(0)):
        raise ValueError(
            "Mismatch between previous capsule dim and `dim_transform`: "
            f"{int(prev_predictions.size(-1))} vs {int(dim_transform.size(0))}."
        )
    if int(dim_transform.size(1)) != d_out:
        raise ValueError(
            f"`dim_transform` output dim ({int(dim_transform.size(1))}) must match vote dim ({d_out})."
        )
    if int(hierarchical_gate.size(1)) != n_out:
        raise ValueError(
            f"`hierarchical_gate` N_out ({int(hierarchical_gate.size(1))}) must match votes N_out ({n_out})."
        )
    if int(hierarchical_gate.size(2)) != int(prev_predictions.size(1)):
        raise ValueError(
            "`hierarchical_gate` N_parent must match previous-level capsule count: "
            f"{int(hierarchical_gate.size(2))} vs {int(prev_predictions.size(1))}."
        )

    prev_transformed = torch.matmul(prev_predictions, dim_transform)  # [B, N_parent, D_out]

    votes_reshaped = votes.reshape(bsz, n_in * n_out, d_out)
    agreement = torch.matmul(votes_reshaped, prev_transformed.transpose(1, 2))
    agreement = agreement.reshape(bsz, n_in, n_out, prev_predictions.size(1))

    gate = hierarchical_gate.unsqueeze(1)  # [1, 1, N_out, N_parent]
    gated = agreement * gate
    consistency = torch.sigmoid(gated.sum(dim=-1, keepdim=True))  # [B, N_in, N_out, 1]
    return votes * consistency
