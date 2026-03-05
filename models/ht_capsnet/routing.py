from typing import Optional, Tuple

import torch


def squash(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    sq_norm = (x * x).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * x / torch.sqrt(sq_norm + eps)


def taxonomy_mask_from_matrix(
    taxonomy_matrix: Optional[torch.Tensor],
    prev_parent_probs: Optional[torch.Tensor],
    temperature: float,
    mask_high: float,
    mask_low: float,
    mask_center: float,
) -> Optional[torch.Tensor]:
    if taxonomy_matrix is None:
        return None

    if prev_parent_probs is None:
        weighted = taxonomy_matrix.unsqueeze(0)
    else:
        weighted = torch.matmul(prev_parent_probs, taxonomy_matrix).unsqueeze(1)

    mask_range = mask_high - mask_low
    return mask_range * torch.sigmoid(temperature * (weighted - mask_center)) + mask_low


def taxonomy_guided_routing(
    votes: torch.Tensor,
    num_iters: int = 3,
    taxonomy_matrix: Optional[torch.Tensor] = None,
    prev_parent_probs: Optional[torch.Tensor] = None,
    taxonomy_temperature: float = 5.0,
    mask_threshold_high: float = 0.9,
    mask_threshold_low: float = 0.1,
    mask_temperature: float = 0.5,
    mask_center: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    votes: [B, N_in, N_out, D]
    taxonomy_matrix: [N_parent, N_out] with entries {0,1}
    prev_parent_probs: [B, N_parent]
    returns:
      capsules: [B, N_out, D]
      coupling: [B, N_in, N_out]
    """
    bsz, n_in, n_out, _ = votes.shape
    logits = torch.zeros((bsz, n_in, n_out), device=votes.device, dtype=votes.dtype)

    for _ in range(num_iters):
        coupling = torch.softmax(logits * mask_temperature, dim=-1)

        mask = taxonomy_mask_from_matrix(
            taxonomy_matrix=taxonomy_matrix,
            prev_parent_probs=prev_parent_probs,
            temperature=taxonomy_temperature,
            mask_high=mask_threshold_high,
            mask_low=mask_threshold_low,
            mask_center=mask_center,
        )
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            coupling = coupling * mask
            coupling = coupling / coupling.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        s = (coupling.unsqueeze(-1) * votes).sum(dim=1)
        v = squash(s, dim=-1)

        agreement = (votes * v.unsqueeze(1)).sum(dim=-1)
        logits = logits + agreement

    return v, coupling
