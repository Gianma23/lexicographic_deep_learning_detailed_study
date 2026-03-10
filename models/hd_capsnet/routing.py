from typing import Tuple

import torch


def squash(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    sq_norm = (x * x).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * x / torch.sqrt(sq_norm + eps)


def dynamic_routing(votes: torch.Tensor, num_iters: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    votes: [B, N_in, N_out, D]
    returns:
      capsules: [B, N_out, D]
      coupling: [B, N_in, N_out]
    """
    if num_iters <= 0:
        raise ValueError("num_iters must be > 0 for dynamic routing.")

    bsz, n_in, n_out, _ = votes.shape
    logits = torch.zeros((bsz, n_in, n_out), device=votes.device, dtype=votes.dtype)

    for _ in range(num_iters):
        coupling = torch.softmax(logits, dim=-1)
        s = (coupling.unsqueeze(-1) * votes).sum(dim=1)
        v = squash(s, dim=-1)

        agreement = (votes * v.unsqueeze(1)).sum(dim=-1)
        logits = logits + agreement

    return v, coupling
