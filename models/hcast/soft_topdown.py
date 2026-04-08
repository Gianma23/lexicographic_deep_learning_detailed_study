from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_parent_index_array(
    name: str,
    mapping: Any,
    expected_children: int,
    max_parent: int,
) -> List[int]:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"`taxonomy['{name}']` must be a mapping child->parent.")

    out: Dict[int, int] = {}
    for child_raw, parent_raw in mapping.items():
        try:
            child = int(child_raw)
            parent = int(parent_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"`taxonomy['{name}']` keys/values must be integers.") from exc

        if child < 0 or child >= expected_children:
            raise ValueError(
                f"`taxonomy['{name}']` child id {child} out of range, expected [0, {expected_children})."
            )
        if parent < 0 or parent >= max_parent:
            raise ValueError(
                f"`taxonomy['{name}']` parent id {parent} out of range, expected [0, {max_parent})."
            )
        if child in out and out[child] != parent:
            raise ValueError(f"`taxonomy['{name}']` has conflicting parents for child {child}.")
        out[child] = parent

    missing = [child for child in range(expected_children) if child not in out]
    if missing:
        raise ValueError(
            f"`taxonomy['{name}']` must define exactly one parent for each child id in [0, {expected_children}). "
            f"Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    return [out[child] for child in range(expected_children)]


def _extract_parent_indices_per_transition(
    taxonomy: Dict[str, Any],
    num_classes_per_level: List[int],
) -> List[List[int]]:
    if taxonomy is None:
        raise ValueError(
            "taxonomy is required when cfg.model.soft_topdown.enabled=true."
        )

    parent_of = taxonomy.get("parent_of")
    if not isinstance(parent_of, Mapping):
        raise ValueError(
            "Unsupported taxonomy format for soft top-down gating. "
            "Expected `taxonomy['parent_of']` with child->parent mappings for each transition level."
        )

    parent_indices_per_transition: List[List[int]] = []
    for level in range(1, len(num_classes_per_level)):
        mapping = parent_of.get(level, parent_of.get(str(level)))
        if mapping is None:
            raise ValueError(
                f"`taxonomy['parent_of']` is missing transition mapping for level {level} "
                f"(from level {level - 1} to level {level})."
            )
        parent_indices_per_transition.append(
            _as_parent_index_array(
                name=f"parent_of[{level}]",
                mapping=mapping,
                expected_children=int(num_classes_per_level[level]),
                max_parent=int(num_classes_per_level[level - 1]),
            )
        )

    return parent_indices_per_transition


class SoftTopDownKLGating(nn.Module):
    """Soft top-down gating + KL regularization plugin for H-CAST."""

    def __init__(
        self,
        num_classes_per_level: List[int],
        taxonomy: Dict[str, Any],
        temperature: float = 1.0,
        gate_strength: float = 1.0,
        tau: float = 0.0,
        use_gated_logits: bool = True,
        detach_upper_probs: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.num_classes_per_level = [int(v) for v in num_classes_per_level]
        if len(self.num_classes_per_level) < 2:
            raise ValueError("SoftTopDownKLGating requires at least 2 hierarchy levels.")

        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError("SoftTopDownKLGating.temperature must be > 0.")

        self.gate_strength = float(gate_strength)
        self.tau = float(tau)
        if self.tau < 0.0:
            self.tau = 0.0
        self.use_gated_logits = bool(use_gated_logits)
        self.detach_upper_probs = bool(detach_upper_probs)
        self.eps = float(eps) if float(eps) > 0.0 else 1e-12

        parent_indices_per_transition = _extract_parent_indices_per_transition(
            taxonomy=taxonomy,
            num_classes_per_level=self.num_classes_per_level,
        )
        self._parent_index_buffer_names: List[str] = []
        for transition_idx, parent_indices in enumerate(parent_indices_per_transition, start=1):
            name = f"parent_index_transition_{transition_idx}"
            self.register_buffer(name, torch.tensor(parent_indices, dtype=torch.long), persistent=False)
            self._parent_index_buffer_names.append(name)

    def _parent_indices(self, child_level: int) -> torch.Tensor:
        if child_level <= 0:
            raise ValueError("child_level must be >= 1.")
        buffer_name = self._parent_index_buffer_names[child_level - 1]
        return getattr(self, buffer_name)

    def forward(self, logits_per_level: List[torch.Tensor]) -> Dict[str, Any]:
        if len(logits_per_level) != len(self.num_classes_per_level):
            raise ValueError(
                f"Expected {len(self.num_classes_per_level)} logits tensors in coarse->fine order, "
                f"got {len(logits_per_level)}."
            )

        batch_size: Optional[int] = None
        for level, logits in enumerate(logits_per_level):
            if logits.ndim != 2:
                raise ValueError(f"logits_per_level[{level}] must be rank-2 [batch, classes].")
            if int(logits.size(-1)) != int(self.num_classes_per_level[level]):
                raise ValueError(
                    f"logits_per_level[{level}] has {int(logits.size(-1))} classes, "
                    f"expected {int(self.num_classes_per_level[level])}."
                )
            if batch_size is None:
                batch_size = int(logits.size(0))
            elif int(logits.size(0)) != batch_size:
                raise ValueError("All logits levels must have the same batch size.")

        priors_per_level: List[Optional[torch.Tensor]] = [None]
        penalties: List[torch.Tensor] = [logits_per_level[0].new_zeros(())]
        gated_logits_per_level: List[torch.Tensor] = [logits_per_level[0]]
        diagnostics: Dict[str, float] = {
            "soft_topdown_temperature": float(self.temperature),
            "soft_topdown_gate_strength": float(self.gate_strength),
            "soft_topdown_tau": float(self.tau),
            "soft_topdown_use_gated_logits": 1.0 if self.use_gated_logits else 0.0,
            "soft_topdown_detach_upper_probs": 1.0 if self.detach_upper_probs else 0.0,
        }

        for level in range(1, len(logits_per_level)):
            upper_logits = logits_per_level[level - 1]
            lower_logits = logits_per_level[level]
            parent_indices = self._parent_indices(level).to(device=lower_logits.device)

            work_dtype = torch.float32 if lower_logits.dtype in {torch.float16, torch.bfloat16} else lower_logits.dtype
            upper_probs = F.softmax(upper_logits.to(dtype=work_dtype) / self.temperature, dim=-1)
            if self.detach_upper_probs:
                upper_probs = upper_probs.detach()

            prior_unnorm = upper_probs.index_select(dim=1, index=parent_indices)
            prior = prior_unnorm / prior_unnorm.sum(dim=1, keepdim=True).clamp_min(self.eps)
            child_probs = F.softmax(lower_logits.to(dtype=work_dtype), dim=-1)

            kl = (child_probs * (torch.log(child_probs.clamp_min(self.eps)) - torch.log(prior.clamp_min(self.eps)))).sum(
                dim=-1
            ).mean()
            penalty = torch.clamp(kl - self.tau, min=0.0) if self.tau > 0.0 else kl

            log_prior = torch.log(prior.clamp_min(self.eps)).to(dtype=lower_logits.dtype)
            gated_logits = lower_logits + (self.gate_strength * log_prior)

            priors_per_level.append(prior.to(dtype=lower_logits.dtype))
            penalties.append(penalty)
            gated_logits_per_level.append(gated_logits)

            with torch.no_grad():
                diagnostics[f"soft_topdown_kl_level_{level}"] = float(kl.detach().item())
                diagnostics[f"soft_topdown_penalty_level_{level}"] = float(penalty.detach().item())
                diagnostics[f"soft_topdown_prior_min_level_{level}"] = float(prior.min().item())
                diagnostics[f"soft_topdown_prior_max_level_{level}"] = float(prior.max().item())

        return {
            "soft_topdown_priors_per_level": priors_per_level,
            "soft_topdown_penalties": penalties,
            "soft_topdown_diagnostics": diagnostics,
            "gated_logits_per_level": gated_logits_per_level if self.use_gated_logits else None,
        }
