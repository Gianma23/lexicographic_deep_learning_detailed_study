from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_int_list(name: str, values: Any, expected_len: int, max_parent: int) -> List[int]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"`taxonomy['{name}']` must be a list/tuple of length {expected_len}.")
    if len(values) != expected_len:
        raise ValueError(f"`taxonomy['{name}']` must have length {expected_len}, got {len(values)}.")

    out: List[int] = []
    for idx, parent in enumerate(values):
        try:
            parent_id = int(parent)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"`taxonomy['{name}'][{idx}]` must be an integer parent index.") from exc
        if parent_id < 0 or parent_id >= max_parent:
            raise ValueError(
                f"`taxonomy['{name}'][{idx}]` out of range: got {parent_id}, expected [0, {max_parent})."
            )
        out.append(parent_id)
    return out


def _as_int_mapping(name: str, mapping: Any, expected_children: int, max_parent: int) -> Dict[int, int]:
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
    return out


def _extract_parent_arrays(
    taxonomy: Dict[str, Any],
    c1: int,
    c2: int,
    c3: int,
) -> Tuple[List[int], List[int]]:
    parent_of = taxonomy.get("parent_of")
    if isinstance(parent_of, Mapping):
        # Existing repo format: parent_of[level] where level 1 means middle->coarse,
        # and level 2 means fine->middle.
        m12_raw = parent_of.get(1, parent_of.get("1"))
        m23_raw = parent_of.get(2, parent_of.get("2"))
        if m12_raw is not None and m23_raw is not None:
            map12 = _as_int_mapping("parent_of[1]", m12_raw, expected_children=c2, max_parent=c1)
            map23 = _as_int_mapping("parent_of[2]", m23_raw, expected_children=c3, max_parent=c2)
            parent_of_middle = [map12[child] for child in range(c2)]
            parent_of_fine = [map23[child] for child in range(c3)]
            return parent_of_middle, parent_of_fine

    # Explicit fallback format.
    if "parent_of_middle" in taxonomy and "parent_of_fine" in taxonomy:
        parent_of_middle = _as_int_list("parent_of_middle", taxonomy["parent_of_middle"], c2, c1)
        parent_of_fine = _as_int_list("parent_of_fine", taxonomy["parent_of_fine"], c3, c2)
        return parent_of_middle, parent_of_fine

    raise ValueError(
        "Unsupported taxonomy format for hcc projector. Expected either "
        "`taxonomy['parent_of']` with transitions 1 and 2, or explicit "
        "`parent_of_middle`/`parent_of_fine` arrays."
    )


def _build_mapping_matrix(num_parents: int, parent_of_child: List[int]) -> torch.Tensor:
    num_children = len(parent_of_child)
    mat = torch.zeros((num_parents, num_children), dtype=torch.float32)
    for child, parent in enumerate(parent_of_child):
        mat[parent, child] = 1.0
    return mat


class HierarchicalAffineProjector(nn.Module):
    """Affine-output projector enforcing p1=M12@p2 and p2=M23@p3 for 3-level hierarchies."""

    def __init__(
        self,
        num_classes_per_level: List[int],
        taxonomy: Dict[str, Any],
        eps: float = 1e-12,
        stabilize_simplex: bool = True,
    ):
        super().__init__()
        classes = [int(x) for x in num_classes_per_level]
        if len(classes) != 3:
            raise ValueError(
                f"HierarchicalAffineProjector currently supports exactly 3 levels, got {len(classes)}."
            )
        if taxonomy is None:
            raise ValueError("taxonomy is required when hcc is enabled.")

        c1, c2, c3 = classes
        parent_of_middle, parent_of_fine = _extract_parent_arrays(taxonomy, c1=c1, c2=c2, c3=c3)
        m12 = _build_mapping_matrix(c1, parent_of_middle)
        m23 = _build_mapping_matrix(c2, parent_of_fine)

        self.num_classes_per_level = classes
        self.eps = float(eps)
        self.stabilize_simplex = bool(stabilize_simplex)
        self.register_buffer("m12", m12, persistent=False)
        self.register_buffer("m23", m23, persistent=False)
        self.register_buffer(
            "parent_of_middle",
            torch.tensor(parent_of_middle, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "parent_of_fine",
            torch.tensor(parent_of_fine, dtype=torch.long),
            persistent=False,
        )

    def _mass_renormalize(
        self,
        child_scores: torch.Tensor,
        parent_mass: torch.Tensor,
        parent_of_child: torch.Tensor,
    ) -> torch.Tensor:
        """Map unconstrained child scores to positive probabilities that sum to parent mass."""
        bsz, _ = child_scores.shape
        num_parents = int(parent_mass.size(-1))
        idx = parent_of_child.to(device=child_scores.device)
        idx_batch = idx.unsqueeze(0).expand(bsz, -1)

        # Softplus keeps gradients alive even when affine projection makes values negative.
        child_pos = F.softplus(child_scores).clamp_min(self.eps)

        child_parent_mass = parent_mass[:, idx]
        mass_per_parent = torch.zeros(
            (bsz, num_parents),
            dtype=child_scores.dtype,
            device=child_scores.device,
        ).scatter_add_(1, idx_batch, child_pos)
        child_parent_denom = mass_per_parent[:, idx].clamp_min(self.eps)
        return child_pos * (child_parent_mass / child_parent_denom)

    def forward(
        self,
        probs_per_level: List[torch.Tensor],
    ) -> Dict[str, Any]:
        if len(probs_per_level) != 3:
            raise ValueError(f"Expected 3 probability tensors in coarse->fine order, got {len(probs_per_level)}.")

        p1, p2, p3 = probs_per_level
        if p1.ndim != 2 or p2.ndim != 2 or p3.ndim != 2:
            raise ValueError("Each probability tensor must be rank-2 [batch, classes].")
        if p1.shape[0] != p2.shape[0] or p2.shape[0] != p3.shape[0]:
            raise ValueError("All probability tensors must have the same batch size.")

        expected = self.num_classes_per_level
        current = [p1.shape[1], p2.shape[1], p3.shape[1]]
        if current != expected:
            raise ValueError(f"Probability shapes {current} do not match expected hierarchy classes {expected}.")

        compute_dtype = torch.float32 if p2.dtype in {torch.float16, torch.bfloat16} else p2.dtype
        autocast_off = nullcontext()
        if p2.device.type in {"cpu", "cuda", "xpu", "mps"}:
            autocast_off = torch.autocast(device_type=p2.device.type, enabled=False)

        with autocast_off:
            p1_work = p1.to(dtype=compute_dtype)
            p2_work = p2.to(dtype=compute_dtype)
            p3_work = p3.to(dtype=compute_dtype)
            m12 = self.m12.to(device=p2.device, dtype=compute_dtype)
            m23 = self.m23.to(device=p2.device, dtype=compute_dtype)

            # residual before projection for [M12 p2 - p1, M23 p3 - p2]
            residual_12_before = p2_work @ m12.transpose(0, 1).contiguous() - p1_work
            residual_23_before = p3_work @ m23.transpose(0, 1).contiguous() - p2_work
            residual_before = torch.cat([residual_12_before, residual_23_before], dim=1)

            gram_12 = m12 @ m12.transpose(0, 1).contiguous()
            eye_12 = torch.eye(gram_12.shape[0], dtype=compute_dtype, device=p2.device)
            coeff_12 = torch.linalg.solve(
                gram_12 + self.eps * eye_12,
                residual_12_before.transpose(0, 1).contiguous(),
            ).transpose(0, 1).contiguous()
            p2_hat = p2_work - coeff_12 @ m12

            residual_23_stage = p3_work @ m23.transpose(0, 1).contiguous() - p2_hat
            gram_23 = m23 @ m23.transpose(0, 1).contiguous()
            eye_23 = torch.eye(gram_23.shape[0], dtype=compute_dtype, device=p2.device)
            coeff_23 = torch.linalg.solve(
                gram_23 + self.eps * eye_23,
                residual_23_stage.transpose(0, 1).contiguous(),
            ).transpose(0, 1).contiguous()
            p3_hat = p3_work - coeff_23 @ m23

            p1_hat = p1_work
            if self.stabilize_simplex:
                # p1 is not projected in HCC; keep it unchanged and only
                # re-normalize children to match parent masses.
                p2_hat = self._mass_renormalize(
                    child_scores=p2_hat,
                    parent_mass=p1_hat,
                    parent_of_child=self.parent_of_middle,
                )
                p3_hat = self._mass_renormalize(
                    child_scores=p3_hat,
                    parent_mass=p2_hat,
                    parent_of_child=self.parent_of_fine,
                )

            residual_12_after = p2_hat @ m12.transpose(0, 1).contiguous() - p1_hat
            residual_23_after = p3_hat @ m23.transpose(0, 1).contiguous() - p2_hat
            residual_after = torch.cat([residual_12_after, residual_23_after], dim=1)

        return {
            "projected_probs_per_level": [
                p1_hat.to(dtype=p1.dtype),
                p2_hat.to(dtype=p2.dtype),
                p3_hat.to(dtype=p3.dtype),
            ],
            "residual_before": residual_before,
            "residual_after": residual_after,
        }
