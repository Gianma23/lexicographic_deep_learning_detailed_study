from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Tuple

import torch
import torch.nn as nn


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
        "Unsupported taxonomy format for design1 projector. Expected either "
        "`taxonomy['parent_of']` with transitions 1 and 2, or explicit "
        "`parent_of_middle`/`parent_of_fine` arrays."
    )


def _build_mapping_matrix(num_parents: int, parent_of_child: List[int]) -> torch.Tensor:
    num_children = len(parent_of_child)
    mat = torch.zeros((num_parents, num_children), dtype=torch.float32)
    for child, parent in enumerate(parent_of_child):
        mat[parent, child] = 1.0
    return mat


def _build_affine_constraint_matrix(m12: torch.Tensor, m23: torch.Tensor) -> torch.Tensor:
    c1, c2 = m12.shape
    c2_b, c3 = m23.shape
    if c2_b != c2:
        raise ValueError(f"Incompatible mapping shapes: M12 is [{c1}, {c2}] while M23 is [{c2_b}, {c3}].")

    zeros_1_3 = torch.zeros((c1, c3), dtype=torch.float32)
    zeros_2_1 = torch.zeros((c2, c1), dtype=torch.float32)
    top = torch.cat([torch.eye(c1, dtype=torch.float32), -m12, zeros_1_3], dim=1)
    bottom = torch.cat([zeros_2_1, torch.eye(c2, dtype=torch.float32), -m23], dim=1)
    return torch.cat([top, bottom], dim=0)


class HierarchicalAffineProjector(nn.Module):
    """Affine-output projector enforcing p1=M12@p2 and p2=M23@p3 for 3-level hierarchies."""

    def __init__(
        self,
        num_classes_per_level: List[int],
        taxonomy: Dict[str, Any],
        eps: float = 1e-12,
    ):
        super().__init__()
        classes = [int(x) for x in num_classes_per_level]
        if len(classes) != 3:
            raise ValueError(
                f"HierarchicalAffineProjector currently supports exactly 3 levels, got {len(classes)}."
            )
        if taxonomy is None:
            raise ValueError("taxonomy is required when design1 projector is enabled.")

        c1, c2, c3 = classes
        parent_of_middle, parent_of_fine = _extract_parent_arrays(taxonomy, c1=c1, c2=c2, c3=c3)
        m12 = _build_mapping_matrix(c1, parent_of_middle)
        m23 = _build_mapping_matrix(c2, parent_of_fine)
        a = _build_affine_constraint_matrix(m12, m23)

        self.num_classes_per_level = classes
        self.eps = float(eps)
        self.register_buffer("m12", m12, persistent=False)
        self.register_buffer("m23", m23, persistent=False)
        self.register_buffer("a", a, persistent=False)

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

        p = torch.cat([p1, p2, p3], dim=1)
        compute_dtype = torch.float32 if p.dtype in {torch.float16, torch.bfloat16} else p.dtype
        autocast_off = nullcontext()
        if p.device.type in {"cpu", "cuda", "xpu", "mps"}:
            autocast_off = torch.autocast(device_type=p.device.type, enabled=False)

        with autocast_off:
            p_work = p.to(dtype=compute_dtype)
            a = self.a.to(device=p.device, dtype=compute_dtype)
            a_t = a.transpose(0, 1).contiguous()
            aat = a @ a_t
            reg_eye = torch.eye(aat.shape[0], dtype=compute_dtype, device=p.device)
            aat_reg = aat + self.eps * reg_eye

            residual_before = p_work @ a_t
            solve_rhs = residual_before.transpose(0, 1).contiguous().to(dtype=aat_reg.dtype)
            correction_coeff = torch.linalg.solve(aat_reg, solve_rhs).transpose(0, 1).contiguous()
            p_hat = p_work - (correction_coeff @ a)
            residual_after = p_hat @ a_t

        p1_hat, p2_hat, p3_hat = torch.split(p_hat, expected, dim=1)
        return {
            "projected_probs_per_level": [
                p1_hat.to(dtype=p1.dtype),
                p2_hat.to(dtype=p2.dtype),
                p3_hat.to(dtype=p3.dtype),
            ],
            "residual_before": residual_before,
            "residual_after": residual_after,
        }
