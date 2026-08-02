"""Shared Hierarchical Constraint Cascade (HCC) plumbing.

`HierarchicalAffineProjector` performs the linear affine hierarchy correction
(least-squares projection onto "children sum to parent" via a fixed 0/1
taxonomy map, staged coarse->fine with detached anchors). `HccController`
wraps it with the epoch-dependent temperature/alpha schedule, raw/projected
blending, and diagnostics originally written inline in H-CAST's model class,
so every hierarchical model in this repo can reuse the same schedule instead
of reimplementing it. `select_effective_logits` is the shared "prefer
HCC-corrected per-level tensors over raw ones" helper used by each model's
`losses.py`.
"""

import math
from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
    """Affine-output projector enforcing z1=M12@z2 and z2=M23@z3 for 3-level hierarchies."""

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
            raise ValueError("taxonomy is required when hcc is enabled.")

        c1, c2, c3 = classes
        parent_of_middle, parent_of_fine = _extract_parent_arrays(taxonomy, c1=c1, c2=c2, c3=c3)
        m12 = _build_mapping_matrix(c1, parent_of_middle)
        m23 = _build_mapping_matrix(c2, parent_of_fine)

        self.num_classes_per_level = classes
        self.eps = float(eps)
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

    @staticmethod
    def _constraint_residual(
        child_scores: torch.Tensor,
        parent_scores: torch.Tensor,
        mapping: torch.Tensor,
    ) -> torch.Tensor:
        return child_scores @ mapping.transpose(0, 1).contiguous() - parent_scores

    def forward(
        self,
        logits_per_level: List[torch.Tensor],
    ) -> Dict[str, Any]:
        if len(logits_per_level) != 3:
            raise ValueError(f"Expected 3 logit tensors in coarse->fine order, got {len(logits_per_level)}.")

        z1, z2, z3 = logits_per_level
        if z1.ndim != 2 or z2.ndim != 2 or z3.ndim != 2:
            raise ValueError("Each logit tensor must be rank-2 [batch, classes].")
        if z1.shape[0] != z2.shape[0] or z2.shape[0] != z3.shape[0]:
            raise ValueError("All logit tensors must have the same batch size.")

        expected = self.num_classes_per_level
        current = [z1.shape[1], z2.shape[1], z3.shape[1]]
        if current != expected:
            raise ValueError(f"Logit shapes {current} do not match expected hierarchy classes {expected}.")

        compute_dtype = torch.float32 if z2.dtype in {torch.float16, torch.bfloat16} else z2.dtype
        autocast_off = nullcontext()
        if z2.device.type in {"cpu", "cuda", "xpu", "mps"}:
            autocast_off = torch.autocast(device_type=z2.device.type, enabled=False)

        with autocast_off:
            z1_work = z1.to(dtype=compute_dtype)
            z2_work = z2.to(dtype=compute_dtype)
            z3_work = z3.to(dtype=compute_dtype)
            # Stage-wise detached anchors:
            # - z1 is treated as fixed when projecting z2.
            # - z2_hat is treated as fixed when projecting z3.
            z1_anchor = z1_work.detach()
            m12 = self.m12.to(device=z2.device, dtype=compute_dtype)
            m23 = self.m23.to(device=z2.device, dtype=compute_dtype)

            # residual before projection for [M12 z2 - z1, M23 z3 - z2]
            residual_12_before = self._constraint_residual(z2_work, z1_anchor, m12)
            residual_23_before = self._constraint_residual(z3_work, z2_work, m23)
            residual_before = torch.cat([residual_12_before, residual_23_before], dim=1)

            gram_12 = m12 @ m12.transpose(0, 1).contiguous()
            eye_12 = torch.eye(gram_12.shape[0], dtype=compute_dtype, device=z2.device)
            coeff_12 = torch.linalg.solve(
                gram_12 + self.eps * eye_12,
                residual_12_before.transpose(0, 1).contiguous(),
            ).transpose(0, 1).contiguous()
            z2_hat = z2_work - coeff_12 @ m12

            z2_anchor = z2_hat.detach()
            residual_23_stage = self._constraint_residual(z3_work, z2_anchor, m23)
            gram_23 = m23 @ m23.transpose(0, 1).contiguous()
            eye_23 = torch.eye(gram_23.shape[0], dtype=compute_dtype, device=z2.device)
            coeff_23 = torch.linalg.solve(
                gram_23 + self.eps * eye_23,
                residual_23_stage.transpose(0, 1).contiguous(),
            ).transpose(0, 1).contiguous()
            z3_hat = z3_work - coeff_23 @ m23

            z1_hat = z1_work
            residual_12_after = self._constraint_residual(z2_hat, z1_hat, m12)
            residual_23_after = self._constraint_residual(z3_hat, z2_hat, m23)
            residual_after = torch.cat([residual_12_after, residual_23_after], dim=1)

        return {
            "projected_logits_per_level": [
                z1_hat.to(dtype=z1.dtype),
                z2_hat.to(dtype=z2.dtype),
                z3_hat.to(dtype=z3.dtype),
            ],
            "residual_before": residual_before,
            "residual_after": residual_after,
        }


def build_hcc_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = cfg or {}
    enabled = bool(cfg.get("enabled", False))
    if enabled and "temperature" not in cfg:
        raise ValueError(
            "hcc.temperature is required when hcc.enabled=true."
        )

    raw_temperature = cfg.get("temperature", 1.0)
    try:
        temperature = float(raw_temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("hcc.temperature must be a valid float.") from exc
    if enabled and temperature <= 0.0:
        raise ValueError(
            "hcc.temperature must be > 0 when hcc.enabled=true."
        )

    eps = float(cfg.get("eps", 1e-12))
    if eps <= 0.0:
        eps = 1e-12
    alpha_schedule = cfg.get("alpha_schedule", "exp")
    if not isinstance(alpha_schedule, str):
        raise ValueError("hcc.alpha_schedule must be a string.")
    if alpha_schedule not in {"exp", "tanh", "linear", "step"}:
        raise ValueError(
            "hcc.alpha_schedule must be one of ['exp', 'tanh', 'linear', 'step']."
        )
    try:
        alpha_start_epoch = int(cfg.get("alpha_start_epoch", 0))
    except (TypeError, ValueError):
        alpha_start_epoch = 0
    if alpha_start_epoch < 0:
        alpha_start_epoch = 0
    try:
        alpha_ramp_epochs = int(cfg.get("alpha_ramp_epochs", 0))
    except (TypeError, ValueError):
        alpha_ramp_epochs = 0
    if alpha_ramp_epochs <= 0:
        alpha_ramp_epochs = 0
    alpha_tanh_beta = float(cfg.get("alpha_tanh_beta", 3.0))
    if alpha_tanh_beta <= 0.0:
        alpha_tanh_beta = 3.0
    alpha_tanh_center = float(cfg.get("alpha_tanh_center", 0.5))
    if alpha_tanh_center <= 0.0 or alpha_tanh_center >= 1.0:
        alpha_tanh_center = 0.5
    final_test_only = bool(cfg.get("final_test_only", False))

    return {
        "enabled": enabled,
        "temperature": temperature,
        "eps": eps,
        "alpha_schedule": alpha_schedule,
        "alpha_start_epoch": alpha_start_epoch,
        "alpha_ramp_epochs": alpha_ramp_epochs,
        "alpha_tanh_beta": alpha_tanh_beta,
        "alpha_tanh_center": alpha_tanh_center,
        "final_test_only": final_test_only,
    }


def resolve_hcc_cfg_from_top_level(cfg: Any) -> Dict[str, Any]:
    """Read the canonical, model-agnostic top-level `hcc:` config section.

    Raises if a model config accidentally nests `hcc` under `model:`, since
    `hcc` is a top-level key shared across every model that supports it.
    """

    def _section_to_dict(section: Any) -> Dict[str, Any]:
        if section is None:
            return {}
        if isinstance(section, dict):
            return dict(section)
        if hasattr(section, "items"):
            return {k: v for k, v in section.items()}
        return {}

    hcc_cfg = _section_to_dict(getattr(cfg, "hcc", None))
    if hasattr(cfg, "get"):
        hcc_cfg = _section_to_dict(cfg.get("hcc", hcc_cfg))

    model_cfg = _section_to_dict(getattr(cfg, "model", None))
    if "hcc" in model_cfg:
        raise ValueError(
            "Unsupported key `model.hcc`. Use top-level `hcc` only."
        )
    return hcc_cfg


class HccController(nn.Module):
    """Owns an optional `HierarchicalAffineProjector` plus its schedule/blend/diagnostics state.

    Each hierarchical model composes one of these (`self.hcc = HccController(...)`)
    instead of reimplementing the epoch-dependent temperature/alpha ramp, blending,
    and diagnostics inline. Registering it as a submodule keeps the projector's
    buffers moving correctly with `.to(device)`/`.cuda()`.
    """

    def __init__(
        self,
        num_classes_per_level: List[int],
        taxonomy: Optional[Dict[str, Any]],
        hcc_cfg: Optional[Dict[str, Any]],
        train_epochs: int = 1,
    ):
        super().__init__()
        self.cfg = build_hcc_config(hcc_cfg)
        self._current_epoch = 0
        try:
            parsed_train_epochs = int(train_epochs)
        except (TypeError, ValueError):
            parsed_train_epochs = 1
        self._train_epochs = max(parsed_train_epochs, 1)
        self._final_test_active = False

        self.projector: Optional[HierarchicalAffineProjector] = None
        if self.cfg["enabled"]:
            depth = len(num_classes_per_level)
            if depth != 3:
                raise ValueError(
                    "hcc.enabled=true requires exactly 3 hierarchy levels "
                    f"(coarse->middle->fine), got {depth}."
                )
            if taxonomy is None:
                raise ValueError(
                    "hcc.enabled=true requires dataset taxonomy with parent-child mappings."
                )
            self.projector = HierarchicalAffineProjector(
                num_classes_per_level=num_classes_per_level,
                taxonomy=taxonomy,
                eps=float(self.cfg["eps"]),
            )

    def set_epoch(self, epoch: int) -> None:
        try:
            epoch_value = int(epoch)
        except (TypeError, ValueError):
            epoch_value = 0
        self._current_epoch = max(epoch_value, 0)

    def set_final_test_active(self, active: bool) -> None:
        self._final_test_active = bool(active)

    def enabled(self) -> bool:
        if not self.cfg["enabled"]:
            return False
        if self.projector is None:
            return False
        if self.cfg.get("final_test_only", False) and not self._final_test_active:
            return False
        return True

    def temperature_and_alpha(self) -> Tuple[float, float]:
        base_temperature = float(self.cfg["temperature"])
        eps = float(self.cfg["eps"])
        alpha_schedule = self.cfg.get("alpha_schedule", "exp")
        if not isinstance(alpha_schedule, str):
            raise ValueError("hcc.alpha_schedule must be a string.")
        alpha_start_epoch = int(self.cfg.get("alpha_start_epoch", 0))
        if alpha_start_epoch < 0:
            alpha_start_epoch = 0
        raw_alpha_ramp_epochs = int(self.cfg.get("alpha_ramp_epochs", 0))
        default_ramp_epochs = max(self._train_epochs - alpha_start_epoch, 1)
        alpha_ramp_epochs = raw_alpha_ramp_epochs if raw_alpha_ramp_epochs > 0 else default_ramp_epochs
        if alpha_ramp_epochs <= 0:
            alpha_ramp_epochs = 1
        alpha_tanh_beta = float(self.cfg.get("alpha_tanh_beta", 3.0))
        if alpha_tanh_beta <= 0.0:
            alpha_tanh_beta = 3.0
        alpha_tanh_center = float(self.cfg.get("alpha_tanh_center", 0.5))

        # For temperature <= 1, constraints are fully on from the start.
        if base_temperature <= 1.0:
            return 1.0, 1.0

        if self._current_epoch < alpha_start_epoch:
            return float(base_temperature), 0.0

        if alpha_schedule == "step":
            return 1.0, 1.0

        if alpha_ramp_epochs <= 1:
            progress = 1.0
        else:
            progress = float(self._current_epoch - alpha_start_epoch) / float(max(alpha_ramp_epochs - 1, 1))
            progress = min(max(progress, 0.0), 1.0)

        if alpha_schedule == "tanh":
            # Centered tanh sigmoid on progress in [0, 1]:
            # alpha(0)=0, alpha(center)=0.5, alpha(1)=1.
            center = min(max(alpha_tanh_center, eps), 1.0 - eps)
            if progress <= center:
                shifted_progress = 0.5 * (progress / max(center, eps))
            else:
                shifted_progress = 0.5 + (0.5 * ((progress - center) / max(1.0 - center, eps)))
            beta = max(alpha_tanh_beta, eps)
            tanh_beta = math.tanh(beta)
            denom = 2.0 * tanh_beta
            if abs(denom) <= eps:
                alpha = shifted_progress
            else:
                centered_progress = (2.0 * shifted_progress) - 1.0
                alpha = (math.tanh(beta * centered_progress) + tanh_beta) / denom
            temperature = base_temperature - (alpha * (base_temperature - 1.0))
        elif alpha_schedule == "linear":
            alpha = progress
            temperature = base_temperature - (alpha * (base_temperature - 1.0))
        else:
            temperature = base_temperature ** (1.0 - progress)
            alpha = (base_temperature - temperature) / max(base_temperature - 1.0, eps)

        alpha = min(max(alpha, 0.0), 1.0)
        return float(temperature), float(alpha)

    @staticmethod
    def blend(
        logits_per_level: List[torch.Tensor],
        projected_logits_per_level: List[torch.Tensor],
        alpha: float,
        eps: float,
    ) -> List[torch.Tensor]:
        if alpha <= eps:
            # Start without any projection effect.
            return logits_per_level
        if alpha >= (1.0 - eps):
            return projected_logits_per_level
        return [
            ((1.0 - alpha) * logits) + (alpha * projected)
            for logits, projected in zip(logits_per_level, projected_logits_per_level)
        ]

    @staticmethod
    def diagnostics(
        temperature: float,
        alpha: float,
        projector_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        diagnostics = {
            "proj_temperature": float(temperature),
            "proj_constraint_alpha": float(alpha),
        }
        if isinstance(projector_output, dict):
            residual_before = projector_output.get("residual_before")
            residual_after = projector_output.get("residual_after")
            if isinstance(residual_before, torch.Tensor) and residual_before.numel() > 0:
                diagnostics["proj_logit_residual_before_l1"] = float(residual_before.abs().mean().item())
            if isinstance(residual_after, torch.Tensor) and residual_after.numel() > 0:
                diagnostics["proj_logit_residual_after_l1"] = float(residual_after.abs().mean().item())
            if "proj_logit_residual_before_l1" in diagnostics and "proj_logit_residual_after_l1" in diagnostics:
                before = diagnostics["proj_logit_residual_before_l1"]
                after = diagnostics["proj_logit_residual_after_l1"]
                diagnostics["proj_logit_residual_reduction"] = float(before - after)
        return diagnostics

    def apply(self, logits_per_level: List[torch.Tensor]) -> Dict[str, Any]:
        """Run one HCC step over coarse->fine per-level logits for the current batch.

        Returns `projected_logits_per_level`/`effective_logits_per_level`/
        `hcc_diagnostics`, all `None` when HCC is disabled or hasn't activated
        yet for this batch (pre-activation epochs stay identical to the
        uncorrected baseline model).
        """
        projected_logits_per_level = None
        effective_logits_per_level = None
        hcc_diagnostics = None
        if self.enabled():
            if self.cfg.get("final_test_only", False) and self._final_test_active:
                temperature, alpha = 1.0, 1.0
            else:
                temperature, alpha = self.temperature_and_alpha()
            eps = float(self.cfg["eps"])
            projector_output = None

            # Keep pre-activation batches identical to the uncorrected baseline.
            # Downstream loss/metrics use raw logits whenever effective logits are absent.
            if alpha > eps:
                projector_output = self.projector(logits_per_level)
                projected_logits_per_level = projector_output["projected_logits_per_level"]
                effective_logits_per_level = self.blend(
                    logits_per_level=logits_per_level,
                    projected_logits_per_level=projected_logits_per_level,
                    alpha=alpha,
                    eps=eps,
                )

            with torch.no_grad():
                hcc_diagnostics = self.diagnostics(
                    temperature=temperature,
                    alpha=alpha,
                    projector_output=projector_output,
                )

        return {
            "projected_logits_per_level": projected_logits_per_level,
            "effective_logits_per_level": effective_logits_per_level,
            "hcc_diagnostics": hcc_diagnostics,
        }


def select_effective_logits(
    output: Dict[str, Any],
    raw_key: str = "logits_per_level",
    effective_key: str = "effective_logits_per_level",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Prefer HCC-corrected per-level tensors over raw ones when present.

    Returns `(raw_list, score_source_list)`. `score_source_list` is `raw_list`
    itself before HCC activates or when HCC is absent/disabled.
    """
    raw = output[raw_key]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected non-empty list at output['{raw_key}'].")

    effective = output.get(effective_key)
    if effective is not None:
        if not isinstance(effective, list) or len(effective) != len(raw):
            raise ValueError(
                f"output['{effective_key}'] must be None or a list aligned with output['{raw_key}']."
            )
        return raw, effective

    return raw, raw
