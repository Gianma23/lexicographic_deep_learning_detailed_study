import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hard_hierarchy import HierarchicalAffineProjector
from .segments import build_seeds_segments, supports_seeds

try:
    # Import registers H-CAST variants into timm's global model registry.
    from . import internal as _hcast_internal  # noqa: F401
except Exception:  # pragma: no cover
    _hcast_internal = None

try:
    from timm import create_model as timm_create_model
except Exception:  # pragma: no cover
    timm_create_model = None


class HCASTModel(nn.Module):
    """Adapter that builds H-CAST via timm and normalizes output format."""

    def __init__(
        self,
        num_classes_per_level: List[int],
        variant: str = "cast_small",
        model_kwargs: Optional[Dict[str, Any]] = None,
        segments_cfg: Optional[Dict[str, Any]] = None,
        taxonomy: Optional[Dict[str, Any]] = None,
        hcc_cfg: Optional[Dict[str, Any]] = None,
        train_epochs: int = 1,
    ):
        super().__init__()
        self.num_classes_per_level = list(num_classes_per_level)
        self.depth = len(self.num_classes_per_level)

        if self.depth < 2:
            raise ValueError("H-CAST requires at least 2 hierarchy levels.")

        model_kwargs = model_kwargs or {}
        segments_cfg = segments_cfg or {}
        self.segment_mode = segments_cfg.get("mode", "grid")
        if not isinstance(self.segment_mode, str):
            raise ValueError("model.segments.mode must be a string.")
        if self.segment_mode not in {"grid", "seeds"}:
            raise ValueError("model.segments.mode must be one of ['grid', 'seeds'].")
        if self.segment_mode == "seeds" and not supports_seeds():
            raise RuntimeError(
                "H-CAST SEEDS mode requires OpenCV ximgproc with "
                "createSuperpixelSEEDS. Install opencv-contrib-python; "
                "SEEDS mode does not fall back to patch-grid segments."
            )
        self.segment_patch_size = int(segments_cfg.get("patch_size", 8))
        self.segment_mean = list(segments_cfg.get("mean", [0.485, 0.456, 0.406]))
        self.segment_std = list(segments_cfg.get("std", [0.229, 0.224, 0.225]))
        self.seeds_num_superpixels = int(segments_cfg.get("num_superpixels", 196))
        self.seeds_num_levels = int(segments_cfg.get("num_levels", 1))
        self.seeds_prior = int(segments_cfg.get("prior", 2))
        self.seeds_histogram_bins = int(segments_cfg.get("histogram_bins", 5))
        self.seeds_double_step = bool(segments_cfg.get("double_step", False))
        self.seeds_num_iterations = int(segments_cfg.get("num_iterations", 15))
        self._current_epoch = 0
        try:
            parsed_train_epochs = int(train_epochs)
        except (TypeError, ValueError):
            parsed_train_epochs = 1
        self._train_epochs = max(parsed_train_epochs, 1)
        self.hcc_cfg = self._build_hcc_cfg(hcc_cfg)
        self._hcc_final_test_active = False
        self.hcc_projector: Optional[HierarchicalAffineProjector] = None

        if self.hcc_cfg["enabled"]:
            if self.depth != 3:
                raise ValueError(
                    "hcc.enabled=true requires exactly 3 hierarchy levels "
                    f"(coarse->middle->fine), got {self.depth}."
                )
            if taxonomy is None:
                raise ValueError(
                    "hcc.enabled=true requires dataset taxonomy with parent-child mappings."
                )
            self.hcc_projector = HierarchicalAffineProjector(
                num_classes_per_level=self.num_classes_per_level,
                taxonomy=taxonomy,
                eps=float(self.hcc_cfg["eps"]),
            )

        if timm_create_model is None:
            raise ImportError("HCASTModel requires timm to be installed, but timm.create_model is unavailable.")

        # Upstream H-CAST expects classes from fine->coarse order.
        self.nb_classes_upstream = list(reversed(self.num_classes_per_level))
        timm_kwargs = dict(model_kwargs)
        pretrained = bool(timm_kwargs.pop("pretrained", False))
        img_size = int(timm_kwargs.pop("img_size", 224))
        drop_rate = float(timm_kwargs.pop("drop_rate", 0.0))
        drop_path_rate = float(timm_kwargs.pop("drop_path_rate", 0.1))
        drop_block_rate = timm_kwargs.pop("drop_block_rate", None)
        num_classes = int(self.nb_classes_upstream[0]) if self.nb_classes_upstream else 0
        self.model = timm_create_model(
            variant,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            drop_block_rate=drop_block_rate,
            img_size=img_size,
            nb_classes=self.nb_classes_upstream,
            **timm_kwargs,
        )

    @staticmethod
    def _build_hcc_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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

    def set_epoch(self, epoch: int) -> None:
        try:
            epoch_value = int(epoch)
        except (TypeError, ValueError):
            epoch_value = 0
        self._current_epoch = max(epoch_value, 0)

    def set_hcc_final_test_active(self, active: bool) -> None:
        self._hcc_final_test_active = bool(active)

    def _hcc_enabled(self) -> bool:
        if not self.hcc_cfg["enabled"]:
            return False
        if self.hcc_projector is None:
            return False
        if self.hcc_cfg.get("final_test_only", False) and not self._hcc_final_test_active:
            return False
        return True

    def _hcc_temperature_and_alpha(self) -> Tuple[float, float]:
        base_temperature = float(self.hcc_cfg["temperature"])
        eps = float(self.hcc_cfg["eps"])
        alpha_schedule = self.hcc_cfg.get("alpha_schedule", "exp")
        if not isinstance(alpha_schedule, str):
            raise ValueError("hcc.alpha_schedule must be a string.")
        alpha_start_epoch = int(self.hcc_cfg.get("alpha_start_epoch", 0))
        if alpha_start_epoch < 0:
            alpha_start_epoch = 0
        raw_alpha_ramp_epochs = int(self.hcc_cfg.get("alpha_ramp_epochs", 0))
        default_ramp_epochs = max(self._train_epochs - alpha_start_epoch, 1)
        alpha_ramp_epochs = raw_alpha_ramp_epochs if raw_alpha_ramp_epochs > 0 else default_ramp_epochs
        if alpha_ramp_epochs <= 0:
            alpha_ramp_epochs = 1
        alpha_tanh_beta = float(self.hcc_cfg.get("alpha_tanh_beta", 3.0))
        if alpha_tanh_beta <= 0.0:
            alpha_tanh_beta = 3.0
        alpha_tanh_center = float(self.hcc_cfg.get("alpha_tanh_center", 0.5))

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
    def _blend_hcc_logits(
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
    def _hcc_diagnostics(
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

    @staticmethod
    def _build_grid_segments(images: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
        """Generate deterministic patch-aligned segment ids for H-CAST."""
        bsz, _, h, w = images.shape
        gh = max(1, h // patch_size)
        gw = max(1, w // patch_size)
        grid = torch.arange(gh * gw, device=images.device, dtype=torch.float32).view(1, 1, gh, gw)
        segments = F.interpolate(grid, size=(h, w), mode="nearest").squeeze(1).long()
        return segments.expand(bsz, -1, -1).contiguous()

    def _build_segments(self, images: torch.Tensor) -> torch.Tensor:
        if self.segment_mode == "seeds":
            segments = build_seeds_segments(
                images=images,
                mean=self.segment_mean,
                std=self.segment_std,
                num_superpixels=self.seeds_num_superpixels,
                num_levels=self.seeds_num_levels,
                prior=self.seeds_prior,
                histogram_bins=self.seeds_histogram_bins,
                double_step=self.seeds_double_step,
                num_iterations=self.seeds_num_iterations,
            )
            if segments is not None:
                return segments
            if not supports_seeds():
                raise RuntimeError(
                    "H-CAST SEEDS became unavailable at runtime. "
                    "SEEDS mode does not fall back to patch-grid segments."
                )
            raise RuntimeError(
                "H-CAST SEEDS generation failed for the current input. "
                "Expected RGB images and three matching normalization mean/std values; "
                "SEEDS mode does not fall back to patch-grid segments."
            )

        return self._build_grid_segments(images, patch_size=self.segment_patch_size)

    def forward(
        self,
        x: torch.Tensor,
        segments: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if segments is None:
            segments = self._build_segments(x)

        raw = self.model(x, segments)
        if not isinstance(raw, (tuple, list)):
            raw = (raw,)

        # Upstream order is fine->coarse; unified API is coarse->fine.
        logits_per_level = list(reversed(list(raw)))

        projected_logits_per_level = None
        effective_logits_per_level = None
        hcc_diagnostics = None
        if self._hcc_enabled():
            if self.hcc_cfg.get("final_test_only", False) and self._hcc_final_test_active:
                temperature, alpha = 1.0, 1.0
            else:
                temperature, alpha = self._hcc_temperature_and_alpha()
            eps = float(self.hcc_cfg["eps"])
            projector_output = None

            # Keep pre-activation epochs identical to baseline H-CAST.
            # Downstream loss/metrics use raw logits whenever effective logits are absent.
            if alpha > eps:
                projector_output = self.hcc_projector(logits_per_level)
                projected_logits_per_level = projector_output["projected_logits_per_level"]
                effective_logits_per_level = self._blend_hcc_logits(
                    logits_per_level=logits_per_level,
                    projected_logits_per_level=projected_logits_per_level,
                    alpha=alpha,
                    eps=eps,
                )

            with torch.no_grad():
                hcc_diagnostics = self._hcc_diagnostics(
                    temperature=temperature,
                    alpha=alpha,
                    projector_output=projector_output,
                )

        return {
            "logits_per_level": logits_per_level,
            "orthonormal_plugin_scores_per_level": logits_per_level,
            "projected_logits_per_level": projected_logits_per_level,
            "effective_logits_per_level": effective_logits_per_level,
            "hcc_diagnostics": hcc_diagnostics,
        }
