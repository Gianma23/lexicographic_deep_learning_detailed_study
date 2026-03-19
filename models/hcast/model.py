import warnings
from typing import Any, Dict, List, Optional

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
        design1_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.num_classes_per_level = list(num_classes_per_level)
        self.depth = len(self.num_classes_per_level)

        if self.depth < 2:
            raise ValueError("H-CAST requires at least 2 hierarchy levels.")

        model_kwargs = model_kwargs or {}
        segments_cfg = segments_cfg or {}
        self.segment_mode = str(segments_cfg.get("mode", "grid")).strip().lower()
        self.segment_patch_size = int(segments_cfg.get("patch_size", 8))
        self.segment_mean = list(segments_cfg.get("mean", [0.485, 0.456, 0.406]))
        self.segment_std = list(segments_cfg.get("std", [0.229, 0.224, 0.225]))
        self.seeds_num_superpixels = int(segments_cfg.get("num_superpixels", 196))
        self.seeds_num_levels = int(segments_cfg.get("num_levels", 1))
        self.seeds_prior = int(segments_cfg.get("prior", 2))
        self.seeds_histogram_bins = int(segments_cfg.get("histogram_bins", 5))
        self.seeds_double_step = bool(segments_cfg.get("double_step", False))
        self.seeds_num_iterations = int(segments_cfg.get("num_iterations", 15))
        self._seeds_warned = False
        self.design1_cfg = self._build_design1_cfg(design1_cfg)
        self.design1_projector: Optional[HierarchicalAffineProjector] = None

        if self.design1_cfg["enabled"]:
            if self.depth != 3:
                raise ValueError(
                    "cfg.model.design1.enabled=true requires exactly 3 hierarchy levels "
                    f"(coarse->middle->fine), got {self.depth}."
                )
            if taxonomy is None:
                raise ValueError(
                    "cfg.model.design1.enabled=true requires dataset taxonomy with parent-child mappings."
                )
            self.design1_projector = HierarchicalAffineProjector(
                num_classes_per_level=self.num_classes_per_level,
                taxonomy=taxonomy,
                eps=float(self.design1_cfg["eps"]),
            )

        if timm_create_model is None:
            raise ImportError("HCASTModel requires timm to be installed, but timm.create_model is unavailable.")

        # Upstream H-CAST expects classes from fine->coarse order.
        self.nb_classes_upstream = list(reversed(self.num_classes_per_level))
        timm_kwargs = dict(model_kwargs)
        pretrained = bool(timm_kwargs.pop("pretrained", False))
        self.model = timm_create_model(
            variant,
            pretrained=pretrained,
            nb_classes=self.nb_classes_upstream,
            **timm_kwargs,
        )

    @staticmethod
    def _build_design1_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = cfg or {}
        scope = str(cfg.get("scope", "train_eval")).strip().lower()
        if scope not in {"train_eval", "train", "eval"}:
            scope = "train_eval"
        eps = float(cfg.get("eps", 1e-12))
        if eps <= 0.0:
            eps = 1e-12
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "scope": scope,
            "use_projected_for_loss": bool(cfg.get("use_projected_for_loss", False)),
            "eps": eps,
        }

    def _design1_active(self) -> bool:
        if not self.design1_cfg["enabled"]:
            return False
        if self.design1_projector is None:
            return False
        scope = self.design1_cfg["scope"]
        if scope == "train" and not self.training:
            return False
        if scope == "eval" and self.training:
            return False
        return True

    @staticmethod
    def _build_grid_segments(images: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
        """Generate deterministic patch-aligned segment ids for H-CAST."""
        bsz, _, h, w = images.shape
        gh = max(1, h // patch_size)
        gw = max(1, w // patch_size)
        grid = torch.arange(gh * gw, device=images.device, dtype=torch.float32).view(1, 1, gh, gw)
        segments = F.interpolate(grid, size=(h, w), mode="nearest").squeeze(1).long()
        return segments.expand(bsz, -1, -1).contiguous()

    def _build_default_segments(self, images: torch.Tensor) -> torch.Tensor:
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
            if not self._seeds_warned:
                self._seeds_warned = True
                if not supports_seeds():
                    warnings.warn(
                        "H-CAST SEEDS mode requested, but OpenCV ximgproc is unavailable. "
                        "Falling back to patch-grid segments.",
                        RuntimeWarning,
                    )
                else:
                    warnings.warn(
                        "H-CAST SEEDS mode requested, but SEEDS generation failed on this input. "
                        "Falling back to patch-grid segments.",
                        RuntimeWarning,
                    )

        return self._build_grid_segments(images, patch_size=self.segment_patch_size)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        segments: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if segments is None:
            segments = self._build_default_segments(x)

        upstream_targets = None
        if isinstance(targets, torch.Tensor) and targets.ndim == 2 and targets.shape[1] >= self.depth:
            # Unified order is coarse->fine while upstream CAST uses fine->coarse.
            upstream_targets = torch.flip(targets[:, :self.depth], dims=[1]).contiguous()

        raw = self.model(x, segments, targets=upstream_targets)
        if not isinstance(raw, (tuple, list)):
            raw = (raw,)

        # Upstream order is fine->coarse; unified API is coarse->fine.
        logits_per_level = list(reversed(list(raw)))
        probs_per_level = [F.softmax(logits, dim=-1) for logits in logits_per_level]
        design1_active = self._design1_active()
        projected_probs_per_level = None
        design1_stats = None
        if design1_active:
            projector_output = self.design1_projector(probs_per_level)
            projected_probs_per_level = projector_output["projected_probs_per_level"]
            residual_before = projector_output["residual_before"]
            residual_after = projector_output["residual_after"]
            design1_stats = {
                "residual_before_max": float(residual_before.abs().max().detach().item()),
                "residual_after_max": float(residual_after.abs().max().detach().item()),
            }

        return {
            "logits_per_level": logits_per_level,
            "probs_per_level": probs_per_level,
            "projected_probs_per_level": projected_probs_per_level,
            "design1_active": design1_active,
            "design1_stats": design1_stats,
            "segments": segments,
        }
