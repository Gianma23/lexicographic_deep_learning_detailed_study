from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.hcc import HccController
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
        self.hcc = HccController(
            num_classes_per_level=self.num_classes_per_level,
            taxonomy=taxonomy,
            hcc_cfg=hcc_cfg,
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

    def set_hcc_final_test_active(self, active: bool) -> None:
        self.hcc.set_final_test_active(active)

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

        hcc_output = self.hcc.apply(logits_per_level)

        return {
            "logits_per_level": logits_per_level,
            "orthonormal_plugin_scores_per_level": logits_per_level,
            **hcc_output,
        }
