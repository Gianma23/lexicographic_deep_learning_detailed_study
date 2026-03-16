import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class HCASTLite(nn.Module):
    """Fallback model when the full timm-backed H-CAST stack is unavailable."""

    def __init__(self, num_classes_per_level: List[int], hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        from torchvision.models import resnet18

        self.num_classes_per_level = num_classes_per_level
        backbone = resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, c) for c in num_classes_per_level])

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        _ = targets
        feat = self.feature_extractor(x).flatten(1)
        z = self.proj(feat)
        logits_per_level = [head(z) for head in self.heads]
        probs_per_level = [F.softmax(logits, dim=-1) for logits in logits_per_level]
        return {"logits_per_level": logits_per_level, "probs_per_level": probs_per_level, "features": z}


class HCASTModel(nn.Module):
    """Adapter that builds H-CAST via timm and normalizes output format."""

    def __init__(
        self,
        num_classes_per_level: List[int],
        variant: str = "cast_small",
        fallback_cfg: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        segments_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.num_classes_per_level = list(num_classes_per_level)
        self.depth = len(self.num_classes_per_level)

        if self.depth < 2:
            raise ValueError("H-CAST requires at least 2 hierarchy levels.")

        fallback_cfg = fallback_cfg or {}
        model_kwargs = model_kwargs or {}
        segments_cfg = segments_cfg or {}

        self._use_fallback = timm_create_model is None
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

        if self._use_fallback:
            self.fallback = HCASTLite(
                num_classes_per_level=self.num_classes_per_level,
                hidden_dim=int(fallback_cfg.get("hidden_dim", 512)),
                dropout=float(fallback_cfg.get("dropout", 0.2)),
            )
            self.model = None
            return

        # Upstream H-CAST expects classes from fine->coarse order.
        self.nb_classes_upstream = list(reversed(self.num_classes_per_level))
        try:
            timm_kwargs = dict(model_kwargs)
            pretrained = bool(timm_kwargs.pop("pretrained", False))
            self.model = timm_create_model(
                variant,
                pretrained=pretrained,
                nb_classes=self.nb_classes_upstream,
                **timm_kwargs,
            )
            self.fallback = None
        except Exception:
            self._use_fallback = True
            self.model = None
            self.fallback = HCASTLite(
                num_classes_per_level=self.num_classes_per_level,
                hidden_dim=int(fallback_cfg.get("hidden_dim", 512)),
                dropout=float(fallback_cfg.get("dropout", 0.2)),
            )

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
        if self._use_fallback:
            return self.fallback(x, targets=targets)

        if segments is None:
            segments = self._build_default_segments(x)

        raw = self.model(x, segments)
        if not isinstance(raw, (tuple, list)):
            raw = (raw,)

        # Upstream order is fine->coarse; unified API is coarse->fine.
        logits_per_level = list(reversed(list(raw)))
        probs_per_level = [F.softmax(logits, dim=-1) for logits in logits_per_level]

        return {
            "logits_per_level": logits_per_level,
            "probs_per_level": probs_per_level,
            "segments": segments,
        }
