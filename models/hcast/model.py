from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .internal import cast_base, cast_base_deep, cast_small, cast_small_deep
except Exception:  # pragma: no cover
    cast_base = cast_base_deep = cast_small = cast_small_deep = None


class HCASTLite(nn.Module):
    """Fallback model when timm/DGL stack is unavailable."""

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
    """Unified wrapper around upstream H-CAST implementation."""

    _VARIANT_MAP = {
        "cast_small": cast_small,
        "cast_small_deep": cast_small_deep,
        "cast_base": cast_base,
        "cast_base_deep": cast_base_deep,
    }

    def __init__(self, num_classes_per_level: List[int], variant: str = "cast_small", fallback_cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.num_classes_per_level = list(num_classes_per_level)
        self.depth = len(self.num_classes_per_level)

        if self.depth < 2:
            raise ValueError("H-CAST requires at least 2 hierarchy levels.")

        fallback_cfg = fallback_cfg or {}
        factory = self._VARIANT_MAP.get(variant, cast_small)
        self._use_fallback = factory is None

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
            self.model = factory(nb_classes=self.nb_classes_upstream)
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
    def _build_default_segments(images: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
        """Generate deterministic patch-aligned segment ids for H-CAST."""
        bsz, _, h, w = images.shape
        gh = max(1, h // patch_size)
        gw = max(1, w // patch_size)
        grid = torch.arange(gh * gw, device=images.device, dtype=torch.float32).view(1, 1, gh, gw)
        segments = F.interpolate(grid, size=(h, w), mode="nearest").squeeze(1).long()
        return segments.expand(bsz, -1, -1).contiguous()

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
