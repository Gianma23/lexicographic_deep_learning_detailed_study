import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routing import (
    hierarchical_agreement,
    safe_norm,
    squash,
    taxonomy_guided_routing_weights,
)


@dataclass(frozen=True)
class _CapsuleLevelSpec:
    """Static shape description for one routing level."""

    n_in: int
    d_in: int
    n_out: int
    d_out: int


class _ConvBackbone(nn.Module):
    def __init__(self, in_ch: int = 3, num_blocks: int = 4, initial_filters: int = 64, filter_increment: int = 2):
        super().__init__()
        layers = []
        c_in = in_ch
        for i in range(num_blocks):
            c_out = initial_filters * (filter_increment ** i)
            layers.extend(
                [
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TimmFeatureBackbone(nn.Module):
    """Wrap timm `features_only` models and return the last feature map."""

    def __init__(self, timm_model: nn.Module):
        super().__init__()
        self.timm_model = timm_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.timm_model(x)
        if isinstance(feats, (list, tuple)):
            if not feats:
                raise RuntimeError("timm features_only backbone returned no feature maps.")
            return feats[-1]
        return feats


def _safe_num_heads(embed_dim: int, requested_heads: int) -> int:
    """Return a valid `MultiheadAttention` head count for `embed_dim`."""
    req = max(1, int(requested_heads))
    max_heads = max(1, int(embed_dim))
    req = min(req, max_heads)
    for heads in range(req, 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


def _normalize_attn_postprocess(value: Optional[str]) -> str:
    if value is None:
        text = "layernorm"
    else:
        if not isinstance(value, str):
            raise ValueError("HT-CapsNet model.attn_postprocess must be a string.")
        text = value
    if text == "layernorm":
        return "layernorm"
    if text == "squash":
        return "squash"
    raise ValueError(
        f"Unsupported HT-CapsNet attention postprocess '{value}'. "
        "Supported values: layernorm, squash."
    )


def _normalize_backbone_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("HT-CapsNet model.backbone_net must be a string.")
    value = name
    if value == "custom":
        return "custom"
    if value == "efficientnet_b7":
        return "efficientnet_b7"
    raise ValueError(
        f"Unsupported HT-CapsNet backbone '{name}'. "
        "Supported backbones: custom, efficientnet_b7."
    )


def _normalize_backbone_weights(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("HT-CapsNet model.backbone_net_weights must be a string or null.")
    text = value
    if text == "none":
        return None
    if text == "imagenet":
        return "imagenet"
    raise ValueError(
        f"Unsupported HT-CapsNet backbone weights '{value}'. "
        "Supported values: imagenet, none."
    )


def _build_backbone(
    backbone_name: str,
    backbone_weights: Optional[str],
    num_blocks: int,
    initial_filters: int,
    filter_increment: int,
) -> nn.Module:
    normalized_name = _normalize_backbone_name(backbone_name)
    normalized_weights = _normalize_backbone_weights(backbone_weights)

    if normalized_name == "custom":
        return _ConvBackbone(
            in_ch=3,
            num_blocks=num_blocks,
            initial_filters=initial_filters,
            filter_increment=filter_increment,
        )

    if normalized_name == "efficientnet_b7":
        # Prefer timm for random initialization; use torchvision when
        # explicit ImageNet pretrained weights are requested.
        timm_exc: Optional[Exception] = None
        if normalized_weights is None:
            try:
                import timm

                for candidate in ("tf_efficientnet_b7", "efficientnet_b7"):
                    try:
                        timm_model = timm.create_model(candidate, pretrained=False, features_only=True)
                        return _TimmFeatureBackbone(timm_model)
                    except Exception:
                        continue
                raise RuntimeError("No compatible timm EfficientNet-B7 model variant available.")
            except Exception as exc:
                timm_exc = exc

        from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

        weights = None
        if normalized_weights == "imagenet":
            weights = EfficientNet_B7_Weights.IMAGENET1K_V1

        try:
            model = efficientnet_b7(weights=weights)
        except Exception as tv_exc:
            if weights is None:
                if timm_exc is not None:
                    warnings.warn(
                        "timm EfficientNet-B7 (non-pretrained) initialization failed and torchvision fallback also failed. "
                        f"timm error: {timm_exc} | torchvision error: {tv_exc}",
                        RuntimeWarning,
                    )
                raise
            warnings.warn(
                "Failed to load EfficientNet-B7 ImageNet weights. "
                f"Falling back to random initialization. Original error: {tv_exc}",
                RuntimeWarning,
            )
            model = efficientnet_b7(weights=None)
        return model.features

    raise ValueError(f"Unsupported HT-CapsNet backbone '{backbone_name}'.")


class HTCapsNet(nn.Module):
    """PyTorch HT-CapsNet aligned with the original repo's TensorFlow architecture."""

    def __init__(
        self,
        num_classes_per_level: List[int],
        taxonomy: Optional[Dict[str, Any]] = None,
        primary_dim: int = 16,
        secondary_dims: Optional[List[int]] = None,
        routing_iters: int = 3,
        num_blocks: int = 4,
        initial_filters: int = 64,
        filter_increment: int = 2,
        backbone_name: str = "custom",
        backbone_weights: Optional[str] = None,
        taxonomy_temperature: float = 0.5,
        mask_threshold_high: float = 0.9,
        mask_threshold_low: float = 0.1,
        mask_temperature: float = 0.5,
        mask_center: float = 0.5,
        attn_heads: int = 16,
        attn_dropout: float = 0.0,
        attn_postprocess: str = "layernorm",
        input_size: int = 224,
    ):
        super().__init__()
        self.num_classes_per_level = [int(v) for v in num_classes_per_level]
        self.depth = len(self.num_classes_per_level)
        self.routing_iters = int(routing_iters)
        self.primary_dim = int(primary_dim)
        self.attn_postprocess = _normalize_attn_postprocess(attn_postprocess)

        if self.depth < 2:
            raise ValueError("HT-CapsNet expects at least 2 hierarchy levels.")

        if secondary_dims is None:
            secondary_dims = [self.primary_dim for _ in self.num_classes_per_level]
        if len(secondary_dims) != self.depth:
            raise ValueError("secondary_dims must match hierarchy depth.")
        self.secondary_dims = [int(v) for v in secondary_dims]
        if any(dim <= 0 for dim in self.secondary_dims):
            raise ValueError("secondary_dims values must be positive.")

        self.backbone = _build_backbone(
            backbone_name=backbone_name,
            backbone_weights=backbone_weights,
            num_blocks=num_blocks,
            initial_filters=initial_filters,
            filter_increment=filter_increment,
        )

        # Infer the primary-capsule layout from the backbone feature shape.
        # Probe in eval mode so BatchNorm layers do not update running stats
        # from a single-item dummy batch during initialization.
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, int(input_size), int(input_size))
                feat = self.backbone(self._prepare_backbone_input(dummy))
                flat_dim = int(feat[0].numel())
        finally:
            self.backbone.train(was_training)

        if flat_dim % self.primary_dim != 0:
            raise ValueError(
                f"Backbone flattened feature dim ({flat_dim}) must be divisible by primary_dim ({self.primary_dim})."
            )
        self.primary_caps_count = int(flat_dim // self.primary_dim)
        self._primary_caps_flat_dim = int(self.primary_caps_count * self.primary_dim)
        self.level_specs = self._build_level_specs()

        self.W = nn.ParameterList()
        self.h_gates = nn.ParameterList()
        self.dim_transforms = nn.ParameterList()
        self.post_attn_norms = nn.ModuleList()

        for level, spec in enumerate(self.level_specs):
            if level > 0:
                gate = nn.Parameter(torch.full((1, spec.n_out, self.num_classes_per_level[level - 1]), 0.5))
                dim_t = nn.Parameter(torch.randn(spec.d_in, spec.d_out) * 0.1)
                self.h_gates.append(gate)
                self.dim_transforms.append(dim_t)

            w_level = nn.Parameter(torch.randn(1, spec.n_in, spec.n_out, spec.d_out, spec.d_in) * 0.1)
            self.W.append(w_level)
            self.post_attn_norms.append(nn.LayerNorm(spec.d_out))

        self.attn_layers = nn.ModuleList()
        for level, spec in enumerate(self.level_specs):
            q_dim = spec.d_out
            kv_dim = spec.d_in if level > 0 else q_dim
            heads = _safe_num_heads(q_dim, attn_heads)
            if heads != int(attn_heads):
                warnings.warn(
                    f"Adjusted attn_heads from {int(attn_heads)} to {heads} for level {level} "
                    f"(embed_dim={q_dim}) to satisfy PyTorch MultiheadAttention constraints.",
                    RuntimeWarning,
                )
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=q_dim,
                    num_heads=heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    kdim=kv_dim,
                    vdim=kv_dim,
                )
            )

        self.taxonomy_temperature = float(taxonomy_temperature)
        self.mask_threshold_high = float(mask_threshold_high)
        self.mask_threshold_low = float(mask_threshold_low)
        self.mask_temperature = float(mask_temperature)
        self.mask_center = float(mask_center)
        self.parent_of = self._normalize_parent_of(taxonomy)

    @staticmethod
    def _normalize_parent_of(taxonomy: Optional[Dict[str, Any]]) -> Dict[int, Dict[int, int]]:
        if not taxonomy or "parent_of" not in taxonomy:
            return {}
        out: Dict[int, Dict[int, int]] = {}
        for k, v in taxonomy["parent_of"].items():
            out[int(k)] = {int(ck): int(pk) for ck, pk in v.items()}
        return out

    def _primary_caps_count_for_target_dim(self, target_dim: int) -> int:
        if target_dim <= 0:
            raise ValueError(f"Target capsule dim must be positive, got {target_dim}.")
        if self._primary_caps_flat_dim % target_dim != 0:
            raise ValueError(
                f"Primary capsule flattened dim ({self._primary_caps_flat_dim}) must be divisible by target dim ({target_dim})."
            )
        return int(self._primary_caps_flat_dim // target_dim)

    def _build_level_specs(self) -> List[_CapsuleLevelSpec]:
        specs: List[_CapsuleLevelSpec] = []
        for level in range(self.depth):
            n_out = int(self.num_classes_per_level[level])
            d_out = int(self.secondary_dims[level])
            if level == 0:
                specs.append(
                    _CapsuleLevelSpec(
                        n_in=int(self.primary_caps_count),
                        d_in=int(self.primary_dim),
                        n_out=n_out,
                        d_out=d_out,
                    )
                )
                continue

            d_prev = int(self.secondary_dims[level - 1])
            new_n_caps = self._primary_caps_count_for_target_dim(d_prev)
            specs.append(
                _CapsuleLevelSpec(
                    n_in=int(new_n_caps + self.num_classes_per_level[level - 1]),
                    d_in=d_prev,
                    n_out=n_out,
                    d_out=d_out,
                )
            )
        return specs

    @staticmethod
    def _prepare_backbone_input(x: torch.Tensor) -> torch.Tensor:
        """Mirror upstream pre-backbone guards for small or grayscale inputs."""
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor [B, C, H, W], got shape {tuple(x.shape)}.")

        channels = int(x.size(1))
        if channels == 1:
            x = x.repeat(1, 3, 1, 1)
        elif channels != 3:
            raise ValueError(f"HT-CapsNet expects 1 or 3 input channels, got {channels}.")

        if int(x.size(-2)) < 32 or int(x.size(-1)) < 32:
            x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        return x

    def _taxonomy_matrix(self, level: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if level <= 0:
            return None
        mapping = self.parent_of.get(level)
        if not mapping:
            return None

        parents = self.num_classes_per_level[level - 1]
        children = self.num_classes_per_level[level]
        mat = torch.zeros((parents, children), dtype=dtype, device=device)
        for child, parent in mapping.items():
            if 0 <= child < children and 0 <= parent < parents:
                mat[parent, child] = 1.0
        return mat

    def _build_primary_caps(self, feat: torch.Tensor) -> torch.Tensor:
        bsz = feat.size(0)
        flat = feat.reshape(bsz, -1)
        if flat.size(1) % self.primary_dim != 0:
            raise RuntimeError(
                f"Flattened feature dim ({flat.size(1)}) is not divisible by primary_dim ({self.primary_dim})."
            )
        caps = flat.view(bsz, self.primary_caps_count, self.primary_dim)
        return squash(caps, dim=-1)

    def _reshape_primary_for_level(self, primary_caps: torch.Tensor, target_dim: int) -> torch.Tensor:
        bsz = primary_caps.size(0)
        new_n_caps = self._primary_caps_count_for_target_dim(target_dim)
        return primary_caps.reshape(bsz, new_n_caps, target_dim)

    @staticmethod
    def _predict_votes(x_caps: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Project input capsules into per-class vote vectors.

        Args:
            x_caps: Tensor with shape ``[B, N_in, D_in]``.
            weight: Tensor with shape ``[1, N_in, N_out, D_out, D_in]``.

        Returns:
            Tensor with shape ``[B, N_in, N_out, D_out]``.
        """
        bsz, _, _, = x_caps.shape
        x_expanded = x_caps.unsqueeze(2).unsqueeze(-1)  # [B, N_in, 1, D_in, 1]
        x_tiled = x_expanded.repeat(1, 1, weight.size(2), 1, 1)  # [B, N_in, N_out, D_in, 1]
        w_tiled = weight.expand(bsz, -1, -1, -1, -1)
        votes = torch.matmul(w_tiled, x_tiled).squeeze(-1)  # [B, N_in, N_out, D_out]
        return votes

    def _fuse_attention_output(self, level: int, caps_out: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        fused = caps_out + attn_out
        if self.attn_postprocess == "layernorm":
            return self.post_attn_norms[level](fused)
        return squash(fused, dim=-1)

    def _route_level(
        self,
        level: int,
        x_caps: torch.Tensor,
        prev_predictions: Optional[torch.Tensor],
        taxonomy_matrix: Optional[torch.Tensor],
    ) -> torch.Tensor:
        spec = self.level_specs[level]
        bsz, n_in, _ = x_caps.shape
        n_out = spec.n_out

        votes = self._predict_votes(x_caps, self.W[level])

        if prev_predictions is not None:
            votes = hierarchical_agreement(
                votes=votes,
                prev_predictions=prev_predictions,
                dim_transform=self.dim_transforms[level - 1],
                hierarchical_gate=self.h_gates[level - 1],
            )

        raw_weights = torch.zeros((bsz, n_in, n_out), dtype=x_caps.dtype, device=x_caps.device)
        caps_out = None
        routing_weights = None
        for iter_idx in range(self.routing_iters):
            routing_weights = taxonomy_guided_routing_weights(
                raw_weights=raw_weights,
                level=level,
                taxonomy_matrix=taxonomy_matrix,
                prev_predictions=prev_predictions,
                taxonomy_temperature=self.taxonomy_temperature,
                mask_threshold_high=self.mask_threshold_high,
                mask_threshold_low=self.mask_threshold_low,
                mask_temperature=self.mask_temperature,
                mask_center=self.mask_center,
            )

            weighted_sum = (routing_weights.unsqueeze(-1) * votes).sum(dim=1)
            caps_out = squash(weighted_sum, dim=-1)

            if iter_idx < self.routing_iters - 1:
                agreement = (votes * caps_out.unsqueeze(1)).sum(dim=-1)
                raw_weights = raw_weights + agreement

        assert caps_out is not None and routing_weights is not None
        if prev_predictions is not None:
            attn_out, _ = self.attn_layers[level](caps_out, prev_predictions, prev_predictions)
        else:
            attn_out, _ = self.attn_layers[level](caps_out, caps_out, caps_out)
        return self._fuse_attention_output(level, caps_out, attn_out)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        _ = targets
        feat = self.backbone(self._prepare_backbone_input(x))
        primary_caps = self._build_primary_caps(feat)

        logits_per_level: List[torch.Tensor] = []
        secondary_caps: List[torch.Tensor] = []
        routing_stats: Dict[str, torch.Tensor] = {}

        for level in range(self.depth):
            if level == 0:
                inp = primary_caps
                prev_predictions = None
            else:
                p_caps_lvl = self._reshape_primary_for_level(primary_caps, self.secondary_dims[level - 1])
                inp = torch.cat([p_caps_lvl, secondary_caps[-1]], dim=1)
                prev_predictions = secondary_caps[-1]

            taxonomy_matrix = self._taxonomy_matrix(level, inp.device, inp.dtype)
            caps = self._route_level(
                level=level,
                x_caps=inp,
                prev_predictions=prev_predictions,
                taxonomy_matrix=taxonomy_matrix,
            )
            logits = safe_norm(caps, dim=-1)

            secondary_caps.append(caps)
            logits_per_level.append(logits)
            routing_stats[f"level_{level}_caps_norm_mean"] = logits.mean().detach()

        return {
            "logits_per_level": logits_per_level,
            "orthonormal_plugin_scores_per_level": logits_per_level,
            "secondary_caps": secondary_caps,
            "routing_stats": routing_stats,
        }
