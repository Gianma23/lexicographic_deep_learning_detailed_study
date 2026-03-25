import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routing import (
    hierarchical_agreement,
    squash,
    taxonomy_guided_routing_weights,
)


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
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(c_out),
                    nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(c_out),
                    nn.MaxPool2d(2),
                ]
            )
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _HTRMultiHeadAttention(nn.Module):
    """
    TensorFlow-style MHA parity:
    - independent `num_heads` and `key_dim`
    - output projection back to query dimensionality
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int,
        key_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query_dim = int(query_dim)
        self.kv_dim = int(kv_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        if self.num_heads < 1:
            raise ValueError("attn_heads must be >= 1.")
        if self.key_dim < 1:
            raise ValueError("attn_key_dim must be >= 1.")

        self.attn_dim = self.num_heads * self.key_dim
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(self.query_dim, self.attn_dim, bias=True)
        self.k_proj = nn.Linear(self.kv_dim, self.attn_dim, bias=True)
        self.v_proj = nn.Linear(self.kv_dim, self.attn_dim, bias=True)
        self.out_proj = nn.Linear(self.attn_dim, self.query_dim, bias=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = query.shape
        k_len = key.size(1)

        q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, k_len, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, k_len, self.num_heads, self.key_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, self.attn_dim)
        return self.out_proj(attn)


def _normalize_backbone_name(name: str) -> str:
    value = str(name or "custom").strip().lower().replace("-", "_")
    if value in {"custom"}:
        return "custom"
    if value in {"efficientnetb7", "efficientnet_b7"}:
        return "efficientnet_b7"
    raise ValueError(
        f"Unsupported HT-CapsNet backbone '{name}'. "
        "Supported backbones: custom, efficientnet_b7."
    )


def _normalize_backbone_weights(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text in {"imagenet", "imagenet1k"}:
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
        from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

        weights = None
        if normalized_weights == "imagenet":
            weights = EfficientNet_B7_Weights.IMAGENET1K_V1

        try:
            model = efficientnet_b7(weights=weights)
        except Exception as exc:
            if weights is None:
                raise
            warnings.warn(
                "Failed to load EfficientNet-B7 ImageNet weights. "
                f"Falling back to random initialization. Original error: {exc}",
                RuntimeWarning,
            )
            model = efficientnet_b7(weights=None)
        return model.features

    raise ValueError(f"Unsupported HT-CapsNet backbone '{backbone_name}'.")


class HTCapsNet(nn.Module):
    """Closer structural alignment to the upstream TensorFlow HTR-CapsNet."""

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
        attn_key_dim: int = 32,
        attn_dropout: float = 0.0,
        input_size: int = 224,
    ):
        super().__init__()
        self.num_classes_per_level = [int(v) for v in num_classes_per_level]
        self.depth = len(self.num_classes_per_level)
        self.routing_iters = int(routing_iters)
        self.primary_dim = int(primary_dim)

        if self.depth < 2:
            raise ValueError("HT-CapsNet expects at least 2 hierarchy levels.")

        if secondary_dims is None:
            secondary_dims = [self.primary_dim for _ in self.num_classes_per_level]
        if len(secondary_dims) != self.depth:
            raise ValueError("secondary_dims must match hierarchy depth.")
        self.secondary_dims = [int(v) for v in secondary_dims]

        self.backbone = _build_backbone(
            backbone_name=backbone_name,
            backbone_weights=backbone_weights,
            num_blocks=num_blocks,
            initial_filters=initial_filters,
            filter_increment=filter_increment,
        )

        # Upstream reshapes raw backbone features into primary capsules.
        # Run this shape probe in eval mode so BatchNorm layers don't expect
        # training statistics from a single-item dummy batch.
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                probe_size = max(32, int(input_size))
                dummy = torch.zeros(1, 3, probe_size, probe_size)
                feat = self.backbone(self._prepare_backbone_input(dummy))
                flat_dim = int(feat[0].numel())
        finally:
            self.backbone.train(was_training)
        if flat_dim % self.primary_dim != 0:
            raise ValueError(
                f"Backbone flattened feature dim ({flat_dim}) must be divisible by primary_dim ({self.primary_dim})."
            )
        self.primary_caps_count = int(flat_dim // self.primary_dim)

        self.W = nn.ParameterList()
        self.h_gates = nn.ParameterList()
        self.dim_transforms = nn.ParameterList()

        for level in range(self.depth):
            n_out = self.num_classes_per_level[level]
            d_out = self.secondary_dims[level]
            if level == 0:
                n_in = self.primary_caps_count
                d_in = self.primary_dim
            else:
                d_prev = self.secondary_dims[level - 1]
                new_n_caps = (self.primary_caps_count * self.primary_dim) // d_prev
                n_in = new_n_caps + self.num_classes_per_level[level - 1]
                d_in = d_prev

                gate = nn.Parameter(torch.full((1, n_out, self.num_classes_per_level[level - 1]), 0.5))
                dim_t = nn.Parameter(torch.randn(d_in, d_out) * 0.1)
                self.h_gates.append(gate)
                self.dim_transforms.append(dim_t)

            w_level = nn.Parameter(torch.randn(1, n_in, n_out, d_out, d_in) * 0.1)
            self.W.append(w_level)

        self.attn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for level in range(self.depth):
            q_dim = self.secondary_dims[level]
            kv_dim = self.secondary_dims[level - 1] if level > 0 else q_dim
            self.attn_layers.append(
                _HTRMultiHeadAttention(
                    query_dim=q_dim,
                    kv_dim=kv_dim,
                    num_heads=int(attn_heads),
                    key_dim=int(attn_key_dim),
                    dropout=attn_dropout,
                )
            )
            self.norm_layers.append(nn.LayerNorm(q_dim))

        self.taxonomy_temperature = float(taxonomy_temperature)
        self.mask_threshold_high = float(mask_threshold_high)
        self.mask_threshold_low = float(mask_threshold_low)
        self.mask_temperature = float(mask_temperature)
        self.mask_center = float(mask_center)
        self.parent_of = self._normalize_parent_of(taxonomy)

    @staticmethod
    def _prepare_backbone_input(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [B, C, H, W], got shape={tuple(x.shape)}.")
        channels = int(x.size(1))
        if channels == 1:
            x = x.repeat(1, 3, 1, 1)
        elif channels != 3:
            raise ValueError(f"HT-CapsNet expects 1 or 3 input channels, got {channels}.")

        h, w = int(x.size(-2)), int(x.size(-1))
        if h < 32 or w < 32:
            x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        return x

    @staticmethod
    def _normalize_parent_of(taxonomy: Optional[Dict[str, Any]]) -> Dict[int, Dict[int, int]]:
        if not taxonomy or "parent_of" not in taxonomy:
            return {}
        out: Dict[int, Dict[int, int]] = {}
        for k, v in taxonomy["parent_of"].items():
            out[int(k)] = {int(ck): int(pk) for ck, pk in v.items()}
        return out

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
            if child < children and parent < parents:
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
        total = self.primary_caps_count * self.primary_dim
        if total % target_dim != 0:
            raise RuntimeError(
                f"Primary capsule flattened dim ({total}) is not divisible by target level dim ({target_dim})."
            )
        new_n_caps = total // target_dim
        return primary_caps.reshape(bsz, new_n_caps, target_dim)

    @staticmethod
    def _predict_votes(x_caps: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        x_caps: [B, N_in, D_in]
        weight: [1, N_in, N_out, D_out, D_in]
        returns: [B, N_in, N_out, D_out]
        """
        bsz, _, _, = x_caps.shape
        x_expanded = x_caps.unsqueeze(2).unsqueeze(-1)  # [B, N_in, 1, D_in, 1]
        x_tiled = x_expanded.repeat(1, 1, weight.size(2), 1, 1)  # [B, N_in, N_out, D_in, 1]
        w_tiled = weight.expand(bsz, -1, -1, -1, -1)
        votes = torch.matmul(w_tiled, x_tiled).squeeze(-1)  # [B, N_in, N_out, D_out]
        return votes

    def _route_level(
        self,
        level: int,
        x_caps: torch.Tensor,
        prev_predictions: Optional[torch.Tensor],
        taxonomy_matrix: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, n_in, _ = x_caps.shape
        n_out = self.num_classes_per_level[level]

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
            attn_out = self.attn_layers[level](caps_out, prev_predictions, prev_predictions)
        else:
            attn_out = self.attn_layers[level](caps_out, caps_out, caps_out)
        caps_out = self.norm_layers[level](caps_out + attn_out)
        return caps_out

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        _ = targets
        x = self._prepare_backbone_input(x)
        feat = self.backbone(x)
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
            logits = torch.norm(caps, dim=-1)

            secondary_caps.append(caps)
            logits_per_level.append(logits)
            routing_stats[f"level_{level}_caps_norm_mean"] = logits.mean().detach()

        return {
            "logits_per_level": logits_per_level,
            "secondary_caps": secondary_caps,
            "routing_stats": routing_stats,
        }
