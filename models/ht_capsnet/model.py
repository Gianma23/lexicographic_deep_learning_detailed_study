from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

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


def _safe_num_heads(embed_dim: int, requested_heads: int) -> int:
    req = max(1, min(int(requested_heads), int(embed_dim)))
    for heads in range(req, 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


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
        taxonomy_temperature: float = 0.5,
        mask_threshold_high: float = 0.9,
        mask_threshold_low: float = 0.1,
        mask_temperature: float = 0.5,
        mask_center: float = 0.5,
        attn_heads: int = 16,
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

        self.backbone = _ConvBackbone(
            in_ch=3,
            num_blocks=num_blocks,
            initial_filters=initial_filters,
            filter_increment=filter_increment,
        )

        # Upstream reshapes raw backbone features into primary capsules.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, int(input_size), int(input_size))
            feat = self.backbone(dummy)
            flat_dim = int(feat[0].numel())
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
            heads = _safe_num_heads(q_dim, attn_heads)
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
            self.norm_layers.append(nn.LayerNorm(q_dim))

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
            attn_out, _ = self.attn_layers[level](caps_out, prev_predictions, prev_predictions)
        else:
            attn_out, _ = self.attn_layers[level](caps_out, caps_out, caps_out)
        caps_out = self.norm_layers[level](caps_out + attn_out)
        return caps_out

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        _ = targets
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
