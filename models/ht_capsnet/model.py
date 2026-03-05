from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .routing import taxonomy_guided_routing


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


class HTCapsNet(nn.Module):
    """PyTorch translation of HT-CapsNet taxonomy-guided routing structure."""

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
        taxonomy_temperature: float = 5.0,
        mask_threshold_high: float = 0.9,
        mask_threshold_low: float = 0.1,
        mask_temperature: float = 0.5,
        mask_center: float = 0.5,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes_per_level = list(num_classes_per_level)
        self.depth = len(self.num_classes_per_level)
        self.routing_iters = routing_iters

        if self.depth < 2:
            raise ValueError("HT-CapsNet expects at least 2 hierarchy levels.")

        if secondary_dims is None:
            secondary_dims = [primary_dim for _ in self.num_classes_per_level]
        if len(secondary_dims) != self.depth:
            raise ValueError("secondary_dims must match hierarchy depth.")

        self.secondary_dims = secondary_dims

        self.backbone = _ConvBackbone(
            in_ch=3,
            num_blocks=num_blocks,
            initial_filters=initial_filters,
            filter_increment=filter_increment,
        )

        self.primary_caps_count = int(max(8, self.num_classes_per_level[0]))
        self.primary_proj = nn.Conv2d(
            self.backbone.out_channels,
            self.primary_caps_count * primary_dim,
            kernel_size=1,
        )

        in_dims = [primary_dim] + [secondary_dims[i - 1] for i in range(1, self.depth)]
        self.primary_skip_projs = nn.ModuleList(
            [nn.Linear(primary_dim, in_dims[level]) for level in range(1, self.depth)]
        )

        self.vote_layers = nn.ModuleList(
            [nn.Linear(in_dims[level], self.num_classes_per_level[level] * secondary_dims[level]) for level in range(self.depth)]
        )

        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=secondary_dims[level],
                    num_heads=max(1, min(attn_heads, secondary_dims[level])),
                    dropout=attn_dropout,
                    batch_first=True,
                )
                for level in range(self.depth)
            ]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(secondary_dims[level]) for level in range(self.depth)])

        self.taxonomy_temperature = taxonomy_temperature
        self.mask_threshold_high = mask_threshold_high
        self.mask_threshold_low = mask_threshold_low
        self.mask_temperature = mask_temperature
        self.mask_center = mask_center

        self.parent_of = self._normalize_parent_of(taxonomy)

    @staticmethod
    def _normalize_parent_of(taxonomy: Optional[Dict[str, Any]]) -> Dict[int, Dict[int, int]]:
        if not taxonomy or "parent_of" not in taxonomy:
            return {}
        out: Dict[int, Dict[int, int]] = {}
        for k, v in taxonomy["parent_of"].items():
            out[int(k)] = {int(ck): int(pk) for ck, pk in v.items()}
        return out

    def _taxonomy_matrix(self, level: int, device: torch.device) -> Optional[torch.Tensor]:
        if level <= 0:
            return None
        mapping = self.parent_of.get(level)
        if not mapping:
            return None

        parents = self.num_classes_per_level[level - 1]
        children = self.num_classes_per_level[level]
        mat = torch.zeros((parents, children), dtype=torch.float32, device=device)
        for child, parent in mapping.items():
            if child < children and parent < parents:
                mat[parent, child] = 1.0
        return mat

    def _build_primary_caps(self, feat: torch.Tensor) -> torch.Tensor:
        p = self.primary_proj(feat)  # [B, C, H, W]
        bsz, _, h, w = p.shape
        p = p.view(bsz, self.primary_caps_count, self.secondary_dims[0], h, w)
        p = p.permute(0, 3, 4, 1, 2).contiguous().view(bsz, -1, self.secondary_dims[0])
        return p

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
                prev_parent_probs = None
            else:
                p_skip = self.primary_skip_projs[level - 1](primary_caps)
                inp = torch.cat([p_skip, secondary_caps[-1]], dim=1)
                prev_parent_probs = torch.softmax(logits_per_level[level - 1], dim=-1)

            bsz, nin, din = inp.shape
            nout = self.num_classes_per_level[level]
            dout = self.secondary_dims[level]

            votes = self.vote_layers[level](inp).view(bsz, nin, nout, dout)

            taxonomy_matrix = self._taxonomy_matrix(level, inp.device)
            caps, coupling = taxonomy_guided_routing(
                votes=votes,
                num_iters=self.routing_iters,
                taxonomy_matrix=taxonomy_matrix,
                prev_parent_probs=prev_parent_probs,
                taxonomy_temperature=self.taxonomy_temperature,
                mask_threshold_high=self.mask_threshold_high,
                mask_threshold_low=self.mask_threshold_low,
                mask_temperature=self.mask_temperature,
                mask_center=self.mask_center,
            )

            attn_out, _ = self.attn_layers[level](caps, caps, caps)
            caps = self.norm_layers[level](caps + attn_out)

            logits = torch.norm(caps, dim=-1)

            secondary_caps.append(caps)
            logits_per_level.append(logits)
            routing_stats[f"level_{level}_coupling_mean"] = coupling.mean().detach()
            routing_stats[f"level_{level}_caps_norm_mean"] = logits.mean().detach()

        return {
            "logits_per_level": logits_per_level,
            "secondary_caps": secondary_caps,
            "routing_stats": routing_stats,
        }
