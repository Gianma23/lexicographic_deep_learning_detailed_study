from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .routing import dynamic_routing


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


class HDCapsNet(nn.Module):
    """PyTorch implementation of HD-CapsNet (coarse->fine routing with skip connections)."""

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
    ):
        super().__init__()
        _ = taxonomy
        self.num_classes_per_level = list(num_classes_per_level)
        self.depth = len(self.num_classes_per_level)
        self.primary_dim = int(primary_dim)
        self.routing_iters = int(routing_iters)

        if self.depth < 2:
            raise ValueError("HD-CapsNet expects at least 2 hierarchy levels.")

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

        self.primary_caps_count = int(max(8, self.num_classes_per_level[0]))
        self.primary_proj = nn.Conv2d(
            self.backbone.out_channels,
            self.primary_caps_count * self.primary_dim,
            kernel_size=1,
        )

        in_dims = [self.primary_dim] + [self.secondary_dims[i - 1] for i in range(1, self.depth)]
        self.primary_skip_projs = nn.ModuleList(
            [nn.Linear(self.primary_dim, in_dims[level]) for level in range(1, self.depth)]
        )

        self.vote_layers = nn.ModuleList(
            [
                nn.Linear(in_dims[level], self.num_classes_per_level[level] * self.secondary_dims[level])
                for level in range(self.depth)
            ]
        )

    def _build_primary_caps(self, feat: torch.Tensor) -> torch.Tensor:
        p = self.primary_proj(feat)  # [B, C, H, W]
        bsz, _, h, w = p.shape
        p = p.view(bsz, self.primary_caps_count, self.primary_dim, h, w)
        p = p.permute(0, 3, 4, 1, 2).contiguous().view(bsz, -1, self.primary_dim)
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
            else:
                p_skip = self.primary_skip_projs[level - 1](primary_caps)
                inp = torch.cat([p_skip, secondary_caps[-1]], dim=1)

            bsz, nin, _ = inp.shape
            nout = self.num_classes_per_level[level]
            dout = self.secondary_dims[level]

            votes = self.vote_layers[level](inp).view(bsz, nin, nout, dout)
            caps, coupling = dynamic_routing(votes=votes, num_iters=self.routing_iters)
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
