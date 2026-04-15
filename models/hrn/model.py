import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    """Post-activation conv block used by upstream HRN."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        relu: bool = True,
        bn: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def _build_resnet50_backbone(pretrained: bool):
    from torchvision.models import resnet50

    if not pretrained:
        try:
            return resnet50(weights=None)
        except TypeError:
            return resnet50(pretrained=False)

    try:
        from torchvision.models import ResNet50_Weights

        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except Exception as exc:
        warnings.warn(
            f"HRN pretrained ResNet-50 unavailable ({exc}). Falling back to random initialization.",
            RuntimeWarning,
        )
        try:
            return resnet50(pretrained=True)
        except Exception:
            try:
                return resnet50(weights=None)
            except TypeError:
                return resnet50(pretrained=False)


class HRNModel(nn.Module):
    """Paper-faithful 3-level HRN with explicit OHier/OCE channels."""

    def __init__(
        self,
        num_classes_per_level: List[int],
        backbone: str = "resnet50",
        pretrained: bool = True,
        branch_hidden_dim: int = 1024,
        embedding_dim: int = 512,
        dropout: float = 0.0,
        trunk_grad_scale: float = 0.1,
    ):
        super().__init__()
        self.num_classes_per_level = [int(n) for n in num_classes_per_level]
        if len(self.num_classes_per_level) != 3:
            raise ValueError(f"HRN expects exactly 3 hierarchy levels, got: {self.num_classes_per_level}")

        backbone_name = str(backbone).strip().lower()
        if backbone_name != "resnet50":
            raise ValueError(f"Unsupported HRN backbone '{backbone}'. Only 'resnet50' is supported.")

        trunk = _build_resnet50_backbone(pretrained=bool(pretrained))
        self.features = nn.Sequential(*list(trunk.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.num_ftrs = int(trunk.fc.in_features)
        self.trunk_grad_scale = float(trunk_grad_scale)

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs, int(branch_hidden_dim), kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(int(branch_hidden_dim), self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs, int(branch_hidden_dim), kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(int(branch_hidden_dim), self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, int(branch_hidden_dim), kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(int(branch_hidden_dim), self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True),
        )

        self.fc1 = self._build_fc_block(int(branch_hidden_dim), int(embedding_dim), float(dropout))
        self.fc2 = self._build_fc_block(int(branch_hidden_dim), int(embedding_dim), float(dropout))
        self.fc3 = self._build_fc_block(int(branch_hidden_dim), int(embedding_dim), float(dropout))

        self.classifier_1 = nn.Linear(int(embedding_dim), self.num_classes_per_level[0])
        self.classifier_2 = nn.Linear(int(embedding_dim), self.num_classes_per_level[1])
        self.classifier_3 = nn.Linear(int(embedding_dim), self.num_classes_per_level[2])
        self.classifier_3_1 = nn.Linear(int(embedding_dim), self.num_classes_per_level[2])

        self._register_trunk_gradient_scaling()

    def _build_fc_block(self, branch_hidden_dim: int, embedding_dim: int, dropout: float) -> nn.Module:
        layers: List[nn.Module] = [
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, branch_hidden_dim),
            nn.BatchNorm1d(branch_hidden_dim),
            nn.ELU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(branch_hidden_dim, embedding_dim))
        return nn.Sequential(*layers)

    def _register_trunk_gradient_scaling(self) -> None:
        scale = float(self.trunk_grad_scale)
        if scale == 1.0:
            return
        if scale <= 0.0:
            raise ValueError(f"HRN trunk_grad_scale must be > 0, got {scale}.")

        for param in self.features.parameters():
            if not param.requires_grad:
                continue
            param.register_hook(lambda grad, s=scale: grad.mul(s))

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        _ = targets
        x = self.features(x)

        x_order = self.conv_block1(x)
        x_family = self.conv_block2(x)
        x_species = self.conv_block3(x)

        x_order_fc = self.pooling(x_order).flatten(1)
        x_order_fc = self.fc1(x_order_fc)

        x_family_fc = self.pooling(x_family).flatten(1)
        x_family_fc = self.fc2(x_family_fc)

        x_species_fc = self.pooling(x_species).flatten(1)
        x_species_fc = self.fc3(x_species_fc)

        emb_order = x_order_fc
        emb_family = x_family_fc + x_order_fc
        emb_species = x_species_fc + x_family_fc + x_order_fc

        order_logits = self.classifier_1(self.relu(emb_order))
        family_logits = self.classifier_2(self.relu(emb_family))
        species_tree_logits = self.classifier_3(self.relu(emb_species))
        species_ce_logits = self.classifier_3_1(self.relu(emb_species))
        order_sig = torch.sigmoid(order_logits)
        family_sig = torch.sigmoid(family_logits)
        species_sig = torch.sigmoid(species_tree_logits)

        logits_per_level: List[torch.Tensor] = [order_logits, family_logits, species_ce_logits]
        effective_probs_per_level = [F.softmax(logits, dim=-1) for logits in logits_per_level]

        return {
            "logits_per_level": logits_per_level,
            "effective_probs_per_level": effective_probs_per_level,
            "tree_scores_per_level": [order_sig, family_sig, species_sig],
            "species_ce_logits": species_ce_logits,
            "embeddings_per_level": [emb_order, emb_family, emb_species],
        }
