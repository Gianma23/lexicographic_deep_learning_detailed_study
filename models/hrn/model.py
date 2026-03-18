import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class BasicConv(nn.Module):
    """Conv-BN-ReLU block used by HRN branch feature refiners."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        relu: bool = True,
        bn: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

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
            model = resnet50(weights=None)
        except TypeError:
            model = resnet50(pretrained=False)
        return model

    # First try modern torchvision API.
    try:
        from torchvision.models import ResNet50_Weights

        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except Exception as exc:
        warnings.warn(
            f"HRN pretrained ResNet-50 unavailable ({exc}). Falling back to random initialization.",
            RuntimeWarning,
        )

    # Fallback for older torchvision versions.
    try:
        return resnet50(pretrained=True)
    except Exception:
        try:
            return resnet50(weights=None)
        except TypeError:
            return resnet50(pretrained=False)


class HRNModel(nn.Module):
    """3-level Hierarchical Residual Network adapter for unified training API."""

    def __init__(
        self,
        num_classes_per_level: List[int],
        backbone: str = "resnet50",
        pretrained: bool = True,
        branch_hidden_dim: int = 1024,
        embedding_dim: int = 512,
        dropout: float = 0.0,
        taxonomy: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        _ = taxonomy
        self.num_classes_per_level = list(num_classes_per_level)
        self.depth = len(self.num_classes_per_level)
        if self.depth != 3:
            raise ValueError(
                f"HRN expects exactly 3 hierarchy levels, got {self.depth}: {self.num_classes_per_level}"
            )

        backbone_name = str(backbone).lower().strip()
        if backbone_name != "resnet50":
            raise ValueError(f"Unsupported HRN backbone '{backbone}'. Only 'resnet50' is supported.")

        trunk = _build_resnet50_backbone(pretrained=bool(pretrained))
        trunk_out = int(trunk.fc.in_features)
        self.features = nn.Sequential(*list(trunk.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.branch_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    BasicConv(trunk_out, branch_hidden_dim, kernel_size=1, stride=1, padding=0, relu=True),
                    BasicConv(branch_hidden_dim, trunk_out, kernel_size=3, stride=1, padding=1, relu=True),
                )
                for _ in range(3)
            ]
        )

        proj_layers: List[nn.Module] = []
        for _ in range(3):
            layers: List[nn.Module] = [
                nn.BatchNorm1d(trunk_out),
                nn.Linear(trunk_out, branch_hidden_dim),
                nn.BatchNorm1d(branch_hidden_dim),
                nn.ELU(inplace=True),
            ]
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            layers.append(nn.Linear(branch_hidden_dim, embedding_dim))
            proj_layers.append(nn.Sequential(*layers))
        self.proj_heads = nn.ModuleList(proj_layers)

        self.classifier_level0 = nn.Linear(embedding_dim, self.num_classes_per_level[0])
        self.classifier_level1 = nn.Linear(embedding_dim, self.num_classes_per_level[1])
        self.classifier_level2 = nn.Linear(embedding_dim, self.num_classes_per_level[2])

    def _branch_embedding(self, feat_map: torch.Tensor, level: int) -> torch.Tensor:
        branch = self.branch_blocks[level](feat_map)
        pooled = self.pool(branch).flatten(1)
        return self.proj_heads[level](pooled)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        _ = targets
        feat = self.features(x)

        emb0 = self._branch_embedding(feat, level=0)
        emb1 = self._branch_embedding(feat, level=1) + emb0
        emb2 = self._branch_embedding(feat, level=2) + emb1

        logits0 = self.classifier_level0(self.relu(emb0))
        logits1 = self.classifier_level1(self.relu(emb1))
        logits2 = self.classifier_level2(self.relu(emb2))

        tree_scores_per_level = [torch.sigmoid(logits0), torch.sigmoid(logits1), torch.sigmoid(logits2)]
        logits_per_level = [logits0, logits1, logits2]
        return {
            "logits_per_level": logits_per_level,
            "tree_scores_per_level": tree_scores_per_level,
            "embeddings_per_level": [emb0, emb1, emb2],
        }
