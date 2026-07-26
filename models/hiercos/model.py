import math
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from models.orthonormal_plugin.config import parse_bool
from models.orthonormal_plugin.head import FrozenBlockDiagonalClassifier, build_fixed_classifier
from models.orthonormal_plugin.topology import build_topology, normalize_parent_of
from models.orthonormal_plugin.transforms import build_transformation_module


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
            f"Hier-COS pretrained ResNet-50 unavailable ({exc}). Falling back to random initialization.",
            RuntimeWarning,
        )
        try:
            return resnet50(pretrained=True)
        except Exception:
            try:
                return resnet50(weights=None)
            except TypeError:
                return resnet50(pretrained=False)


class _WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        drop_rate: float = 0.0,
        activate_before_residual: bool = False,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.drop_rate = float(drop_rate)
        self.equal_in_out = bool(in_planes == out_planes)
        self.conv_shortcut = None
        if not self.equal_in_out:
            self.conv_shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        self.activate_before_residual = bool(activate_before_residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.equal_in_out and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))

        out = self.relu2(self.bn2(self.conv1(out if self.equal_in_out else x)))
        if self.drop_rate > 0.0:
            out = torch.nn.functional.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.equal_in_out else self.conv_shortcut(x)
        return shortcut + out


class _WideNetworkBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_planes: int,
        out_planes: int,
        stride: int,
        drop_rate: float,
        activate_before_residual: bool,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        for idx in range(int(num_layers)):
            layers.append(
                _WideBasicBlock(
                    in_planes=in_planes if idx == 0 else out_planes,
                    out_planes=out_planes,
                    stride=stride if idx == 0 else 1,
                    drop_rate=drop_rate,
                    activate_before_residual=activate_before_residual and idx == 0,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class _WideResNetNodeBackbone(nn.Module):
    def __init__(
        self,
        out_dim: int,
        depth: int = 28,
        widen_factor: int = 8,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"Invalid WideResNet depth={depth}; expected (depth - 4) % 6 == 0.")

        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n = int((depth - 4) / 6)

        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = _WideNetworkBlock(
            num_layers=n,
            in_planes=n_channels[0],
            out_planes=n_channels[1],
            stride=1,
            drop_rate=drop_rate,
            activate_before_residual=True,
        )
        self.block2 = _WideNetworkBlock(
            num_layers=n,
            in_planes=n_channels[1],
            out_planes=n_channels[2],
            stride=2,
            drop_rate=drop_rate,
            activate_before_residual=False,
        )
        self.block3 = _WideNetworkBlock(
            num_layers=n,
            in_planes=n_channels[2],
            out_planes=n_channels[3],
            stride=2,
            drop_rate=drop_rate,
            activate_before_residual=False,
        )
        self.bn = nn.BatchNorm2d(n_channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.downsample = nn.Conv2d(
            n_channels[3],
            int(out_dim),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                kernel_size = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / kernel_size))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.downsample(out)
        out = torch.nn.functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return out


class _ResNet50NodeBackbone(nn.Module):
    def __init__(self, out_dim: int, pretrained: bool = True, pool: str = "max"):
        super().__init__()
        trunk = _build_resnet50_backbone(pretrained=bool(pretrained))
        self.features = nn.Sequential(
            *list(trunk.children())[:-2],
            nn.Conv2d(2048, int(out_dim), kernel_size=1, stride=1, padding=0, bias=False),
        )

        if not isinstance(pool, str):
            raise ValueError("Hier-COS `model.pool` must be a string.")
        pool_mode = pool
        if pool_mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=7, stride=7)
        elif pool_mode == "average":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(
                f"Unsupported Hier-COS model.pool '{pool_mode}'. "
                "Expected one of ['max', 'average']."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class HierCosModel(nn.Module):
    """Hier-COS fixed-frame model with optional LH-projected learnable level heads."""

    def __init__(
        self,
        num_classes_per_level: List[int],
        taxonomy: Optional[Dict[str, Any]],
        variant: str = "haframe_resnet50",
        transform_mode: str = "full",
        pretrained: bool = True,
        pool: str = "max",
        backbone_lr_scale: float = 0.1,
        transform_lr_scale: float = 1.0,
        fixed_frame_mode: str = "orthonormal_random",
        fixed_frame_per_level: bool = False,
        projection_cfg: Optional[Dict[str, Any]] = None,
        wide_depth: int = 28,
        wide_widen_factor: int = 8,
        wide_drop_rate: float = 0.0,
    ):
        super().__init__()
        if taxonomy is None:
            raise ValueError("Hier-COS requires taxonomy with parent-child mappings.")

        self.num_classes_per_level = [int(v) for v in num_classes_per_level]
        topology = build_topology(
            self.num_classes_per_level,
            taxonomy=taxonomy,
            owner="Hier-COS",
        )

        self.depth = int(topology["depth"])
        self.total_nodes = int(topology["total_nodes"])
        self.backbone_lr_scale = float(backbone_lr_scale)
        self.transform_lr_scale = float(transform_lr_scale)

        self.level_node_id_names: List[str] = []
        self.level_subspace_mask_names: List[str] = []
        for level in range(self.depth):
            node_ids_name = f"level_node_ids_{level}"
            mask_name = f"level_subspace_mask_{level}"
            self.register_buffer(node_ids_name, topology["level_node_ids"][level], persistent=False)
            self.register_buffer(mask_name, topology["level_subspace_masks"][level], persistent=False)
            self.level_node_id_names.append(node_ids_name)
            self.level_subspace_mask_names.append(mask_name)

        self.register_buffer("leaf_to_level_local", topology["leaf_to_level_local"], persistent=False)
        self.register_buffer("node_prob_weights", topology["node_prob_weights"], persistent=False)

        if not isinstance(variant, str):
            raise ValueError("Hier-COS `model.variant` must be a string.")
        self.variant = variant

        if not isinstance(transform_mode, str):
            raise ValueError("Hier-COS `model.transform_mode` must be a string.")
        self.transform_mode = transform_mode
        if self.transform_mode not in {"full", "bn_linear", "final_only"}:
            raise ValueError(
                f"Unsupported Hier-COS model.transform_mode '{self.transform_mode}'. "
                "Expected one of ['full', 'bn_linear', 'final_only']."
            )

        if self.variant == "haframe_wide_resnet":
            self.backbone = _WideResNetNodeBackbone(
                out_dim=self.total_nodes,
                depth=int(wide_depth),
                widen_factor=int(wide_widen_factor),
                drop_rate=float(wide_drop_rate),
            )
        elif self.variant == "haframe_resnet50":
            self.backbone = _ResNet50NodeBackbone(
                out_dim=self.total_nodes,
                pretrained=bool(pretrained),
                pool=pool,
            )
        else:
            raise ValueError(
                f"Unsupported Hier-COS variant '{variant}'. "
                "Expected one of ['haframe_wide_resnet', 'haframe_resnet50']."
            )

        self.f_theta = build_transformation_module(
            self.total_nodes,
            mode=self.transform_mode,
            owner="Hier-COS model",
        )
        self.fixed_frame_mode = "orthonormal_random" if fixed_frame_mode == "orthonormal_block_random" else fixed_frame_mode
        self.fixed_frame_per_level = (
            parse_bool(fixed_frame_per_level, default=False)
            or fixed_frame_mode == "orthonormal_block_random"
        )
        self.fixed_classifier = build_fixed_classifier(
            width=self.total_nodes,
            mode=fixed_frame_mode,
            fixed_frame_per_level=self.fixed_frame_per_level,
            block_sizes=self.num_classes_per_level,
            owner="Hier-COS model",
        )
        projection_cfg = dict(projection_cfg or {})
        self.projection_enabled = parse_bool(projection_cfg.get("enabled", False), default=False)
        self.advantage_enabled = parse_bool(
            projection_cfg.get("advantage_enabled", False),
            default=False,
        )
        if self.advantage_enabled and not self.projection_enabled:
            raise ValueError(
                "Hier-COS `model.projection.advantage_enabled=true` requires "
                "`model.projection.enabled=true`."
            )
        self.projection_eps = float(projection_cfg.get("eps", 1e-6))
        if self.projection_eps <= 0.0:
            raise ValueError("Hier-COS `model.projection.eps` must be > 0.")
        self.projection_heads = nn.ModuleList(
            [
                nn.Linear(self.total_nodes, int(num_classes))
                for num_classes in self.num_classes_per_level
            ]
            if self.projection_enabled
            else []
        )
        self.parent_index_buffer_names: List[Optional[str]] = [None] * self.depth
        if self.advantage_enabled:
            parent_of = normalize_parent_of(taxonomy, owner="Hier-COS advantage topology")
            for level in range(1, self.depth):
                mapping = parent_of[level]
                parent_index = torch.tensor(
                    [
                        int(mapping[child])
                        for child in range(self.num_classes_per_level[level])
                    ],
                    dtype=torch.long,
                )
                buffer_name = f"parent_index_level_{level}"
                self.register_buffer(buffer_name, parent_index, persistent=False)
                self.parent_index_buffer_names[level] = buffer_name

    def parameter_groups(
        self,
        base_lr: float,
        backbone_lr_scale: Optional[float] = None,
        transform_lr_scale: Optional[float] = None,
    ):
        backbone_scale = self.backbone_lr_scale if backbone_lr_scale is None else float(backbone_lr_scale)
        transform_scale = self.transform_lr_scale if transform_lr_scale is None else float(transform_lr_scale)
        backbone_lr = float(base_lr) * float(backbone_scale)
        transform_lr = float(base_lr) * float(transform_scale)

        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        transform_params = (
            [p for p in self.f_theta.parameters() if p.requires_grad]
            if self.transform_mode != "final_only"
            else []
        )
        classifier_params = [
            p
            for module in (self.fixed_classifier, self.projection_heads)
            for p in module.parameters()
            if p.requires_grad
        ]

        if self.variant == "haframe_wide_resnet":
            # Upstream wraps the WideResNet transformation head inside features_2,
            # so both backbone and transform head use the low LR group.
            other_params = classifier_params
            backbone_params = backbone_params + transform_params
        else:
            other_params = classifier_params

        groups = []
        if other_params:
            groups.append({"params": other_params, "lr": float(base_lr)})
        if transform_params and self.variant != "haframe_wide_resnet":
            groups.append({"params": transform_params, "lr": float(transform_lr)})
        if backbone_params:
            groups.append({"params": backbone_params, "lr": float(backbone_lr)})
        return groups

    def _level_subspace_scores(self, node_logits: torch.Tensor) -> List[torch.Tensor]:
        squared = node_logits.pow(2)
        scores: List[torch.Tensor] = []
        for mask_name in self.level_subspace_mask_names:
            mask = getattr(self, mask_name).to(device=node_logits.device, dtype=node_logits.dtype)
            level_scores_sq = torch.matmul(squared, mask.transpose(0, 1).contiguous())
            scores.append(torch.sqrt(level_scores_sq.clamp_min(0.0)))
        return scores

    def _level_node_ids(self) -> List[torch.Tensor]:
        return [getattr(self, name) for name in self.level_node_id_names]

    def _parent_baseline(self, level: int, previous_logits: torch.Tensor) -> torch.Tensor:
        if level <= 0:
            raise ValueError("Hier-COS advantage is defined only for levels greater than zero.")
        buffer_name = self.parent_index_buffer_names[level]
        if buffer_name is None:
            raise RuntimeError(f"Missing Hier-COS advantage parent indices for level {level}.")
        parent_index = getattr(self, buffer_name).to(device=previous_logits.device)
        return previous_logits.index_select(dim=1, index=parent_index)

    @staticmethod
    def _compute_projection_component(
        z: torch.Tensor,
        prev_weights: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        # A^(l) is the row-wise stack [W_1; ...; W_(l-1)] with no
        # activation-derivative factor.
        az = torch.matmul(z, prev_weights.transpose(0, 1).contiguous())
        gram = torch.matmul(prev_weights, prev_weights.transpose(0, 1).contiguous())
        eye = torch.eye(gram.size(-1), dtype=gram.dtype, device=gram.device)
        gram = gram + float(eps) * eye
        coefficients = torch.linalg.solve(gram, az.transpose(0, 1).contiguous())
        return torch.matmul(
            coefficients.transpose(0, 1).contiguous(),
            prev_weights,
        )

    def _projected_branch_logits_per_level(
        self,
        z: torch.Tensor,
    ) -> List[torch.Tensor]:
        if len(self.projection_heads) != self.depth:
            raise RuntimeError(
                "Enabled Hier-COS LH-DNN projection requires one learnable head per hierarchy level."
            )

        compute_dtype = torch.float32 if z.dtype in {torch.float16, torch.bfloat16} else z.dtype
        autocast_off = nullcontext()
        if z.device.type in {"cpu", "cuda", "xpu", "mps"}:
            autocast_off = torch.autocast(device_type=z.device.type, enabled=False)

        logits_per_level: List[torch.Tensor] = []
        with autocast_off:
            z_work = z.to(dtype=compute_dtype)

            for level, head in enumerate(self.projection_heads):
                if level == 0:
                    z_level = z
                else:
                    previous_weights = torch.cat(
                        [
                            self.projection_heads[previous_level].weight.detach().to(
                                dtype=compute_dtype
                            )
                            for previous_level in range(level)
                        ],
                        dim=0,
                    )
                    c_level = self._compute_projection_component(
                        z=z_work,
                        prev_weights=previous_weights,
                        eps=self.projection_eps,
                    ).to(dtype=z.dtype)
                    z_level = z - c_level + c_level.detach()

                head_input = z_level.to(dtype=head.weight.dtype)
                logits_level = torch.matmul(
                    head_input,
                    head.weight.transpose(0, 1).contiguous(),
                )
                if head.bias is not None:
                    logits_level = logits_level + head.bias
                if self.advantage_enabled and level > 0:
                    logits_level = logits_level + self._parent_baseline(
                        level=level,
                        previous_logits=logits_per_level[level - 1].detach(),
                    )
                logits_per_level.append(logits_level)
        return logits_per_level

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        z = self.backbone(x)
        if self.projection_enabled:
            transformed = self.f_theta(z) if self.transform_mode != "final_only" else z
            branch_logits_per_level = self._projected_branch_logits_per_level(
                z=transformed,
            )
            fixed_input = torch.cat(branch_logits_per_level, dim=1)
            node_logits = self.fixed_classifier(fixed_input)
            node_logits_per_level = list(
                torch.split(node_logits, self.num_classes_per_level, dim=1)
            )
        else:
            transformed = self.f_theta(z) if self.transform_mode != "final_only" else z
            node_logits = self.fixed_classifier(transformed)
            if isinstance(self.fixed_classifier, FrozenBlockDiagonalClassifier):
                node_logits_per_level = list(torch.split(node_logits, self.num_classes_per_level, dim=1))
            else:
                node_logits_per_level = None
        logits_per_level = self._level_subspace_scores(node_logits)
        effective_probs_per_level = [torch.softmax(level_logits, dim=-1) for level_logits in logits_per_level]

        level_node_ids = self._level_node_ids()
        return {
            "logits_per_level": logits_per_level,
            "effective_probs_per_level": effective_probs_per_level,
            "leaf_logits": logits_per_level[-1],
            "node_logits": node_logits,
            "orthonormal_plugin_node_logits": node_logits,
            "node_logits_per_level": node_logits_per_level,
            "orthonormal_plugin_node_logits_per_level": node_logits_per_level,
            "hiercos_level_node_ids": level_node_ids,
            "orthonormal_plugin_level_node_ids": level_node_ids,
            "leaf_to_level_local": self.leaf_to_level_local,
            "orthonormal_plugin_leaf_to_level_local": self.leaf_to_level_local,
            "node_prob_weights": self.node_prob_weights,
            "orthonormal_plugin_node_prob_weights": self.node_prob_weights,
        }
