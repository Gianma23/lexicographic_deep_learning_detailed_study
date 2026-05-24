import math
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn


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


def _normalize_parent_of(taxonomy: Dict[str, Any]) -> Dict[int, Dict[int, int]]:
    if not taxonomy or "parent_of" not in taxonomy:
        raise ValueError("Hier-COS requires taxonomy with `parent_of` mappings.")

    parent_of_raw = taxonomy["parent_of"]
    if not isinstance(parent_of_raw, Mapping):
        raise ValueError("`taxonomy['parent_of']` must be a mapping of level -> (child -> parent).")

    normalized: Dict[int, Dict[int, int]] = {}
    for level_key, mapping in parent_of_raw.items():
        if not isinstance(mapping, Mapping):
            raise ValueError(f"`taxonomy['parent_of'][{level_key}]` must be a child -> parent mapping.")
        level = int(level_key)
        normalized[level] = {int(child): int(parent) for child, parent in mapping.items()}
    return normalized


def _level_offsets(num_classes_per_level: Sequence[int]) -> List[int]:
    offsets = [0]
    for classes in num_classes_per_level[:-1]:
        offsets.append(offsets[-1] + int(classes))
    return offsets


def _build_topology(
    num_classes_per_level: Sequence[int],
    taxonomy: Dict[str, Any],
) -> Dict[str, Any]:
    num_classes = [int(v) for v in num_classes_per_level]
    if len(num_classes) < 2:
        raise ValueError("Hier-COS requires hierarchy depth >= 2.")
    if any(v <= 0 for v in num_classes):
        raise ValueError(f"All class counts must be > 0, got {num_classes}.")

    parent_of = _normalize_parent_of(taxonomy)
    depth = len(num_classes)
    offsets = _level_offsets(num_classes)
    total_nodes = int(sum(num_classes))

    parent_global = [-1 for _ in range(total_nodes)]
    children_global: List[List[int]] = [[] for _ in range(total_nodes)]

    for level in range(1, depth):
        mapping = parent_of.get(level)
        if mapping is None:
            raise ValueError(
                f"Missing taxonomy mapping for level transition {level - 1}->{level}. "
                f"Expected `taxonomy['parent_of'][{level}]`."
            )
        num_children = int(num_classes[level])
        num_parents = int(num_classes[level - 1])
        expected_children = set(range(num_children))
        if set(mapping.keys()) != expected_children:
            missing = sorted(expected_children - set(mapping.keys()))
            extra = sorted(set(mapping.keys()) - expected_children)
            raise ValueError(
                f"Invalid taxonomy mapping at level={level}. "
                f"Missing children: {missing[:10]}, extra children: {extra[:10]}."
            )

        for child_local in range(num_children):
            parent_local = int(mapping[child_local])
            if parent_local < 0 or parent_local >= num_parents:
                raise ValueError(
                    f"Invalid parent id={parent_local} for child={child_local} at level={level}; "
                    f"expected [0, {num_parents})."
                )
            child_global = int(offsets[level] + child_local)
            parent_global_idx = int(offsets[level - 1] + parent_local)
            parent_global[child_global] = parent_global_idx
            children_global[parent_global_idx].append(child_global)

    descendants: List[set] = [set() for _ in range(total_nodes)]
    for node in range(total_nodes - 1, -1, -1):
        for child in children_global[node]:
            descendants[node].add(int(child))
            descendants[node].update(descendants[child])

    ancestors: List[List[int]] = [[] for _ in range(total_nodes)]
    for node in range(total_nodes):
        p = parent_global[node]
        anc: List[int] = []
        while p >= 0:
            anc.append(int(p))
            p = parent_global[p]
        ancestors[node] = anc

    level_node_ids: List[torch.Tensor] = []
    level_subspace_masks: List[torch.Tensor] = []
    for level, classes in enumerate(num_classes):
        node_ids = torch.arange(offsets[level], offsets[level] + classes, dtype=torch.long)
        level_node_ids.append(node_ids)

        mask = torch.zeros((classes, total_nodes), dtype=torch.bool)
        for local_id in range(classes):
            global_id = int(offsets[level] + local_id)
            indices = set(ancestors[global_id])
            indices.add(global_id)
            indices.update(descendants[global_id])
            mask[local_id, list(sorted(indices))] = True
        level_subspace_masks.append(mask)

    num_leaf = int(num_classes[-1])
    leaf_to_level_local = torch.full((num_leaf, depth), -1, dtype=torch.long)
    for leaf_local in range(num_leaf):
        g = int(offsets[-1] + leaf_local)
        cur = g
        for level in range(depth - 1, -1, -1):
            leaf_to_level_local[leaf_local, level] = int(cur - offsets[level])
            if level > 0:
                cur_parent = parent_global[cur]
                if cur_parent < 0:
                    raise ValueError(
                        f"Leaf node id={leaf_local} has no valid ancestor at level={level - 1}."
                    )
                cur = cur_parent

    level_weights = torch.arange(depth, 0, -1, dtype=torch.float32)
    level_weights = torch.exp(1.0 / level_weights)
    level_weights = level_weights / torch.norm(level_weights, p=2).clamp_min(1e-12)
    level_weights = level_weights.pow(2)

    return {
        "depth": depth,
        "total_nodes": total_nodes,
        "level_node_ids": level_node_ids,
        "level_subspace_masks": level_subspace_masks,
        "leaf_to_level_local": leaf_to_level_local,
        "node_prob_weights": level_weights,
    }


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


def _get_activation(name: str, channels: int) -> nn.Module:
    if not isinstance(name, str):
        raise ValueError("Hier-COS activation name must be a string.")
    mode = name
    if mode == "relu":
        return nn.ReLU()
    if mode == "elu":
        return nn.ELU()
    if mode == "tanh":
        return nn.Tanh()
    if mode == "prelu":
        return nn.PReLU(num_parameters=int(channels))
    raise ValueError(f"Unsupported activation '{name}'.")


class _PointResidualTransformationLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        activation: str = "prelu",
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.act1 = _get_activation(activation, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = _get_activation(activation, out_channels)
        if int(in_channels) == int(out_channels):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x + residual


class _NarrowResidualTransformationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "prelu"):
        super().__init__()
        self.layer1 = _PointResidualTransformationLayer(
            in_channels=in_channels,
            hidden_channels=out_channels,
            out_channels=out_channels,
            activation=activation,
        )
        self.layer2 = _PointResidualTransformationLayer(
            in_channels=out_channels,
            hidden_channels=out_channels,
            out_channels=out_channels,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return x


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
    """Hier-COS model with fixed orthonormal node frame and taxonomy-driven subspaces."""

    def __init__(
        self,
        num_classes_per_level: List[int],
        taxonomy: Optional[Dict[str, Any]],
        variant: str = "haframe_resnet50",
        pretrained: bool = True,
        pool: str = "max",
        backbone_lr_scale: float = 0.1,
        fixed_frame_mode: str = "orthonormal_random",
        wide_depth: int = 28,
        wide_widen_factor: int = 8,
        wide_drop_rate: float = 0.0,
    ):
        super().__init__()
        if taxonomy is None:
            raise ValueError("Hier-COS requires taxonomy with parent-child mappings.")

        self.num_classes_per_level = [int(v) for v in num_classes_per_level]
        topology = _build_topology(self.num_classes_per_level, taxonomy=taxonomy)

        self.depth = int(topology["depth"])
        self.total_nodes = int(topology["total_nodes"])
        self.backbone_lr_scale = float(backbone_lr_scale)

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

        self.f_theta = self._build_transformation_module(self.total_nodes)
        self.fixed_classifier = nn.Linear(self.total_nodes, self.total_nodes, bias=False)
        self._init_fixed_frame(mode=fixed_frame_mode)

    @staticmethod
    def _build_transformation_module(width: int) -> nn.Module:
        return nn.Sequential(
            nn.BatchNorm1d(width),
            _NarrowResidualTransformationHead(
                in_channels=width,
                out_channels=width,
                activation="prelu",
            ),
        )

    def _init_fixed_frame(self, mode: str = "identity") -> None:
        if not isinstance(mode, str):
            raise ValueError("Hier-COS `model.fixed_frame_mode` must be a string.")
        frame_mode = mode
        with torch.no_grad():
            if frame_mode == "orthonormal_random":
                random_matrix = torch.randn(self.total_nodes, self.total_nodes)
                q, _ = torch.linalg.qr(random_matrix, mode="reduced")
                self.fixed_classifier.weight.copy_(q)
            elif frame_mode == "identity":
                self.fixed_classifier.weight.copy_(torch.eye(self.total_nodes))
            else:
                raise ValueError(
                    f"Unsupported Hier-COS model.fixed_frame_mode '{frame_mode}'. "
                    "Expected one of ['orthonormal_random', 'identity']."
                )
        self.fixed_classifier.weight.requires_grad_(False)

    def parameter_groups(self, base_lr: float, backbone_lr_scale: Optional[float] = None):
        scale = self.backbone_lr_scale if backbone_lr_scale is None else float(backbone_lr_scale)
        backbone_lr = float(base_lr) * float(scale)

        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        transform_params = [p for p in self.f_theta.parameters() if p.requires_grad]
        classifier_params = [p for p in self.fixed_classifier.parameters() if p.requires_grad]

        if self.variant == "haframe_wide_resnet":
            # Upstream wraps the WideResNet transformation head inside features_2,
            # so both backbone and transform head use the low LR group.
            other_params = classifier_params
            backbone_params = backbone_params + transform_params
        else:
            other_params = classifier_params + transform_params

        groups = []
        if other_params:
            groups.append({"params": other_params, "lr": float(base_lr)})
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

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        z = self.backbone(x)
        transformed = self.f_theta(z)
        node_logits = self.fixed_classifier(transformed)
        logits_per_level = self._level_subspace_scores(node_logits)
        effective_probs_per_level = [torch.softmax(level_logits, dim=-1) for level_logits in logits_per_level]

        return {
            "logits_per_level": logits_per_level,
            "effective_probs_per_level": effective_probs_per_level,
            "leaf_logits": logits_per_level[-1],
            "node_logits": node_logits,
            "hiercos_level_node_ids": self._level_node_ids(),
            "leaf_to_level_local": self.leaf_to_level_local,
            "node_prob_weights": self.node_prob_weights,
        }
