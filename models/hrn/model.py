from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.hcc import HccController
from ..common.cifar_wide_resnet import CifarWideResNetFeatures
from ..common.config import parse_bool
from .losses import _normalize_parent_of


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
        raise RuntimeError(
            "HRN paper reproduction requires ImageNet-pretrained ResNet-50 weights, "
            "but torchvision could not load them. Cache/download the weights or set "
            "model.pretrained=false for a deliberately non-paper run."
        ) from exc


class HRNModel(nn.Module):
    """Three-level HRN with explicit OHier/OCE channels."""

    def __init__(
        self,
        num_classes_per_level: List[int],
        backbone: str = "resnet50",
        pretrained: bool = True,
        wide_depth: int = 28,
        wide_widen_factor: int = 8,
        wide_drop_rate: float = 0.0,
        branch_hidden_dim: int = 1024,
        embedding_dim: int = 512,
        dropout: float = 0.0,
        trunk_lr_scale: float = 0.1,
        projection_cfg: Optional[Dict[str, Any]] = None,
        taxonomy: Optional[Dict[str, Any]] = None,
        hcc_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.num_classes_per_level = [int(n) for n in num_classes_per_level]
        if len(self.num_classes_per_level) != 3:
            raise ValueError(f"HRN expects exactly 3 hierarchy levels, got: {self.num_classes_per_level}")

        projection_cfg = dict(projection_cfg or {})
        self.projection_enabled = parse_bool(projection_cfg.get("enabled", False), default=False)
        self.projection_eps = float(projection_cfg.get("eps", 1e-6))
        if self.projection_eps <= 0.0:
            raise ValueError("HRN `model.projection.eps` must be > 0.")
        if self.projection_enabled and float(dropout) > 0.0:
            raise ValueError(
                "HRN `model.projection.enabled=true` replaces the native branch "
                "stacks with direct linear heads, so `model.dropout` must be 0."
            )

        backbone_name = backbone
        if not isinstance(backbone_name, str):
            raise ValueError("HRN model.backbone must be a string.")
        if backbone_name == "resnet50":
            trunk = _build_resnet50_backbone(pretrained=bool(pretrained))
            self.features = nn.Sequential(*list(trunk.children())[:-2])
            self.num_ftrs = int(trunk.fc.in_features)
        elif backbone_name == "wide_resnet":
            if pretrained:
                raise ValueError(
                    "HRN backbone='wide_resnet' has no pretrained weights; set model.pretrained=false."
                )
            self.features = CifarWideResNetFeatures(
                depth=int(wide_depth),
                widen_factor=int(wide_widen_factor),
                drop_rate=float(wide_drop_rate),
            )
            self.num_ftrs = int(self.features.out_channels)
        else:
            raise ValueError(
                f"Unsupported HRN backbone '{backbone}'. Expected one of ['resnet50', 'wide_resnet']."
            )

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.trunk_lr_scale = float(trunk_lr_scale)

        if self.projection_enabled:
            # LH projection needs one explicit affine head per level at a shared
            # vector branch point. Do not retain a factorised chain of linear
            # stand-ins for HRN's conv/fc stacks: without intervening activations
            # that chain is exactly one affine map but adds redundant parameters
            # and a different deep-linear optimisation problem. The projected
            # adaptation therefore uses direct D -> C_l heads.
            protected_rows = int(sum(self.num_classes_per_level[:-1]))
            if self.num_ftrs <= protected_rows:
                raise ValueError(
                    "HRN projection non-triviality condition violated: the pooled trunk "
                    f"width ({self.num_ftrs}) must be greater than "
                    f"sum(num_classes_per_level[:-1]) ({protected_rows})."
                )
            # The shared linear map at the branching point. Same name and role as
            # LH-DNN's `shared_linear`/`shared_relu` pair (Hier-COS's `f_theta`
            # plus its terminal activation play the same part). Without it every
            # shared parameter is conv trunk, and orthogonality of the fine
            # gradient at the branching point does not survive the trunk's Gram
            # factor -- i.e. the projection would protect nothing. This is the
            # layer the guarantee actually bites on, and its ReLU supplies the
            # idempotent, sample-dependent rho'.
            self.shared_linear = nn.Linear(self.num_ftrs, self.num_ftrs)
            self.shared_relu = nn.ReLU()
            self.classifier_1 = nn.Linear(self.num_ftrs, self.num_classes_per_level[0])
            self.classifier_2 = nn.Linear(self.num_ftrs, self.num_classes_per_level[1])
            self.classifier_3 = nn.Linear(self.num_ftrs, self.num_classes_per_level[2])
            self.classifier_3_1 = nn.Linear(self.num_ftrs, self.num_classes_per_level[2])
        else:
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

        self.parent_index_buffer_names: List[Optional[str]] = [None, None, None]
        if self.projection_enabled:
            # Score-space advantage, as in LH-DNN: every level adds its detached
            # parent's advantage score. This is HRN's coarse-to-fine residual
            # moved from the embedding to the score, which is what makes the
            # native `A = W_prev * rho'` sufficient.
            if taxonomy is None:
                raise ValueError(
                    "HRN `model.projection.enabled=true` uses the LH-DNN advantage, "
                    "which requires taxonomy parent mappings."
                )
            parent_of = _normalize_parent_of(taxonomy)
            for level in range(1, 3):
                mapping = parent_of.get(level)
                if mapping is None:
                    raise ValueError(
                        f"Missing taxonomy mapping for level transition {level - 1}->{level}."
                    )
                parent_index = torch.tensor(
                    [int(mapping[child]) for child in range(self.num_classes_per_level[level])],
                    dtype=torch.long,
                )
                buffer_name = f"parent_index_level_{level}"
                self.register_buffer(buffer_name, parent_index, persistent=False)
                self.parent_index_buffer_names[level] = buffer_name

        if bool((hcc_cfg or {}).get("enabled", False)) and self.projection_enabled:
            raise ValueError(
                "hcc.enabled=true is not supported together with "
                "model.projection.enabled=true (HRN's LH-DNN-style branch-point "
                "projection); disable one of the two."
            )
        self.hcc = HccController(
            num_classes_per_level=self.num_classes_per_level,
            taxonomy=taxonomy,
            hcc_cfg=hcc_cfg,
        )

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

    def parameter_groups(
        self,
        base_lr: float,
        backbone_lr_scale: Optional[float] = None,
        trunk_lr_scale: Optional[float] = None,
    ):
        scale = self.trunk_lr_scale
        if trunk_lr_scale is not None:
            scale = float(trunk_lr_scale)
        elif backbone_lr_scale is not None:
            scale = float(backbone_lr_scale)
        if scale <= 0.0:
            raise ValueError(f"HRN trunk_lr_scale must be > 0, got {scale}.")

        if self.projection_enabled:
            head_modules = [
                self.classifier_1,
                self.classifier_2,
                self.classifier_3,
                self.classifier_3_1,
                self.shared_linear,
            ]
        else:
            head_modules = [
                self.classifier_1,
                self.classifier_2,
                self.classifier_3,
                self.classifier_3_1,
                self.fc1,
                self.fc2,
                self.fc3,
                self.conv_block1,
                self.conv_block2,
                self.conv_block3,
            ]
        groups = [
            {"params": module.parameters(), "lr": float(base_lr)}
            for module in head_modules
        ]
        groups.append({"params": self.features.parameters(), "lr": float(base_lr) * float(scale)})
        return groups

    @staticmethod
    def _compute_projection_component(
        z: torch.Tensor,
        a: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        # LH-DNN form: c[b] = A[b]^T (A[b] A[b]^T + eps I)^-1 A[b] z[b], with
        # A[b] = W_prev * rho'[b] the Jacobian of the already-decided levels'
        # logits w.r.t. the branching point. The ReLU derivative makes the
        # protected subspace sample-dependent, exactly as in LH-DNN.
        az = torch.matmul(a, z.unsqueeze(-1))
        gram = torch.matmul(a, a.transpose(1, 2).contiguous())
        eye = torch.eye(gram.size(-1), dtype=gram.dtype, device=gram.device).unsqueeze(0)
        coefficients = torch.linalg.solve(gram + float(eps) * eye, az)
        return torch.matmul(a.transpose(1, 2).contiguous(), coefficients).squeeze(-1)

    def _parent_baseline(self, level: int, prev_logits: torch.Tensor) -> torch.Tensor:
        buffer_name = self.parent_index_buffer_names[level]
        if buffer_name is None:
            raise RuntimeError(f"Missing parent index buffer for level {level}.")
        parent_index = getattr(self, buffer_name).to(device=prev_logits.device)
        return prev_logits.index_select(dim=1, index=parent_index)

    def _projected_head_logits(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """LH-DNN-form projection at HRN's branching point.

        `z` is the pooled trunk vector. It passes through the shared linear map
        `shared_linear` and `shared_relu`, giving the branching-point vector the
        heads read; `rho' = 1[pre_activation > 0]` is therefore idempotent and
        sample-dependent, as in LH-DNN. That shared map is what the guarantee
        actually bites on: stepping the convolutional trunk remains outside its
        scope. Each level is one direct affine head `W_l`, so
        `A = [W_1; ...; W_(l-1)] * rho'` is available without collapsing a
        redundant chain of linear factors. The coarse-to-fine residual is the
        score-space advantage built from detached parent scores.
        """
        # The fine level carries separate direct tree and leaf-CE heads. Neither
        # takes part in any projector because no lower-priority level follows
        # the fine level; both simply read the same projected branch input.
        classifiers_per_level = [
            [self.classifier_1],
            [self.classifier_2],
            [self.classifier_3, self.classifier_3_1],
        ]

        compute_dtype = torch.float32 if z.dtype in {torch.float16, torch.bfloat16} else z.dtype
        autocast_off = nullcontext()
        if z.device.type in {"cpu", "cuda", "xpu", "mps"}:
            autocast_off = torch.autocast(device_type=z.device.type, enabled=False)

        branch_inputs: List[torch.Tensor] = []
        advantage_per_level: List[List[torch.Tensor]] = []
        jacobians: List[torch.Tensor] = []

        with autocast_off:
            pre_activation = self.shared_linear(z.to(dtype=compute_dtype))
            z_work = self.shared_relu(pre_activation)
            rho_prime = (pre_activation > 0).to(dtype=compute_dtype)

            for level, classifiers in enumerate(classifiers_per_level):
                if level == 0:
                    z_level = z_work
                else:
                    a = torch.cat(jacobians, dim=1)
                    c_level = self._compute_projection_component(
                        z=z_work,
                        a=a,
                        eps=self.projection_eps,
                    )
                    # Backward-only: the forward value is untouched, only the
                    # gradient that reaches the trunk is projected.
                    z_level = z_work - c_level + c_level.detach()

                branch_inputs.append(z_level)
                natives = [
                    classifier(z_level.to(dtype=classifier.weight.dtype))
                    for classifier in classifiers
                ]
                if level == 0:
                    advantages = natives
                else:
                    baseline = self._parent_baseline(
                        level=level,
                        prev_logits=advantage_per_level[level - 1][0].detach(),
                    )
                    advantages = [native + baseline for native in natives]
                advantage_per_level.append(advantages)

                if level + 1 < len(classifiers_per_level):
                    # A is built from the NATIVE head weights, as in LH-DNN.
                    # Protecting every native score protects every advantage
                    # score by induction, so no cumulative term is needed.
                    head_matrix = classifiers[0].weight.detach().to(dtype=compute_dtype)
                    jacobians.append(
                        head_matrix.unsqueeze(0) * rho_prime.unsqueeze(1)
                    )

        return {
            "order_logits": advantage_per_level[0][0],
            "family_logits": advantage_per_level[1][0],
            "species_tree_logits": advantage_per_level[2][0],
            "species_ce_logits": advantage_per_level[2][1],
            # Keep the shared output contract. In the direct-head adaptation
            # there is no separate E-dimensional embedding, so these are the
            # D-dimensional representations read by the three level heads.
            "embeddings_per_level": branch_inputs,
        }

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        _ = targets
        x = self.features(x)

        if self.projection_enabled:
            # Pool before branching: the LH projector needs the shared vertex the
            # direct linear heads read from.
            projected = self._projected_head_logits(self.pooling(x).flatten(1))
            order_logits = projected["order_logits"]
            family_logits = projected["family_logits"]
            species_tree_logits = projected["species_tree_logits"]
            species_ce_logits = projected["species_ce_logits"]
            embeddings_per_level = projected["embeddings_per_level"]
        else:
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
            embeddings_per_level = [emb_order, emb_family, emb_species]

            order_logits = self.classifier_1(self.relu(emb_order))
            family_logits = self.classifier_2(self.relu(emb_family))
            species_tree_logits = self.classifier_3(self.relu(emb_species))
            species_ce_logits = self.classifier_3_1(self.relu(emb_species))

        order_sig = torch.sigmoid(order_logits)
        family_sig = torch.sigmoid(family_logits)
        species_sig = torch.sigmoid(species_tree_logits)

        logits_per_level: List[torch.Tensor] = [order_logits, family_logits, species_ce_logits]
        # Upstream HRN evaluates the tree heads as independent sigmoid scores
        # and the auxiliary leaf CE head as a class distribution.
        effective_probs_per_level = [
            order_sig,
            family_sig,
            F.softmax(species_ce_logits, dim=-1),
        ]

        # HCC constrains the emitted score triple: the coarse and middle tree
        # logits and `species_ce_logits` (classifier_3_1), the head that produces
        # HRN's reported fine score. Constraining the fine tree head instead
        # would leave the reported prediction untouched, and its logits carry a
        # large shared negative offset that makes the children sum sit far from
        # the parent. The corrected middle logits re-enter the tree branch, whose
        # own fine head keeps its raw logits.
        tree_logits_per_level = [order_logits, family_logits, species_tree_logits]
        hcc_output = self.hcc.apply(logits_per_level)
        effective_logits_per_level = hcc_output["effective_logits_per_level"]
        effective_tree_scores_per_level = None
        if effective_logits_per_level is not None:
            effective_probs_per_level = [
                torch.sigmoid(effective_logits_per_level[0]),
                torch.sigmoid(effective_logits_per_level[1]),
                F.softmax(effective_logits_per_level[2], dim=-1),
            ]
            effective_tree_scores_per_level = [
                torch.sigmoid(effective_logits_per_level[0]),
                torch.sigmoid(effective_logits_per_level[1]),
                species_sig,
            ]

        return {
            "logits_per_level": logits_per_level,
            "effective_logits_per_level": effective_logits_per_level,
            "effective_probs_per_level": effective_probs_per_level,
            "tree_scores_per_level": [order_sig, family_sig, species_sig],
            "tree_logits_per_level": tree_logits_per_level,
            "projected_logits_per_level": hcc_output["projected_logits_per_level"],
            "effective_tree_scores_per_level": effective_tree_scores_per_level,
            "hcc_diagnostics": hcc_output["hcc_diagnostics"],
            "species_ce_logits": species_ce_logits,
            "embeddings_per_level": embeddings_per_level,
        }
