from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.hcc import HccController
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
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01),
                    nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01),
                    nn.MaxPool2d(2),
                ]
            )
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TimmFeatureBackbone(nn.Module):
    """Expose the final spatial feature map from a timm model."""

    def __init__(self, timm_model: nn.Module):
        super().__init__()
        self.timm_model = timm_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.timm_model.forward_features(x)


class KerasMultiHeadAttention(nn.Module):
    """Keras MHA projection semantics backed by PyTorch's optimized SDPA."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        key_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError(
                "HT-CapsNet requires torch.nn.functional.scaled_dot_product_attention "
                "(PyTorch 2.0 or newer)."
            )
        self.query_dim = int(query_dim)
        self.context_dim = int(context_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.dropout = float(dropout)
        if self.query_dim <= 0 or self.context_dim <= 0:
            raise ValueError("Attention query/context dimensions must be positive.")
        if self.num_heads <= 0 or self.key_dim <= 0:
            raise ValueError("Attention num_heads/key_dim must be positive.")

        projection_dim = self.num_heads * self.key_dim
        self.query_projection = nn.Linear(self.query_dim, projection_dim)
        self.key_projection = nn.Linear(self.context_dim, projection_dim)
        self.value_projection = nn.Linear(self.context_dim, projection_dim)
        self.output_projection = nn.Linear(projection_dim, self.query_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
        ):
            nn.init.xavier_uniform_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.key_dim,
        ).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q = self._split_heads(self.query_projection(query))
        k = self._split_heads(self.key_projection(key))
        v = self._split_heads(self.value_projection(value))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            query.size(0), query.size(1), self.num_heads * self.key_dim
        )
        return self.output_projection(attended)


def _initial_level_weights(num_classes_per_level: List[int]) -> torch.Tensor:
    total = float(sum(num_classes_per_level))
    inverse_shares = [1.0 - (float(value) / total) for value in num_classes_per_level]
    normalizer = float(sum(inverse_shares))
    return torch.tensor([value / normalizer for value in inverse_shares], dtype=torch.float32)


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
        try:
            import timm

            model = timm.create_model(
                "tf_efficientnet_b7",
                pretrained=normalized_weights == "imagenet",
                num_classes=0,
                global_pool="",
            )
        except Exception as exc:
            requested = "ImageNet-pretrained " if normalized_weights == "imagenet" else ""
            raise RuntimeError(
                f"HT-CapsNet requires the timm {requested}tf_efficientnet_b7 backbone. "
                "Install a compatible timm/PyTorch build and cache/download requested weights; "
                "the paper configuration never falls back to random initialization."
            ) from exc
        return _TimmFeatureBackbone(model)

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
        attn_key_dim: int = 32,
        attn_dropout: float = 0.0,
        attn_postprocess: str = "layernorm",
        input_size: int = 224,
        hcc_cfg: Optional[Dict[str, Any]] = None,
        train_epochs: int = 1,
    ):
        super().__init__()
        self.num_classes_per_level = [int(v) for v in num_classes_per_level]
        self.depth = len(self.num_classes_per_level)
        self.routing_iters = int(routing_iters)
        self.primary_dim = int(primary_dim)
        self.attn_postprocess = _normalize_attn_postprocess(attn_postprocess)
        self._keras_efficientnet_rescale = _normalize_backbone_name(backbone_name) == "efficientnet_b7"

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
            self.post_attn_norms.append(nn.LayerNorm(spec.d_out, eps=1e-6))

        self.attn_layers = nn.ModuleList()
        for level, spec in enumerate(self.level_specs):
            q_dim = spec.d_out
            kv_dim = spec.d_in if level > 0 else q_dim
            self.attn_layers.append(
                KerasMultiHeadAttention(
                    query_dim=q_dim,
                    context_dim=kv_dim,
                    num_heads=int(attn_heads),
                    key_dim=int(attn_key_dim),
                    dropout=attn_dropout,
                )
            )

        self.register_buffer(
            "level_loss_weights",
            _initial_level_weights(self.num_classes_per_level),
        )

        self.taxonomy_temperature = float(taxonomy_temperature)
        self.mask_threshold_high = float(mask_threshold_high)
        self.mask_threshold_low = float(mask_threshold_low)
        self.mask_temperature = float(mask_temperature)
        self.mask_center = float(mask_center)
        self.parent_of = self._normalize_parent_of(taxonomy)
        self.hcc = HccController(
            num_classes_per_level=self.num_classes_per_level,
            taxonomy=taxonomy,
            hcc_cfg=hcc_cfg,
            train_epochs=train_epochs,
        )

    def set_epoch(self, epoch: int) -> None:
        self.hcc.set_epoch(epoch)

    def set_hcc_final_test_active(self, active: bool) -> None:
        self.hcc.set_final_test_active(active)

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

    def _prepare_backbone_input(self, x: torch.Tensor) -> torch.Tensor:
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
        # Keras EfficientNet embeds a 1/255 rescaling layer. The official
        # HT-CapsNet pipeline applies per-image standardization before it.
        if self._keras_efficientnet_rescale:
            x = x / 255.0
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
        # Keras feature maps are NHWC and are flattened in that memory order.
        flat = feat.permute(0, 2, 3, 1).contiguous().reshape(bsz, -1)
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
            attn_out = self.attn_layers[level](caps_out, prev_predictions, prev_predictions)
        else:
            attn_out = self.attn_layers[level](caps_out, caps_out, caps_out)
        return self._fuse_attention_output(level, caps_out, attn_out)

    @torch.no_grad()
    def post_optimizer_step(self, loss_aux: Optional[Dict[str, Any]]) -> None:
        """Apply callback-style dynamic loss weights for the following batch."""
        if not isinstance(loss_aux, dict):
            return
        next_weights = loss_aux.get("next_level_loss_weights")
        if not isinstance(next_weights, torch.Tensor):
            return
        next_weights = next_weights.to(
            device=self.level_loss_weights.device,
            dtype=self.level_loss_weights.dtype,
        ).view(-1)
        if next_weights.shape != self.level_loss_weights.shape:
            raise ValueError(
                "HT-CapsNet next-level loss weights do not match the configured hierarchy depth."
            )
        self.level_loss_weights.copy_(next_weights)

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

        hcc_output = self.hcc.apply(logits_per_level)

        return {
            "logits_per_level": logits_per_level,
            "effective_probs_per_level": [torch.softmax(logits, dim=-1) for logits in logits_per_level],
            "orthonormal_plugin_scores_per_level": logits_per_level,
            "secondary_caps": secondary_caps,
            "routing_stats": routing_stats,
            "level_loss_weights": self.level_loss_weights.detach().clone(),
            **hcc_output,
        }
