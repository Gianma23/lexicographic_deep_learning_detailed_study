"""Define CAST model for classification following DeiT convention.

Modified from:
    https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    https://github.com/facebookresearch/deit/blob/main/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
try:
    from timm.models.registry import register_model  # old timm
except Exception:  # pragma: no cover
    from timm.models import register_model
try:
    from timm.models.layers import PatchEmbed
    from timm.models.layers import trunc_normal_
except Exception:  # pragma: no cover
    from timm.layers import PatchEmbed
    from timm.layers import trunc_normal_

from .utils import segment_mean_nd
from .graph_pool import GraphPooling
from .modules import Pooling, ConvStem

__all__ = [
    'cast_small',
    'cast_small_deep',
    'cast_base',
    'cast_base_deep',
]


class CAST(VisionTransformer):
    def __init__(self, nb_classes, *args, **kwargs):
        depths = kwargs['depth']
        lexproj_cfg = kwargs.pop("lexproj", None)
        # timm factory may forward metadata kwargs that VisionTransformer
        # does not accept in newer releases.
        for _k in (
            "pretrained_cfg",
            "pretrained_cfg_overlay",
            "checkpoint_path",
            "cache_dir",
            "scriptable",
            "exportable",
            "no_jit",
        ):
            kwargs.pop(_k, None)
        if "num_classes" not in kwargs and len(nb_classes) > 0:
            kwargs["num_classes"] = int(nb_classes[0])
        # These entries do not exist in timm.VisionTransformer.
        num_clusters = kwargs.pop('num_clusters', [64, 32, 16, 8])
        kwargs['depth'] = sum(kwargs['depth'])
        super().__init__(**kwargs)
        if not hasattr(self, "pre_logits"):
            self.pre_logits = nn.Identity()

        # Do not tackle distillation-token heads.
        # In newer timm versions these attributes may be absent altogether.
        assert getattr(self, "dist_token", None) is None, "dist_token is not None."
        assert getattr(self, "head_dist", None) is None, "head_dist is not None."

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        if len(nb_classes) == 3:
            self.num_classes = nb_classes[0]
            self.num_family = nb_classes[1]
            self.num_manufacturer = nb_classes[2]
        elif len(nb_classes) == 2:
            self.num_classes = nb_classes[0]
            self.num_family = nb_classes[1]
            self.num_manufacturer = 0

        self.family_head = nn.Linear(self.embed_dim, self.num_family) if self.num_family > 0 else nn.Identity()
        if len(nb_classes) == 3:
            self.manufacturer_head = nn.Linear(self.embed_dim, self.num_manufacturer) if self.num_manufacturer > 0 else nn.Identity()
            self.manufacturer_head.apply(self._init_weights)

        self.family_head.apply(self._init_weights)
        self.lexproj_cfg = self._build_lexproj_cfg(lexproj_cfg)

        cumsum_depth = [0]
        for d in depths:
            cumsum_depth.append(d + cumsum_depth[-1])

        blocks = []
        pools = []
        for ind, depth in enumerate(depths):

            # Build Attention Blocks.
            blocks.append(self.blocks[cumsum_depth[ind]:cumsum_depth[ind+1]])

            # Build Pooling layers.
            pool = Pooling(
                pool_block=GraphPooling(
                    num_clusters=num_clusters[ind],
                    d_model=kwargs['embed_dim'],
                    l2_normalize_for_fps=False))
            # Last graph pooling is not needed.
            if ind == len(depths) - 1:
                for param in pool.pool_block.fc1.parameters():
                    param.requires_grad = False
                for param in pool.pool_block.fc2.parameters():
                    param.requires_grad = False
                for param in pool.pool_block.centroid_fc.parameters():
                    param.requires_grad = False
            pools.append(pool)

        self.blocks1, self.pool1 = blocks[0], pools[0]
        self.blocks2, self.pool2 = blocks[1], pools[1]
        self.blocks3, self.pool3 = blocks[2], pools[2]
        self.blocks4, self.pool4 = blocks[3], pools[3]

    @staticmethod
    def _build_lexproj_cfg(cfg):
        cfg = cfg or {}
        scope = str(cfg.get("scope", "train_eval")).strip().lower()
        if scope not in {"train_eval", "train", "eval"}:
            scope = "train_eval"
        lower_iter_k = int(cfg.get("lower_iter_k", 3))
        if lower_iter_k < 1:
            lower_iter_k = 1
        backtrack_factor = float(cfg.get("backtrack_factor", 0.5))
        if not (0.0 < backtrack_factor < 1.0):
            backtrack_factor = 0.5
        backtrack_max_steps = int(cfg.get("backtrack_max_steps", 5))
        if backtrack_max_steps < 0:
            backtrack_max_steps = 0
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "scope": scope,
            "eta_lower": float(cfg.get("eta_lower", 5.0)),
            "eta_upper": float(cfg.get("eta_upper", 5.0)),
            "lower_iter_k": lower_iter_k,
            "coarse_tol": float(cfg.get("coarse_tol", 5e-3)),
            "middle_tol": float(cfg.get("middle_tol", 5e-3)),
            "backtrack_factor": backtrack_factor,
            "backtrack_max_steps": backtrack_max_steps,
            "eps": float(cfg.get("eps", 1e-12)),
        }

    def _lexproj_active(self, targets):
        if not self.lexproj_cfg["enabled"]:
            return False
        if self.num_manufacturer <= 0:
            return False
        if not isinstance(targets, torch.Tensor):
            return False
        if targets.ndim != 2 or targets.shape[1] != 3:
            return False
        scope = self.lexproj_cfg["scope"]
        if scope == "train" and not self.training:
            return False
        if scope == "eval" and self.training:
            return False
        return True

    @staticmethod
    def _detach_dict_tensors(payload):
        detached = {}
        for key, value in payload.items():
            if torch.is_tensor(value):
                detached[key] = value.detach()
            else:
                detached[key] = value
        return detached

    def _prepare_tokens(self, x, y):
        x = self.patch_embed(x)
        n, h, w, c = x.shape
        # Collect features within each segment.
        y = y.unsqueeze(1).float()
        y = F.interpolate(y, x.shape[1:3], mode='nearest')
        y = y.squeeze(1).long()
        x = segment_mean_nd(x, y)
        # Create padding mask.
        ones = torch.ones((n, h, w, 1), dtype=x.dtype, device=x.device)
        avg_ones = segment_mean_nd(ones, y).squeeze(-1)
        x_padding_mask = avg_ones <= 0.5

        # Collect positional encodings within each segment.
        pos_embed = self.pos_embed[:, 1:].view(1, h, w, c).expand(n, -1, -1, -1)
        pos_embed = segment_mean_nd(pos_embed, y)

        # Add positional encodings.
        x = self.pos_drop(x + pos_embed)

        # Add class token.
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        cls_token = cls_token + self.pos_embed[:, :1]
        return x, cls_token, x_padding_mask

    @staticmethod
    def _run_block(x, cls_token, nn_block):
        cls_x = torch.cat([cls_token, x], dim=1)
        cls_x = nn_block(cls_x).type_as(x)
        cls_token = cls_x[:, :1, :]
        x = cls_x[:, 1:, :]
        return x, cls_token, cls_x

    def _run_pool(self, cls_token, x, x_pad_mask, pool_block, pool_delta=None):
        graph_pool = pool_block.pool_block
        pool_logit_visual, pool_inds = graph_pool.compute_visual_logits(src=x, mask=x_pad_mask)
        if pool_delta is not None and pool_delta.shape == pool_logit_visual.shape:
            pool_logit = pool_logit_visual + pool_delta.to(
                dtype=pool_logit_visual.dtype, device=pool_logit_visual.device
            )
        else:
            pool_logit = pool_logit_visual
        assignments = graph_pool.assignments_from_logits(pool_logit)
        cls_token, centroid = graph_pool.pool_from_assignments(
            cls_token=cls_token,
            src=x,
            assignments=assignments,
            sampled_inds=pool_inds,
        )
        pool_pad_mask = torch.zeros(
            (pool_logit.shape[0], pool_logit.shape[-1]),
            dtype=torch.bool,
            device=pool_logit.device,
        )
        return cls_token, pool_logit, pool_logit_visual, centroid, pool_pad_mask, pool_inds

    def _run_stage(self, x, cls_token, x_pad_mask, nn_block, pool_block, norm_block, pool_delta=None):
        x, cls_token, cls_x = self._run_block(x=x, cls_token=cls_token, nn_block=nn_block)
        cls_token, pool_logit, pool_logit_visual, centroid, pool_pad_mask, pool_inds = self._run_pool(
            cls_token=cls_token,
            x=x,
            x_pad_mask=x_pad_mask,
            pool_block=pool_block,
            pool_delta=pool_delta,
        )
        if norm_block is not None:
            out = norm_block(cls_x)[:, 0]
        else:
            out = cls_x[:, 0]
        return {
            "block": x,
            "cls_token": cls_token,
            "pool_logit": pool_logit,
            "pool_logit_visual": pool_logit_visual,
            "centroid": centroid,
            "pool_padding_mask": pool_pad_mask,
            "pool_inds": pool_inds,
            "out": out,
        }

    def _forward_backbone(self, x, y, pool1_delta=None, pool2_delta=None):
        x, cls_token, x_padding_mask = self._prepare_tokens(x, y)

        stage1 = self._run_stage(
            x=x,
            cls_token=cls_token,
            x_pad_mask=x_padding_mask,
            nn_block=self.blocks1,
            pool_block=self.pool1,
            norm_block=None,
            pool_delta=pool1_delta,
        )

        stage2 = self._run_stage(
            x=stage1["centroid"],
            cls_token=stage1["cls_token"],
            x_pad_mask=stage1["pool_padding_mask"],
            nn_block=self.blocks2,
            pool_block=self.pool2,
            norm_block=None,
            pool_delta=pool2_delta,
        )

        stage3 = self._run_stage(
            x=stage2["centroid"],
            cls_token=stage2["cls_token"],
            x_pad_mask=stage2["pool_padding_mask"],
            nn_block=self.blocks3,
            pool_block=self.pool3,
            norm_block=None,
            pool_delta=None,
        )

        stage4 = self._run_stage(
            x=stage3["centroid"],
            cls_token=stage3["cls_token"],
            x_pad_mask=stage3["pool_padding_mask"],
            nn_block=self.blocks4,
            pool_block=self.pool4,
            norm_block=self.norm,
            pool_delta=None,
        )
        out4 = self.pre_logits(stage4["out"])

        intermediates = {
            'logit1': stage1["pool_logit"],
            'centroid1': stage1["centroid"],
            'block1': stage1["block"],
            'padding_mask1': x_padding_mask,
            'sampled_inds1': stage1["pool_inds"],
            'logit2': stage2["pool_logit"],
            'centroid2': stage2["centroid"],
            'block2': stage2["block"],
            'padding_mask2': stage1["pool_padding_mask"],
            'sampled_inds2': stage2["pool_inds"],
            'out2': stage2["out"],
            'logit3': stage3["pool_logit"],
            'centroid3': stage3["centroid"],
            'block3': stage3["block"],
            'padding_mask3': stage2["pool_padding_mask"],
            'sampled_inds3': stage3["pool_inds"],
            'out3': stage3["out"],
            'logit4': stage4["pool_logit"],
            'centroid4': stage4["centroid"],
            'block4': stage4["block"],
            'padding_mask4': stage3["pool_padding_mask"],
            'out4': out4,
            'sampled_inds4': stage4["pool_inds"],
        }

        return {
            "intermediates": intermediates,
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
            "stage4": stage4,
        }

    def _compute_three_level_logits(self, intermediates):
        fine_logits = self.head(intermediates['out2'])
        middle_logits = self.family_head(intermediates['out3'])
        coarse_logits = self.manufacturer_head(intermediates['out4'])
        return fine_logits, middle_logits, coarse_logits

    @staticmethod
    def _grad_or_zeros(loss, wrt, retain_graph):
        grad = torch.autograd.grad(loss, wrt, retain_graph=retain_graph, allow_unused=True)[0]
        if grad is None:
            return torch.zeros_like(wrt)
        return grad

    def _project_halfspace(self, delta, grad):
        inner = torch.sum(grad * delta)
        if float(inner.detach().item()) <= 0.0:
            return delta
        denom = torch.sum(grad * grad) + self.lexproj_cfg["eps"]
        return delta - (inner / denom) * grad

    def _lower_backtrack_scale(self, x, y, targets, delta, coarse_ref, middle_ref):
        scale = 1.0
        coarse_tol = self.lexproj_cfg["coarse_tol"]
        middle_tol = self.lexproj_cfg["middle_tol"]
        max_steps = self.lexproj_cfg["backtrack_max_steps"]
        factor = self.lexproj_cfg["backtrack_factor"]
        for _ in range(max_steps + 1):
            with torch.no_grad():
                trial = self._forward_backbone(x=x, y=y, pool1_delta=scale * delta, pool2_delta=None)
                _, middle_logits, coarse_logits = self._compute_three_level_logits(trial["intermediates"])
                coarse_loss = F.cross_entropy(coarse_logits, targets[:, 2])
                middle_loss = F.cross_entropy(middle_logits, targets[:, 1])
            if coarse_loss <= (coarse_ref + coarse_tol) and middle_loss <= (middle_ref + middle_tol):
                return scale
            scale = scale * factor
        return 0.0

    def _upper_backtrack_scale(self, x, y, targets, lower_delta, upper_delta, coarse_ref):
        scale = 1.0
        coarse_tol = self.lexproj_cfg["coarse_tol"]
        max_steps = self.lexproj_cfg["backtrack_max_steps"]
        factor = self.lexproj_cfg["backtrack_factor"]
        for _ in range(max_steps + 1):
            with torch.no_grad():
                trial = self._forward_backbone(
                    x=x,
                    y=y,
                    pool1_delta=lower_delta,
                    pool2_delta=scale * upper_delta,
                )
                _, _, coarse_logits = self._compute_three_level_logits(trial["intermediates"])
                coarse_loss = F.cross_entropy(coarse_logits, targets[:, 2])
            if coarse_loss <= (coarse_ref + coarse_tol):
                return scale
            scale = scale * factor
        return 0.0

    def _forward_features_with_lexproj(self, x, y, targets):
        targets = targets.long()

        # Step 1: visual baseline for lower-stage projection.
        vis_pass = self._forward_backbone(x=x, y=y, pool1_delta=None, pool2_delta=None)
        vis_intermediates = vis_pass["intermediates"]
        fine_logits, middle_logits, coarse_logits = self._compute_three_level_logits(vis_intermediates)
        fine_loss = F.cross_entropy(fine_logits, targets[:, 0])
        middle_loss = F.cross_entropy(middle_logits, targets[:, 1])
        coarse_loss = F.cross_entropy(coarse_logits, targets[:, 2])
        lower_logits_visual = vis_pass["stage1"]["pool_logit_visual"]
        g_coarse_lower = self._grad_or_zeros(coarse_loss, lower_logits_visual, retain_graph=True)
        g_middle_lower = self._grad_or_zeros(middle_loss, lower_logits_visual, retain_graph=True)
        g_fine_lower = self._grad_or_zeros(fine_loss, lower_logits_visual, retain_graph=False)

        delta_lower = -self.lexproj_cfg["eta_lower"] * g_fine_lower
        for _ in range(self.lexproj_cfg["lower_iter_k"]):
            # Priority order: coarse first, then middle.
            delta_lower = self._project_halfspace(delta_lower, g_coarse_lower)
            delta_lower = self._project_halfspace(delta_lower, g_middle_lower)
        delta_lower = delta_lower.detach()
        lower_scale = self._lower_backtrack_scale(
            x=x,
            y=y,
            targets=targets,
            delta=delta_lower,
            coarse_ref=coarse_loss.detach(),
            middle_ref=middle_loss.detach(),
        )
        applied_lower_delta = (lower_scale * delta_lower).detach()

        # Step 2: upper-stage projection while lower-stage projection is applied.
        upper_vis_pass = self._forward_backbone(x=x, y=y, pool1_delta=applied_lower_delta, pool2_delta=None)
        upper_vis_intermediates = upper_vis_pass["intermediates"]
        _, middle_logits_upper, coarse_logits_upper = self._compute_three_level_logits(upper_vis_intermediates)
        middle_loss_upper = F.cross_entropy(middle_logits_upper, targets[:, 1])
        coarse_loss_upper = F.cross_entropy(coarse_logits_upper, targets[:, 2])
        upper_logits_visual = upper_vis_pass["stage2"]["pool_logit_visual"]
        g_coarse_upper = self._grad_or_zeros(coarse_loss_upper, upper_logits_visual, retain_graph=True)
        g_middle_upper = self._grad_or_zeros(middle_loss_upper, upper_logits_visual, retain_graph=False)

        delta_upper = -self.lexproj_cfg["eta_upper"] * g_middle_upper
        delta_upper = self._project_halfspace(delta_upper, g_coarse_upper)
        delta_upper = delta_upper.detach()
        upper_scale = self._upper_backtrack_scale(
            x=x,
            y=y,
            targets=targets,
            lower_delta=applied_lower_delta,
            upper_delta=delta_upper,
            coarse_ref=coarse_loss_upper.detach(),
        )
        applied_upper_delta = (upper_scale * delta_upper).detach()

        # Step 3: final pass with applied lower and upper corrections.
        final_pass = self._forward_backbone(
            x=x,
            y=y,
            pool1_delta=applied_lower_delta,
            pool2_delta=applied_upper_delta,
        )
        return final_pass["intermediates"]

    def forward_features(self, x, y, targets=None):
        if self._lexproj_active(targets):
            if torch.is_grad_enabled():
                return self._forward_features_with_lexproj(x=x, y=y, targets=targets)
            with torch.enable_grad():
                intermediates = self._forward_features_with_lexproj(x=x, y=y, targets=targets)
            return self._detach_dict_tensors(intermediates)

        return self._forward_backbone(x=x, y=y, pool1_delta=None, pool2_delta=None)["intermediates"]

    def forward(self, x, y, targets=None):
        intermediates = self.forward_features(x=x, y=y, targets=targets)
        if self.num_manufacturer:
            manu_out = self.manufacturer_head(intermediates['out4'])
            family_out = self.family_head(intermediates['out3'])
            out = self.head(intermediates['out2'])
            return out, family_out, manu_out

        family_out = self.family_head(intermediates['out4'])
        out = self.head(intermediates['out3'])
        return out, family_out


@register_model
def cast_small(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=384, num_clusters=[64, 32, 16, 8],
        depth=[3, 3, 3, 2], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def cast_small_deep(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=384, num_clusters=[64, 32, 16, 8],
        depth=[6, 3, 3, 3], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def cast_base(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=768, num_clusters=[64, 32, 16, 8],
        depth=[3, 3, 3, 2], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def cast_base_deep(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=768, num_clusters=[64, 32, 16, 8],
        depth=[6, 3, 3, 3], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model
