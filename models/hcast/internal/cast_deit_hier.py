"""Define CAST model for classification following DeiT convention.

Modified from:
    https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    https://github.com/facebookresearch/deit/blob/main/models.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce
from operator import mul

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
        semantic_projection = kwargs.pop("semantic_projection", None) or {}
        semantic_stages = semantic_projection.get("stages", None)
        semantic_floor = float(semantic_projection.get("floor", 0.0))
        self.semantic_projection_enabled = bool(semantic_projection.get("enabled", False))
        self.semantic_projection_floor = min(1.0, max(0.0, semantic_floor))
        self.semantic_projection_eps = float(semantic_projection.get("eps", 1e-6))
        self.semantic_projection_metric = str(semantic_projection.get("metric", "dot")).strip().lower()
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

        #print('nb_classes', nb_classes)
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
        self.semantic_projection_stages = self._parse_projection_stages(semantic_stages)

        cumsum_depth = [0]
        for d in depths:
            cumsum_depth.append(d + cumsum_depth[-1])

        blocks = []
        pools = []
        for ind, depth in enumerate(depths):

            # Build Attention Blocks.
            blocks.append(self.blocks[cumsum_depth[ind]:cumsum_depth[ind+1]])

            # Build Pooling layers
            pool = Pooling(
                pool_block=GraphPooling(
                    num_clusters=num_clusters[ind],
                    d_model=kwargs['embed_dim'],
                    l2_normalize_for_fps=False))
            # Last graph pooling is not needed
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
        # --------------------------------------------------------------------------

    def _default_projection_stages(self):
        if getattr(self, "num_manufacturer", 0) > 0:
            return {2, 3}
        return {3}

    def _parse_projection_stages(self, stages):
        if stages is None:
            return self._default_projection_stages()
        if isinstance(stages, str):
            values = [v.strip() for v in stages.split(",") if v.strip()]
            return {int(v) for v in values}
        if isinstance(stages, (list, tuple, set)):
            return {int(v) for v in stages}
        return self._default_projection_stages()

    def _projection_head_for_stage(self, stage_idx):
        has_family = hasattr(self, "family_head") and not isinstance(self.family_head, nn.Identity)
        has_manufacturer = hasattr(self, "manufacturer_head") and not isinstance(self.manufacturer_head, nn.Identity)

        if has_manufacturer and int(stage_idx) >= 3:
            return self.manufacturer_head
        if has_family:
            return self.family_head
        if has_manufacturer:
            return self.manufacturer_head
        return None

    def _build_projection_ctx(self, cls_repr, stage_idx):
        projection_head = self._projection_head_for_stage(stage_idx=stage_idx)
        if projection_head is None:
            return None

        with torch.no_grad():
            coarse_logits = projection_head(cls_repr.detach())
            if coarse_logits.ndim != 2:
                return None
            p_ref = torch.softmax(coarse_logits, dim=-1)

        return {
            "enabled": True,
            "p_ref": p_ref,
            "projection_head": projection_head,
            "floor": self.semantic_projection_floor,
            "eps": self.semantic_projection_eps,
            "metric": self.semantic_projection_metric,
        }

    def _block_operations(self, x, cls_token, x_pad_mask,
                          nn_block, pool_block, norm_block, stage_idx):
        """Wrapper to define operations per block.
        """
        # Forward nn block with cls_token and x
        cls_x = torch.cat([cls_token, x], dim=1)
        cls_x = nn_block(cls_x).type_as(x)
        cls_token, x = cls_x[:, :1, :], cls_x[:, 1:, :]

        projection_ctx = None
        if self.semantic_projection_enabled and stage_idx in self.semantic_projection_stages:
            projection_ctx = self._build_projection_ctx(cls_x[:, 0, :], stage_idx=stage_idx)

        # Perform pooling only on x
        cls_token, pool_logit, centroid, pool_pad_mask, pool_inds = (
            pool_block(cls_token, x, x_pad_mask, projection_ctx=projection_ctx)
        )

        # Generate output by cls_token
        if norm_block is not None:
            out = norm_block(cls_x)[:, 0]
        else:
            out = cls_x[:, 0]

        return (x, cls_token, pool_logit, centroid,
                pool_pad_mask, pool_inds, out)

    def forward_features(self, x, y): 
        x = self.patch_embed(x) 
        N, H, W, C = x.shape
        # Collect features within each segment
        y = y.unsqueeze(1).float()
        y = F.interpolate(y, x.shape[1:3], mode='nearest')
        y = y.squeeze(1).long()  
        x = segment_mean_nd(x, y) 
        # Create padding mask
        ones = torch.ones((N, H, W, 1), dtype=x.dtype, device=x.device)
        avg_ones = segment_mean_nd(ones, y).squeeze(-1)
        x_padding_mask = avg_ones <= 0.5

        # Collect positional encodings within each segment
        pos_embed = self.pos_embed[:, 1:].view(1, H, W, C).expand(N, -1, -1, -1) 
        pos_embed = segment_mean_nd(pos_embed, y)  

        # Add positional encodings
        x = self.pos_drop(x + pos_embed)  

        # Add class token.
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        cls_token = cls_token + self.pos_embed[:, :1]

        # intermediate results
        intermediates = {}

        # Block1
        (block1, cls_token1, pool_logit1, centroid1,
         pool_padding_mask1, pool_inds1, out1) = self._block_operations(
            x, cls_token, x_padding_mask,
            self.blocks1, self.pool1, None, stage_idx=1)

        intermediates1 = {
            'logit1': pool_logit1, 'centroid1': centroid1, 'block1': block1,
            'padding_mask1': x_padding_mask, 'sampled_inds1': pool_inds1,
        }
        intermediates.update(intermediates1)


        # Block2
        (block2, cls_token2, pool_logit2, centroid2,
         pool_padding_mask2, pool_inds2, out2) = self._block_operations(
            centroid1, cls_token1, pool_padding_mask1,
            self.blocks2, self.pool2, None, stage_idx=2)

        intermediates2 = {
            'logit2': pool_logit2, 'centroid2': centroid2, 'block2': block2,
            'padding_mask2': pool_padding_mask1, 'sampled_inds2': pool_inds2, 'out2': out2, 
        }
        intermediates.update(intermediates2)

        # Block3
        (block3, cls_token3, pool_logit3, centroid3,
         pool_padding_mask3, pool_inds3, out3) = self._block_operations(
            centroid2, cls_token2, pool_padding_mask2,
            self.blocks3, self.pool3, None, stage_idx=3)

        intermediates3 = {
            'logit3': pool_logit3, 'centroid3': centroid3, 'block3': block3,
            'padding_mask3': pool_padding_mask2, 'sampled_inds3': pool_inds3, 'out3': out3,
        }
        intermediates.update(intermediates3)

        # Block4
        (block4, cls_token4, pool_logit4, centroid4,
         pool_padding_mask4, pool_inds4, out4) = self._block_operations(
            centroid3, cls_token3, pool_padding_mask3,
            self.blocks4, self.pool4, self.norm, stage_idx=4)

        out4 = self.pre_logits(out4)

        intermediates4 = {
            'logit4': pool_logit4, 'centroid4': centroid4, 'block4': block4,
            'padding_mask4': pool_padding_mask3, 'out4': out4, 'sampled_inds4': pool_inds4,
        }
        intermediates.update(intermediates4)

        return intermediates

    def forward(self, x, y):
        intermediates = self.forward_features(x, y)  
        if self.num_manufacturer:
            manu_out = self.manufacturer_head(intermediates['out4']) 
            family_out = self.family_head(intermediates['out3'])
            out = self.head(intermediates['out2']) 
            return out, family_out, manu_out
    
        else:
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
