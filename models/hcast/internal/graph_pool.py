"""Define Graph Pooling."""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath

from .utils import segment_mean_nd


def semantic_projection(
    assignments_vis: torch.Tensor,
    p_tokens: torch.Tensor,
    p_centroids: torch.Tensor,
    p_ref: torch.Tensor = None,
    floor: float = 0.0,
    eps: float = 1e-6,
    metric: str = "dot",
) -> torch.Tensor:
    """Apply token-aware post-hoc semantic projection to visual assignments.

    Args:
      assignments_vis: Tensor of shape [B, S, C].
      p_tokens: Detached token coarse probs of shape [B, S, K].
      p_centroids: Detached centroid coarse probs of shape [B, C, K].
      p_ref: Optional detached coarse reference probs of shape [B, K].
      floor: Minimum semantic weight in [0, 1].
      eps: Numerical epsilon for renormalization.
      metric: Compatibility metric ("dot" or "top1_ref").

    Returns:
      Projected assignments with same shape/dtype/device as assignments_vis.
    """
    if assignments_vis.ndim != 3:
        raise ValueError("assignments_vis must have shape [B, S, C].")
    if p_tokens.ndim != 3:
        raise ValueError("p_tokens must have shape [B, S, K].")
    if p_centroids.ndim != 3:
        raise ValueError("p_centroids must have shape [B, C, K].")
    if assignments_vis.shape[0] != p_tokens.shape[0] or assignments_vis.shape[0] != p_centroids.shape[0]:
        raise ValueError("Batch sizes for assignments and coarse probabilities must match.")
    if assignments_vis.shape[1] != p_tokens.shape[1]:
        raise ValueError("Token dimension mismatch between assignments and p_tokens.")
    if assignments_vis.shape[2] != p_centroids.shape[1]:
        raise ValueError("Centroid dimension mismatch between assignments and p_centroids.")
    if p_tokens.shape[2] != p_centroids.shape[2]:
        raise ValueError("Class dimension mismatch between p_tokens and p_centroids.")

    floor = min(1.0, max(0.0, float(floor)))
    eps = float(eps)
    metric = str(metric).strip().lower()

    # Keep projection numerics stable under AMP while preserving outward dtype.
    work_dtype = torch.float32 if assignments_vis.dtype in (torch.float16, torch.bfloat16) else assignments_vis.dtype
    a_vis = assignments_vis.to(dtype=work_dtype)
    p_tokens_detached = p_tokens.detach().to(dtype=work_dtype)
    p_centroids_detached = p_centroids.detach().to(dtype=work_dtype)
    p_ref_detached = None if p_ref is None else p_ref.detach().to(dtype=work_dtype)

    if metric == "dot":
        compat = torch.einsum("bsk,bck->bsc", p_tokens_detached, p_centroids_detached)
    elif metric in {"top1_ref", "top1"}:
        if p_ref_detached is None or p_ref_detached.ndim != 2:
            raise ValueError("p_ref with shape [B, K] is required for top1_ref metric.")
        top1 = torch.argmax(p_ref_detached, dim=-1, keepdim=True)
        tok_idx = top1.unsqueeze(1).expand(-1, p_tokens_detached.shape[1], -1)
        cen_idx = top1.unsqueeze(1).expand(-1, p_centroids_detached.shape[1], -1)
        tok_prob = torch.gather(p_tokens_detached, dim=2, index=tok_idx).squeeze(-1)
        cen_prob = torch.gather(p_centroids_detached, dim=2, index=cen_idx).squeeze(-1)
        compat = tok_prob.unsqueeze(-1) * cen_prob.unsqueeze(1)
    else:
        raise ValueError(f"Unsupported semantic projection metric: {metric}")

    compat = compat.clamp_(0.0, 1.0)
    floor_t = torch.full_like(compat, fill_value=floor)
    weights = floor_t + (1.0 - floor_t) * compat
    weights = weights.clamp_(min=floor, max=1.0)

    a_tmp = a_vis * weights
    denom = a_tmp.sum(dim=-1, keepdim=True)
    a_proj = a_tmp / denom.clamp_min(eps)

    # If a row collapses numerically, keep the original visual assignment.
    collapsed = denom <= eps
    if collapsed.any():
        a_proj = torch.where(collapsed.expand_as(a_proj), a_vis, a_proj)

    return a_proj.to(dtype=assignments_vis.dtype)


class Attention(nn.Module):
    """Similar to timm.models.vision_transformer.Attention but we do not use
    additional Fully Connected Layers.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Same as timm.models.vision_transformer.Block"""

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.bias = nn.Parameter(torch.zeros(dim).normal_(0, 1e-2))

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x - torch.mean(x, dim=1, keepdim=True) + self.bias.view(1, 1, -1)
        return x


class GraphPooling(nn.Module):
    def __init__(
        self,
        num_clusters=4,
        d_model=512,
        dropout=0.1,
        l2_normalize_for_fps=True,
        num_heads=12,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        """Perfrom Graph Pooling."""
        super().__init__()
        self.centroid_fc = Block(dim=d_model, num_heads=num_heads, qkv_bias=qkv_bias, norm_layer=norm_layer)
        self.fc1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(nn.LayerNorm(d_model * 4), nn.Linear(d_model * 4, d_model, bias=True))

        self._num_clusters = num_clusters
        self._l2_normalize_for_fps = l2_normalize_for_fps

    def _fill_with_mean(self, src, mask):
        """A helper function to fill invalid entries with mean values."""
        bs, sl, cs = src.shape
        if mask is not None:
            mean_src = valid_mean(src, ~mask).unsqueeze(1).type_as(src)
            fill_mask = mask.unsqueeze(2).expand(-1, -1, cs)
            filled_src = torch.where(fill_mask, mean_src.expand(-1, sl, -1), src)
        else:
            mean_src = torch.mean(src, dim=1, keepdim=True).type_as(src)
            filled_src = src

        return filled_src, mean_src

    @staticmethod
    @torch.no_grad()
    def _farthest_point_sampler_torch(points: torch.Tensor, npoints: int, start_idx: int = 0) -> torch.Tensor:
        """Batched farthest point sampling implemented in native PyTorch."""
        bs, num_points, _ = points.shape
        if npoints <= 0:
            raise ValueError(f"npoints must be > 0, got {npoints}")
        if num_points <= 0:
            raise ValueError("points must have at least one entry for FPS")

        eff_npoints = min(npoints, num_points)
        sampled_inds = torch.empty((bs, eff_npoints), dtype=torch.long, device=points.device)
        batch_inds = torch.arange(bs, device=points.device)

        start = torch.full((bs,), int(start_idx), dtype=torch.long, device=points.device)
        start = start.clamp_(min=0, max=num_points - 1)
        sampled_inds[:, 0] = start

        min_dist = torch.full((bs, num_points), float("inf"), dtype=points.dtype, device=points.device)
        selected = points[batch_inds, start]
        dist = torch.sum((points - selected.unsqueeze(1)) ** 2, dim=-1)
        min_dist = torch.minimum(min_dist, dist)
        min_dist.scatter_(1, start.unsqueeze(1), -1.0)

        for i in range(1, eff_npoints):
            next_idx = torch.argmax(min_dist, dim=1)
            sampled_inds[:, i] = next_idx
            selected = points[batch_inds, next_idx]
            dist = torch.sum((points - selected.unsqueeze(1)) ** 2, dim=-1)
            min_dist = torch.minimum(min_dist, dist)
            min_dist.scatter_(1, next_idx.unsqueeze(1), -1.0)

        if eff_npoints < npoints:
            pad = sampled_inds[:, -1:].expand(-1, npoints - eff_npoints)
            sampled_inds = torch.cat([sampled_inds, pad], dim=1)

        return sampled_inds

    def forward(self, cls_token, src, mask, projection_ctx=None):
        """Feedforward for clustering with Transformer."""
        bs, sl, cs = src.shape

        filled_src, mean_src = self._fill_with_mean(src, mask)
        padded_src = torch.cat([mean_src, filled_src], dim=1)

        if self._l2_normalize_for_fps:
            sampling_src = F.normalize(padded_src, dim=-1)
        else:
            sampling_src = padded_src

        sampled_inds = self._farthest_point_sampler_torch(
            sampling_src.to(torch.float32),
            self._num_clusters + 1,
            start_idx=0,
        )
        sampled_inds = sampled_inds[:, 1:] - 1
        sampled_inds = sampled_inds.clamp_(min=0, max=max(0, sl - 1))
        assert (sampled_inds >= 0).all()
        unfold_sampled_inds = sampled_inds.unsqueeze(2).expand(-1, -1, cs)

        # Visual assignment logits (unchanged)
        node_features = self.centroid_fc(src)
        centroid_features = torch.gather(node_features, 1, unfold_sampled_inds)
        normed_centroid_features = F.normalize(centroid_features, dim=-1)
        normed_node_features = F.normalize(node_features, dim=-1)
        logits_vis = torch.einsum("bij,bjk->bik", normed_node_features, normed_centroid_features.transpose(1, 2))
        logits_vis = logits_vis * 5
        assignments_vis = torch.softmax(logits_vis, dim=-1)

        pooled_assignments = assignments_vis
        returned_logits = logits_vis

        if projection_ctx is not None and bool(projection_ctx.get("enabled", False)):
            projection_head = projection_ctx.get("projection_head", None)
            if projection_head is None:
                # Backward-compat fallback for older context key.
                projection_head = projection_ctx.get("coarse_head", None)
            p_ref = projection_ctx.get("p_ref", None)
            if projection_head is not None:
                with torch.no_grad():
                    token_coarse_logits = projection_head(node_features.detach())
                    centroid_coarse_logits = projection_head(centroid_features.detach())

                    if token_coarse_logits.ndim == 3 and centroid_coarse_logits.ndim == 3:
                        p_tokens = torch.softmax(token_coarse_logits, dim=-1)
                        p_centroids = torch.softmax(centroid_coarse_logits, dim=-1)
                    else:
                        p_tokens, p_centroids = None, None

                shape_ok = (
                    p_tokens is not None
                    and p_centroids is not None
                    and p_tokens.shape[0] == assignments_vis.shape[0]
                    and p_tokens.shape[1] == assignments_vis.shape[1]
                    and p_centroids.shape[0] == assignments_vis.shape[0]
                    and p_centroids.shape[1] == assignments_vis.shape[2]
                    and p_tokens.shape[2] == p_centroids.shape[2]
                )
                if shape_ok:
                    pooled_assignments = semantic_projection(
                        assignments_vis=assignments_vis,
                        p_tokens=p_tokens,
                        p_centroids=p_centroids,
                        p_ref=p_ref,
                        floor=float(projection_ctx.get("floor", 0.0)),
                        eps=float(projection_ctx.get("eps", 1e-6)),
                        metric=str(projection_ctx.get("metric", "dot")),
                    )
                    proj_eps = float(projection_ctx.get("eps", 1e-6))
                    # Return logits consistent with assignment actually used downstream.
                    returned_logits = torch.log(pooled_assignments.clamp_min(proj_eps)).to(dtype=logits_vis.dtype)

        # Average pooling within clusters using projected (or visual) assignment.
        fc1_cls_token_src = self.fc1(torch.cat([cls_token, src], dim=1))
        fc1_cls_token, fc1_src = fc1_cls_token_src[:, :1], fc1_cls_token_src[:, 1:]
        normalizer = torch.einsum(
            "bij,bjk->bik",
            pooled_assignments.transpose(1, 2),
            torch.ones((bs, sl, 1), dtype=src.dtype, device=src.device),
        )
        centroids = torch.einsum("bij,bjk->bik", pooled_assignments.transpose(1, 2), fc1_src)
        centroids /= normalizer

        fc2_cls_token_centroids = self.fc2(torch.cat([fc1_cls_token, centroids], dim=1))
        centroids = fc2_cls_token_centroids[:, 1:, :] + torch.gather(src, 1, unfold_sampled_inds)
        cls_token = fc2_cls_token_centroids[:, :1, :] + cls_token

        return cls_token, centroids, returned_logits, sampled_inds


def valid_mean(x, mask):
    """Compute mean of x given valid mask."""
    mask = mask.type_as(x).unsqueeze(2)
    sum_mask = torch.clamp(torch.sum(mask, dim=1), min=1)
    masked_x = x * mask
    mean_x = torch.sum(masked_x, dim=1) / sum_mask
    return mean_x

