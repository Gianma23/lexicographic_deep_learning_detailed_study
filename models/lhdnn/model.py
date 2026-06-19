from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn


def _normalize_parent_of(taxonomy: Dict[str, Any]) -> Dict[int, Dict[int, int]]:
    if not taxonomy or "parent_of" not in taxonomy:
        raise ValueError("Taxonomy with `parent_of` mappings is required for LH-DNN advantage topology.")

    parent_of_raw = taxonomy["parent_of"]
    if not isinstance(parent_of_raw, Mapping):
        raise ValueError("`taxonomy['parent_of']` must be a mapping of level->(child->parent).")

    out: Dict[int, Dict[int, int]] = {}
    for level_key, mapping in parent_of_raw.items():
        if not isinstance(mapping, Mapping):
            raise ValueError(f"`taxonomy['parent_of'][{level_key}]` must be a child->parent mapping.")
        level = int(level_key)
        out[level] = {int(child): int(parent) for child, parent in mapping.items()}
    return out


def _build_parent_index_per_level(
    num_classes_per_level: Sequence[int],
    taxonomy: Dict[str, Any],
) -> List[Optional[torch.Tensor]]:
    parent_of = _normalize_parent_of(taxonomy)
    depth = len(num_classes_per_level)
    parent_idx_per_level: List[Optional[torch.Tensor]] = [None]

    for level in range(1, depth):
        mapping = parent_of.get(level)
        if mapping is None:
            raise ValueError(
                f"Missing taxonomy mapping for level transition {level - 1}->{level}. "
                f"Expected `taxonomy['parent_of'][{level}]`."
            )

        num_parents = int(num_classes_per_level[level - 1])
        num_children = int(num_classes_per_level[level])
        parent_idx = torch.full((num_children,), -1, dtype=torch.long)
        for child, parent in mapping.items():
            if child < 0 or child >= num_children:
                raise ValueError(
                    f"Invalid child id {child} in `taxonomy['parent_of'][{level}]`; "
                    f"expected range [0, {num_children})."
                )
            if parent < 0 or parent >= num_parents:
                raise ValueError(
                    f"Invalid parent id {parent} in `taxonomy['parent_of'][{level}]`; "
                    f"expected range [0, {num_parents})."
                )
            parent_idx[child] = parent

        if bool((parent_idx < 0).any().item()):
            missing = torch.nonzero(parent_idx < 0, as_tuple=False).flatten().tolist()
            preview = missing[:10]
            raise ValueError(
                f"`taxonomy['parent_of'][{level}]` must define one parent per child. "
                f"Missing child ids (first 10): {preview}"
            )
        parent_idx_per_level.append(parent_idx)

    return parent_idx_per_level


class _ConvStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LHDNNModel(nn.Module):
    """Paper-aligned LH-DNN with fixed large topology and always-on projection/advantage paths."""

    LARGE_CHANNELS = [64, 128, 256, 512]
    SHARED_DIM = 512

    def __init__(
        self,
        num_classes_per_level: List[int],
        taxonomy: Optional[Dict[str, Any]] = None,
        image_size: int = 32,
        projection_cfg: Optional[Dict[str, Any]] = None,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_classes_per_level = [int(v) for v in num_classes_per_level]
        self.depth = len(self.num_classes_per_level)
        if self.depth < 2:
            raise ValueError("LH-DNN requires hierarchy depth >= 2.")
        if any(v <= 0 for v in self.num_classes_per_level):
            raise ValueError(
                f"All hierarchy class counts must be > 0, got {self.num_classes_per_level}."
            )

        if taxonomy is None:
            raise ValueError(
                "LH-DNN advantage topology is always enabled and requires taxonomy. "
                "Provide dataset taxonomy with parent mappings."
            )

        projection_cfg = projection_cfg or {}
        self.channels = list(self.LARGE_CHANNELS)
        self.shared_dim = int(self.SHARED_DIM)
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("`image_size` must be > 0.")

        self.projection_eps = float(projection_cfg.get("eps", 1e-6))
        if self.projection_eps <= 0.0:
            raise ValueError("`model.projection.eps` must be > 0.")

        # Exposed for parity checks and clarity.
        self.projection_enabled = True
        self.advantage_enabled = True
        self.use_relu_derivative = True

        self.stages = nn.ModuleList()
        current_in = int(in_channels)
        for out_ch in self.channels:
            self.stages.append(_ConvStage(current_in, out_ch))
            current_in = out_ch

        self.flattened_dim = self._infer_flattened_dim(image_size=self.image_size, in_channels=int(in_channels))
        self.shared_linear = nn.Linear(self.flattened_dim, self.shared_dim)
        self.shared_relu = nn.ReLU(inplace=False)
        self.heads = nn.ModuleList(
            [nn.Linear(self.shared_dim, int(classes)) for classes in self.num_classes_per_level]
        )

        cumulative = 0
        for level in range(1, self.depth):
            cumulative += int(self.num_classes_per_level[level - 1])
            if self.shared_dim <= cumulative:
                raise ValueError(
                    "Projection non-triviality condition violated: "
                    f"shared_dim ({self.shared_dim}) must be > sum(classes[:{level}]) ({cumulative})."
                )

        parent_idx_per_level = _build_parent_index_per_level(
            num_classes_per_level=self.num_classes_per_level,
            taxonomy=taxonomy,
        )
        self.parent_index_buffers: List[Optional[str]] = [None]
        for level in range(1, self.depth):
            buffer_name = f"parent_index_level_{level}"
            self.register_buffer(buffer_name, parent_idx_per_level[level], persistent=False)
            self.parent_index_buffers.append(buffer_name)

    def _infer_flattened_dim(self, image_size: int, in_channels: int) -> int:
        dummy = torch.zeros(1, int(in_channels), int(image_size), int(image_size))
        with torch.no_grad():
            out = dummy
            for stage in self.stages:
                out = stage(out)
        flat = int(out[0].numel())
        if flat <= 0:
            raise ValueError(
                "Invalid flattened feature size after shared conv stages. "
                f"Got {flat} for image_size={image_size} and channels={self.channels}."
            )
        return flat

    @staticmethod
    def _compute_projection_component(
        z: torch.Tensor,
        rho_prime: torch.Tensor,
        prev_weights: torch.Tensor,
        eps: float,
    ) -> Dict[str, torch.Tensor]:
        # A[b] = prev_weights * rho_prime[b] where prev_weights rows are detached class vectors.
        # Shapes:
        #   z: [B, D]
        #   rho_prime: [B, D]
        #   prev_weights: [R, D]
        #   A: [B, R, D]
        a = prev_weights.unsqueeze(0) * rho_prime.unsqueeze(1)
        z_col = z.unsqueeze(-1)
        az = torch.matmul(a, z_col)  # [B, R, 1]

        gram = torch.matmul(a, a.transpose(1, 2).contiguous())  # [B, R, R]
        eye = torch.eye(gram.size(-1), dtype=gram.dtype, device=gram.device).unsqueeze(0)
        gram = gram + float(eps) * eye
        gram_inv_az = torch.linalg.solve(gram, az)  # [B, R, 1]
        c = torch.matmul(a.transpose(1, 2).contiguous(), gram_inv_az).squeeze(-1)  # [B, D]
        return {"c": c, "a": a}

    def _parent_baseline(self, level: int, prev_logits: torch.Tensor) -> torch.Tensor:
        if level <= 0:
            raise ValueError("Parent baseline is defined only for levels > 0.")
        buffer_name = self.parent_index_buffers[level]
        if buffer_name is None:
            raise RuntimeError(f"Missing parent index buffer for level {level}.")
        parent_idx = getattr(self, buffer_name).to(device=prev_logits.device)
        return prev_logits.index_select(dim=1, index=parent_idx)

    def forward(
        self,
        x: torch.Tensor,
        return_projection_debug: bool = False,
    ) -> Dict[str, Any]:
        h = x
        for stage in self.stages:
            h = stage(h)

        flat = h.flatten(1)
        shared_input = flat.to(dtype=self.shared_linear.weight.dtype)
        k = torch.matmul(shared_input, self.shared_linear.weight.transpose(0, 1).contiguous())
        if self.shared_linear.bias is not None:
            k = k + self.shared_linear.bias
        z = self.shared_relu(k)

        rho_prime = (k > 0).to(dtype=z.dtype)

        logits_per_level: List[torch.Tensor] = []
        projection_debug: List[Optional[Dict[str, torch.Tensor]]] = []

        compute_dtype = torch.float32 if z.dtype in {torch.float16, torch.bfloat16} else z.dtype
        autocast_off = nullcontext()
        if z.device.type in {"cpu", "cuda", "xpu", "mps"}:
            autocast_off = torch.autocast(device_type=z.device.type, enabled=False)

        with autocast_off:
            z_work = z.to(dtype=compute_dtype)
            rho_work = rho_prime.to(dtype=compute_dtype)

            for level, head in enumerate(self.heads):
                if level > 0:
                    prev_weights = torch.cat(
                        [self.heads[j].weight.detach().to(dtype=compute_dtype) for j in range(level)],
                        dim=0,
                    )
                    proj_out = self._compute_projection_component(
                        z=z_work,
                        rho_prime=rho_work,
                        prev_weights=prev_weights,
                        eps=self.projection_eps,
                    )
                    c_level = proj_out["c"].to(dtype=z.dtype)
                    z_level = z - c_level + c_level.detach()
                    if return_projection_debug:
                        projection_debug.append(
                            {
                                "a": proj_out["a"].detach(),
                                "c": c_level.detach(),
                            }
                        )
                else:
                    z_level = z
                    if return_projection_debug:
                        projection_debug.append(None)

                head_input = z_level.to(dtype=head.weight.dtype)
                logits_level = torch.matmul(head_input, head.weight.transpose(0, 1).contiguous())
                if head.bias is not None:
                    logits_level = logits_level + head.bias
                if level > 0:
                    baseline = self._parent_baseline(level=level, prev_logits=logits_per_level[level - 1].detach())
                    logits_level = logits_level + baseline
                logits_per_level.append(logits_level)

        output: Dict[str, Any] = {
            "logits_per_level": logits_per_level,
            "orthonormal_plugin_scores_per_level": logits_per_level,
        }
        if return_projection_debug:
            output["projection_debug"] = projection_debug
            output["shared_embedding"] = z
            output["rho_prime"] = rho_prime
        return output
