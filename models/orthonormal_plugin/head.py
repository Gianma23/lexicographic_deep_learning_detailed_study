from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .topology import build_topology
from .transforms import build_transformation_module


def _parse_bool_like(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


class FrozenBlockDiagonalClassifier(nn.Module):
    """Block-diagonal classifier with one independent block per level."""

    def __init__(
        self,
        block_sizes: Sequence[int],
        mode: str = "orthonormal_random",
        trainable: bool = False,
        owner: str = "Orthonormal plugin",
    ):
        super().__init__()
        parsed_block_sizes = [int(size) for size in block_sizes]
        if any(size <= 0 for size in parsed_block_sizes):
            raise ValueError(f"{owner} block sizes must be positive, got {parsed_block_sizes}.")
        if mode not in {"orthonormal_random", "identity"}:
            raise ValueError(
                f"Unsupported {owner} block classifier mode '{mode}'. "
                "Expected one of ['orthonormal_random', 'identity']."
            )
        self.block_sizes = parsed_block_sizes
        blocks: List[nn.Linear] = []
        for block_size in self.block_sizes:
            block = nn.Linear(block_size, block_size, bias=False)
            with torch.no_grad():
                if mode == "orthonormal_random":
                    random_matrix = torch.randn(block_size, block_size)
                    q, _ = torch.linalg.qr(random_matrix, mode="reduced")
                    block.weight.copy_(q)
                else:
                    block.weight.copy_(torch.eye(block_size))
            block.weight.requires_grad_(bool(trainable))
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward_chunks(self, chunks: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(chunks) != len(self.block_sizes):
            raise ValueError(f"Expected {len(self.block_sizes)} chunks, got {len(chunks)}.")
        out: List[torch.Tensor] = []
        for level, (chunk, block, block_size) in enumerate(zip(chunks, self.blocks, self.block_sizes)):
            if not isinstance(chunk, torch.Tensor) or chunk.ndim != 2:
                raise ValueError(f"Block classifier chunk {level} must have shape [B, C].")
            if int(chunk.size(1)) != int(block_size):
                raise ValueError(
                    f"Block classifier chunk {level} has width {int(chunk.size(1))}; "
                    f"expected {int(block_size)}."
                )
            out.append(block(chunk))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.block_sizes, dim=1)
        return torch.cat(self.forward_chunks(chunks), dim=1)


def init_fixed_classifier(
    classifier: nn.Linear,
    width: int,
    mode: str,
    block_sizes: Optional[Sequence[int]] = None,
    owner: str = "Orthonormal plugin",
) -> None:
    if not isinstance(mode, str):
        raise ValueError(f"{owner} fixed_frame_mode must be a string.")
    with torch.no_grad():
        if mode == "orthonormal_random":
            random_matrix = torch.randn(int(width), int(width))
            q, _ = torch.linalg.qr(random_matrix, mode="reduced")
            classifier.weight.copy_(q)
        elif mode == "orthonormal_block_random":
            if block_sizes is None:
                raise ValueError(f"{owner} fixed_frame_mode='orthonormal_block_random' requires block sizes.")
            parsed_block_sizes = [int(size) for size in block_sizes]
            if any(size <= 0 for size in parsed_block_sizes):
                raise ValueError(f"{owner} block sizes must be positive, got {parsed_block_sizes}.")
            if sum(parsed_block_sizes) != int(width):
                raise ValueError(
                    f"{owner} block sizes sum to {sum(parsed_block_sizes)}, expected fixed frame width {width}."
                )
            classifier.weight.zero_()
            start = 0
            for block_size in parsed_block_sizes:
                end = start + block_size
                random_matrix = torch.randn(block_size, block_size)
                q, _ = torch.linalg.qr(random_matrix, mode="reduced")
                classifier.weight[start:end, start:end].copy_(q)
                start = end
        elif mode == "identity":
            classifier.weight.copy_(torch.eye(int(width)))
        else:
            raise ValueError(
                f"Unsupported {owner} fixed_frame_mode '{mode}'. "
                "Expected one of ['orthonormal_random', 'orthonormal_block_random', 'identity']."
            )
    classifier.weight.requires_grad_(mode == "learnable_orthonormal")


def build_fixed_classifier(
    width: int,
    mode: str,
    fixed_frame_per_level: bool = False,
    block_sizes: Optional[Sequence[int]] = None,
    owner: str = "Orthonormal plugin",
) -> nn.Module:
    resolved_mode = str(mode)
    trainable = resolved_mode == "learnable_orthonormal"
    if trainable:
        resolved_mode = "orthonormal_random"
    resolved_per_level = _parse_bool_like(fixed_frame_per_level, default=False)
    if resolved_mode == "orthonormal_block_random":
        resolved_mode = "orthonormal_random"
        resolved_per_level = True

    if resolved_per_level:
        if block_sizes is None:
            raise ValueError(f"{owner} fixed_frame_per_level=true requires block sizes.")
        return FrozenBlockDiagonalClassifier(
            block_sizes=block_sizes,
            mode=resolved_mode,
            trainable=trainable,
            owner=owner,
        )

    classifier = nn.Linear(int(width), int(width), bias=False)
    init_fixed_classifier(
        classifier=classifier,
        width=int(width),
        mode="learnable_orthonormal" if trainable else resolved_mode,
        block_sizes=block_sizes,
        owner=owner,
    )
    return classifier


def validate_scores_per_level(
    scores_per_level: Any,
    num_classes_per_level: Sequence[int],
    owner: str = "Orthonormal plugin",
) -> List[torch.Tensor]:
    if not isinstance(scores_per_level, list) or len(scores_per_level) != len(num_classes_per_level):
        raise ValueError(
            f"{owner} requires `orthonormal_plugin_scores_per_level` as a list aligned with hierarchy depth."
        )

    validated: List[torch.Tensor] = []
    batch_size: Optional[int] = None
    for level, (scores, classes) in enumerate(zip(scores_per_level, num_classes_per_level)):
        if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
            raise ValueError(f"{owner} scores for level {level} must be a tensor with shape [B, C].")
        if int(scores.size(1)) != int(classes):
            raise ValueError(
                f"{owner} scores for level {level} have width {int(scores.size(1))}; "
                f"expected {int(classes)}."
            )
        if batch_size is None:
            batch_size = int(scores.size(0))
        elif int(scores.size(0)) != batch_size:
            raise ValueError(f"{owner} scores must use the same batch size at every level.")
        validated.append(scores)
    return validated


class OrthonormalPluginHead(nn.Module):
    def __init__(
        self,
        num_classes_per_level: List[int],
        taxonomy: Dict[str, Any],
        transform_mode: str = "full",
        fixed_frame_mode: str = "orthonormal_random",
        fixed_frame_per_level: bool = False,
        owner: str = "Orthonormal plugin",
    ):
        super().__init__()
        self.owner = owner
        self.num_classes_per_level = [int(v) for v in num_classes_per_level]
        topology = build_topology(
            num_classes_per_level=self.num_classes_per_level,
            taxonomy=taxonomy,
            owner=owner,
        )
        self.depth = int(topology["depth"])
        self.total_nodes = int(topology["total_nodes"])

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

        self.transform_mode = transform_mode
        self.fixed_frame_mode = "orthonormal_random" if fixed_frame_mode == "orthonormal_block_random" else fixed_frame_mode
        self.fixed_frame_per_level = (
            _parse_bool_like(fixed_frame_per_level, default=False)
            or fixed_frame_mode == "orthonormal_block_random"
        )
        self.f_theta = build_transformation_module(self.total_nodes, mode=transform_mode, owner=owner)
        self.fixed_classifier = build_fixed_classifier(
            width=self.total_nodes,
            mode=fixed_frame_mode,
            fixed_frame_per_level=self.fixed_frame_per_level,
            block_sizes=self.num_classes_per_level,
            owner=owner,
        )

    def parameter_groups(self, base_lr: float, transform_lr_scale: float = 1.0):
        transform_params = (
            [p for p in self.f_theta.parameters() if p.requires_grad]
            if self.transform_mode != "final_only"
            else []
        )
        classifier_params = [p for p in self.fixed_classifier.parameters() if p.requires_grad]
        groups = []
        if transform_params:
            groups.append({"params": transform_params, "lr": float(base_lr) * float(transform_lr_scale)})
        if classifier_params:
            groups.append({"params": classifier_params, "lr": float(base_lr)})
        return groups

    def level_node_ids(self) -> List[torch.Tensor]:
        return [getattr(self, name) for name in self.level_node_id_names]

    def _level_subspace_scores(self, node_logits: torch.Tensor) -> List[torch.Tensor]:
        squared = node_logits.pow(2)
        scores: List[torch.Tensor] = []
        for mask_name in self.level_subspace_mask_names:
            mask = getattr(self, mask_name).to(device=node_logits.device, dtype=node_logits.dtype)
            level_scores_sq = torch.matmul(squared, mask.transpose(0, 1).contiguous())
            scores.append(torch.sqrt(level_scores_sq.clamp_min(0.0)))
        return scores

    def forward_tensor(self, z: torch.Tensor) -> Dict[str, Any]:
        if not isinstance(z, torch.Tensor) or z.ndim != 2:
            raise ValueError(f"{self.owner} input must be a tensor with shape [B, N].")
        if int(z.size(1)) != self.total_nodes:
            raise ValueError(
                f"{self.owner} input width {int(z.size(1))} does not match total taxonomy nodes {self.total_nodes}."
            )

        transformed = self.f_theta(z)
        node_logits = self.fixed_classifier(transformed)
        if isinstance(self.fixed_classifier, FrozenBlockDiagonalClassifier):
            node_logits_per_level = list(torch.split(node_logits, self.num_classes_per_level, dim=1))
        else:
            node_logits_per_level = None
        # Hier-COS decodes the raw taxonomy-subspace projection norms returned
        # by `get_distances`; softmax is not part of its prediction rule.
        logits_per_level = self._level_subspace_scores(node_logits)
        level_node_ids = self.level_node_ids()

        return {
            "logits_per_level": logits_per_level,
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

    def forward_scores(self, scores_per_level: Any) -> Dict[str, Any]:
        scores = validate_scores_per_level(
            scores_per_level=scores_per_level,
            num_classes_per_level=self.num_classes_per_level,
            owner=self.owner,
        )
        if self.transform_mode == "final_only" and isinstance(self.fixed_classifier, FrozenBlockDiagonalClassifier):
            node_logits_per_level = self.fixed_classifier.forward_chunks(scores)
            node_logits = torch.cat(node_logits_per_level, dim=1)
            logits_per_level = self._level_subspace_scores(node_logits)
            level_node_ids = self.level_node_ids()
            return {
                "logits_per_level": logits_per_level,
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
        return self.forward_tensor(torch.cat(scores, dim=1))

    def forward(self, scores_per_level: Any) -> Dict[str, Any]:
        return self.forward_scores(scores_per_level)
