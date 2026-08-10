from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn


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
        owner: str = "Hier-COS",
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
    owner: str = "Hier-COS",
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
    owner: str = "Hier-COS",
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
