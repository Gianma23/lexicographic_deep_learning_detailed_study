"""Checkpoint-only evaluation and post-hoc inference utilities."""

from .posthoc_inference import (
    IdentityFrameHierCosInference,
    NativeHierCosNodeSoftmaxInference,
)

__all__ = [
    "IdentityFrameHierCosInference",
    "NativeHierCosNodeSoftmaxInference",
]

