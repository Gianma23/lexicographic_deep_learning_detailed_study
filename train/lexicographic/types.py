from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


GradTuple = Tuple[Optional[torch.Tensor], ...]
MaskMap = Dict[str, List[bool]]
LevelGradMap = Dict[str, GradTuple]

# Exact gradient-support blocks for three coarse-to-fine level objectives.
# The omitted-config default reproduces the historical H-CAST-oriented
# T1/T2/T3 partition: p123 == t1, p12 == t2, and p1 == t3.
GRADIENT_BLOCK_NAMES = ("p1", "p2", "p3", "p12", "p13", "p23", "p123")
DEFAULT_GRADIENT_BLOCKS = ("p123", "p12", "p1")


@dataclass(frozen=True)
class LexicographicConfig:
    enabled: bool = False
    eps: float = 1e-12
    log_metrics: bool = True
    projection_mode: str = "coarse_first"


@dataclass
class TrunkGradState:
    trunk_masks: MaskMap
    level_grad_map: LevelGradMap


@dataclass
class LexicographicUpdateState:
    trunk_masks: MaskMap
    level_grad_map: LevelGradMap
    projected_grads: Dict[str, GradTuple]
