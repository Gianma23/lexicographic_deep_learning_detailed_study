from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


GradTuple = Tuple[Optional[torch.Tensor], ...]
MaskMap = Dict[str, List[bool]]
LevelGradMap = Dict[str, GradTuple]


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
