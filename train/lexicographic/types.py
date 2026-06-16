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
    start_epoch: int = 0
    projection_mode: str = "coarse_first"
    projection_rule: str = "orthogonalize_all"

    def projection_active(self, epoch: int) -> bool:
        return self.enabled and int(epoch) >= int(self.start_epoch)


@dataclass
class TrunkGradState:
    trunk_masks: MaskMap
    level_grad_map: LevelGradMap


@dataclass
class LexicographicUpdateState:
    trunk_masks: MaskMap
    level_grad_map: LevelGradMap
    projected_grads: Dict[str, GradTuple]
