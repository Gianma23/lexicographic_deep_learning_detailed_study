import inspect
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .config import INPUT_KEY, plugin_section, validate_disabled_mixup
from .head import OrthonormalPluginHead


class OrthonormalPluginWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        cfg: Any,
        num_classes_per_level: List[int],
        taxonomy: Optional[Dict[str, Any]],
    ):
        super().__init__()
        if taxonomy is None:
            raise ValueError("orthonormal_plugin.enabled=true requires taxonomy with `parent_of` mappings.")
        validate_disabled_mixup(cfg)
        plugin_cfg = plugin_section(cfg)
        self.base_model = base_model
        self.transform_lr_scale = float(plugin_cfg.get("transform_lr_scale", 1.0))
        self.orthonormal_plugin = OrthonormalPluginHead(
            num_classes_per_level=num_classes_per_level,
            taxonomy=taxonomy,
            transform_mode=plugin_cfg.get("transform_mode", "full"),
            fixed_frame_mode=plugin_cfg.get("fixed_frame_mode", "orthonormal_random"),
        )

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base_model, "set_epoch"):
            self.base_model.set_epoch(epoch)

    def set_hcc_final_test_active(self, active: bool) -> None:
        if hasattr(self.base_model, "set_hcc_final_test_active"):
            self.base_model.set_hcc_final_test_active(active)

    def _base_parameter_groups(self, base_lr: float, **kwargs):
        if not hasattr(self.base_model, "parameter_groups"):
            params = [p for p in self.base_model.parameters() if p.requires_grad]
            return [{"params": params, "lr": float(base_lr)}] if params else []

        signature = inspect.signature(self.base_model.parameter_groups)
        accepted = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters and value is not None
        }
        return list(self.base_model.parameter_groups(base_lr=base_lr, **accepted))

    def parameter_groups(
        self,
        base_lr: float,
        backbone_lr_scale: Optional[float] = None,
        transform_lr_scale: Optional[float] = None,
        trunk_lr_scale: Optional[float] = None,
    ):
        groups = self._base_parameter_groups(
            base_lr=base_lr,
            backbone_lr_scale=backbone_lr_scale,
            transform_lr_scale=transform_lr_scale,
            trunk_lr_scale=trunk_lr_scale,
        )
        plugin_scale = self.transform_lr_scale if transform_lr_scale is None else float(transform_lr_scale)
        groups.extend(
            self.orthonormal_plugin.parameter_groups(
                base_lr=base_lr,
                transform_lr_scale=plugin_scale,
            )
        )
        return groups

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        output = self.base_model(*args, **kwargs)
        if not isinstance(output, dict):
            raise ValueError("orthonormal_plugin expects the wrapped model to return a dict.")
        scores_per_level = output.get(INPUT_KEY)
        if scores_per_level is None:
            raise ValueError(
                f"orthonormal_plugin.enabled=true requires model output key `{INPUT_KEY}`."
            )

        plugin_output = self.orthonormal_plugin(scores_per_level)
        merged = dict(output)
        merged.update(plugin_output)
        merged["effective_logits_per_level"] = None
        return merged
