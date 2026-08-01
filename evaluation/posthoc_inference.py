"""Post-hoc inference rules that do not modify or train model parameters."""

from typing import Any, Dict, List, Mapping, Sequence

import torch

from models.orthonormal_plugin.config import INPUT_KEY
from models.orthonormal_plugin.topology import build_topology


class IdentityFrameHierCosInference:
    """Apply Hier-COS subspace-norm decoding to native per-level scores.

    The concatenated native scores are treated as coordinates in an identity
    taxonomy frame. This is an inference-only assumption: it does not claim
    that the source model was trained to produce a Hier-COS representation.
    """

    def __init__(
        self,
        num_classes_per_level: Sequence[int],
        taxonomy: Mapping[str, Any],
        owner: str = "Post-hoc identity-frame Hier-COS inference",
    ) -> None:
        self.num_classes_per_level = [int(value) for value in num_classes_per_level]
        topology = build_topology(
            num_classes_per_level=self.num_classes_per_level,
            taxonomy=dict(taxonomy),
            owner=owner,
        )
        self.level_subspace_masks = [
            mask.detach().to(device="cpu")
            for mask in topology["level_subspace_masks"]
        ]

    def _native_scores(self, output: Mapping[str, Any]) -> List[torch.Tensor]:
        raw_scores = output.get(INPUT_KEY, output.get("logits_per_level"))
        if not isinstance(raw_scores, (list, tuple)):
            raise ValueError(
                "Post-hoc Hier-COS inference requires model output key "
                f"`{INPUT_KEY}` or `logits_per_level` as a level-aligned list."
            )
        if len(raw_scores) != len(self.num_classes_per_level):
            raise ValueError(
                "Post-hoc Hier-COS inference received "
                f"{len(raw_scores)} score levels, expected {len(self.num_classes_per_level)}."
            )

        validated: List[torch.Tensor] = []
        batch_size = None
        for level, (scores, expected_width) in enumerate(
            zip(raw_scores, self.num_classes_per_level)
        ):
            if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
                raise ValueError(
                    f"Native score level {level} must have shape [B, C], "
                    f"got {type(scores).__name__}."
                )
            if int(scores.size(1)) != int(expected_width):
                raise ValueError(
                    f"Native score level {level} has width {int(scores.size(1))}; "
                    f"expected {int(expected_width)}."
                )
            if batch_size is None:
                batch_size = int(scores.size(0))
            elif int(scores.size(0)) != batch_size:
                raise ValueError("Native score levels must share one batch size.")
            validated.append(scores)
        return validated

    def scores_from_output(self, output: Mapping[str, Any]) -> List[torch.Tensor]:
        """Return raw Hier-COS projection norms for every hierarchy level."""
        native_scores = self._native_scores(output)
        node_coordinates = torch.cat(native_scores, dim=1)
        squared = node_coordinates.pow(2)

        scores_per_level: List[torch.Tensor] = []
        for mask in self.level_subspace_masks:
            device_mask = mask.to(
                device=node_coordinates.device,
                dtype=node_coordinates.dtype,
            )
            score_sq = torch.matmul(
                squared,
                device_mask.transpose(0, 1).contiguous(),
            )
            scores_per_level.append(torch.sqrt(score_sq.clamp_min(0.0)))
        return scores_per_level

    def transform_output(self, output: Mapping[str, Any]) -> Dict[str, Any]:
        """Build the minimal output consumed by the shared metric evaluator."""
        return {"logits_per_level": self.scores_from_output(output)}


class NativeHierCosNodeSoftmaxInference:
    """Decode native Hier-COS directly from its global node softmax values."""

    @staticmethod
    def scores_from_output(output: Mapping[str, Any]) -> List[torch.Tensor]:
        node_logits = output.get("node_logits")
        level_node_ids = output.get("hiercos_level_node_ids")
        if not isinstance(node_logits, torch.Tensor) or node_logits.ndim != 2:
            raise ValueError(
                "Native Hier-COS node-softmax inference requires `node_logits` "
                "with shape [B, N]."
            )
        if not isinstance(level_node_ids, (list, tuple)) or not level_node_ids:
            raise ValueError(
                "Native Hier-COS node-softmax inference requires "
                "`hiercos_level_node_ids` as a non-empty list."
            )

        # Upstream Hier-COS trains its node distribution with one global
        # log_softmax(abs(node_logits)). Use the corresponding probabilities
        # directly, without constructing taxonomy-subspace projection norms.
        node_probabilities = torch.softmax(node_logits.abs(), dim=1)
        scores_per_level: List[torch.Tensor] = []
        for level, node_ids in enumerate(level_node_ids):
            if not isinstance(node_ids, torch.Tensor) or node_ids.ndim != 1:
                raise ValueError(
                    f"Hier-COS node ids for level {level} must be a 1D tensor."
                )
            ids = node_ids.to(device=node_logits.device, dtype=torch.long)
            scores_per_level.append(node_probabilities.index_select(1, ids))
        return scores_per_level

    def transform_output(self, output: Mapping[str, Any]) -> Dict[str, Any]:
        """Build the minimal output consumed by the shared metric evaluator."""
        return {"logits_per_level": self.scores_from_output(output)}

