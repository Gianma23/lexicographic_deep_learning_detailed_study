"""Post-hoc inference rules that do not modify or train model parameters.

Every rule is one cell of a two-axis grid:

- **readout** — how a taxonomy node is scored from the model's node coordinates.
  `node_score` reads the node's own coordinate; `subspace_norm` takes the L2
  norm over the node's ancestors+self+descendants subspace.
- **transform** — nothing, or the HCC affine hierarchy projection applied to the
  coordinates before the readout.

Both readouts consume the same per-level node coordinates, so each model's own
inference is one cell of the grid rather than a separate rule: `node_score` for
classifier-head models such as H-CAST and HRN, `subspace_norm` for native
Hier-COS, and the `hcc_`-prefixed cell of the same readout for a checkpoint
trained with HCC. Nothing here trains or modifies parameters.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from models.common.hcc import HierarchicalAffineProjector
from models.common.subspace_supervision import subspace_norms
from models.hiercos.topology import build_topology


NODE_SCORE = "node_score"
SUBSPACE_NORM = "subspace_norm"
READOUTS = (NODE_SCORE, SUBSPACE_NORM)
HCC_PREFIX = "hcc_"
# The 2x2 grid, native-readout-agnostic and identical for every model.
INFERENCE_RULES = (
    NODE_SCORE,
    SUBSPACE_NORM,
    f"{HCC_PREFIX}{NODE_SCORE}",
    f"{HCC_PREFIX}{SUBSPACE_NORM}",
)


def split_rule_name(rule: str) -> tuple:
    """Split a grid cell name into `(readout, applies_hcc)`."""
    applies_hcc = rule.startswith(HCC_PREFIX)
    readout = rule[len(HCC_PREFIX):] if applies_hcc else rule
    if readout not in READOUTS:
        raise ValueError(
            f"Unknown inference rule `{rule}`; expected one of {list(INFERENCE_RULES)}."
        )
    return readout, applies_hcc


def uses_coordinate_magnitude(model_name: str) -> bool:
    """Report whether a node's evidence is its coordinate's magnitude.

    Hier-COS trains its node distribution as `log_softmax(|node_logits|)`, so the
    objective never constrains the sign of a frame coordinate and only the
    magnitude carries class evidence. Classifier logits are signed: a large
    negative logit means *not this class*, so they are read as-is.
    """
    return str(model_name) == "hiercos"


def _validate_level_scores(
    raw_scores: Any,
    num_classes_per_level: Sequence[int],
    owner: str,
    source_key: str,
) -> List[torch.Tensor]:
    """Check that `raw_scores` is a level-aligned list of `[B, C]` tensors."""
    if not isinstance(raw_scores, (list, tuple)):
        raise ValueError(
            f"{owner} requires model output key `{source_key}` as a "
            "level-aligned list."
        )
    if len(raw_scores) != len(num_classes_per_level):
        raise ValueError(
            f"{owner} received {len(raw_scores)} score levels, "
            f"expected {len(num_classes_per_level)}."
        )

    validated: List[torch.Tensor] = []
    batch_size = None
    for level, (scores, expected_width) in enumerate(
        zip(raw_scores, num_classes_per_level)
    ):
        if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
            raise ValueError(
                f"Node coordinate level {level} must have shape [B, C], "
                f"got {type(scores).__name__}."
            )
        if int(scores.size(1)) != int(expected_width):
            raise ValueError(
                f"Node coordinate level {level} has width {int(scores.size(1))}; "
                f"expected {int(expected_width)}."
            )
        if batch_size is None:
            batch_size = int(scores.size(0))
        elif int(scores.size(0)) != batch_size:
            raise ValueError("Node coordinate levels must share one batch size.")
        validated.append(scores)
    return validated


class PosthocInferenceRule:
    """One cell of the readout x transform grid, evaluated without training.

    The node coordinates are the tensor the model's own readout consumes: the
    fixed-layer node logits for native Hier-COS, and the native per-level scores
    for classifier-head models. For the latter, `subspace_norm` treats those
    per-level scores as coordinates in an identity taxonomy frame. That is an
    inference-only assumption, and because an L2 norm squares its inputs it also
    discards the sign of a signed logit.

    `hcc_`-prefixed rules apply `HierarchicalAffineProjector` to the coordinates
    first. The projector is parameter-free, so this is equivalent to a fully
    activated schedule (`alpha=1`) applied once to a trained checkpoint; it does
    not claim that the checkpoint was trained under HCC.
    """

    def __init__(
        self,
        rule: str,
        num_classes_per_level: Sequence[int],
        taxonomy: Mapping[str, Any],
        model_name: str,
        owner: str = "Post-hoc inference",
    ) -> None:
        self.rule = str(rule)
        self.readout, self.applies_hcc = split_rule_name(self.rule)
        self.num_classes_per_level = [int(value) for value in num_classes_per_level]
        self.model_name = str(model_name)
        self.owner = f"{owner} `{self.rule}`"
        self.magnitude = uses_coordinate_magnitude(self.model_name)

        self.projector: Optional[HierarchicalAffineProjector] = None
        if self.applies_hcc:
            if len(self.num_classes_per_level) != 3:
                raise ValueError(
                    f"Inference rule `{self.rule}` requires exactly 3 hierarchy "
                    "levels (coarse->middle->fine), got "
                    f"{len(self.num_classes_per_level)}."
                )
            self.projector = HierarchicalAffineProjector(
                num_classes_per_level=self.num_classes_per_level,
                taxonomy=dict(taxonomy),
            )
            self.projector.eval()

        self.level_subspace_masks: Optional[List[torch.Tensor]] = None
        if self.readout == SUBSPACE_NORM:
            topology = build_topology(
                num_classes_per_level=self.num_classes_per_level,
                taxonomy=dict(taxonomy),
                owner=self.owner,
            )
            self.level_subspace_masks = [
                mask.detach().to(device="cpu")
                for mask in topology["level_subspace_masks"]
            ]

    def node_coordinates(self, output: Mapping[str, Any]) -> List[torch.Tensor]:
        """Return the per-level coordinates the model's own readout consumes."""
        if self.model_name == "hiercos":
            node_logits = output.get("node_logits")
            if not isinstance(node_logits, torch.Tensor) or node_logits.ndim != 2:
                raise ValueError(
                    f"{self.owner} on native Hier-COS requires `node_logits` "
                    "with shape [B, N]."
                )
            # Node coordinates are ordered in contiguous coarse->fine level
            # blocks, so the same split the model uses for HCC applies here.
            return _validate_level_scores(
                list(torch.split(node_logits, self.num_classes_per_level, dim=1)),
                self.num_classes_per_level,
                owner=self.owner,
                source_key="node_logits",
            )

        return _validate_level_scores(
            output.get("logits_per_level"),
            self.num_classes_per_level,
            owner=self.owner,
            source_key="logits_per_level",
        )

    def scores_from_output(self, output: Mapping[str, Any]) -> List[torch.Tensor]:
        """Return per-level scores for this grid cell."""
        coordinates = self.node_coordinates(output)
        if self.projector is not None:
            coordinates = self.projector(coordinates)["projected_logits_per_level"]

        if self.readout == NODE_SCORE:
            if not self.magnitude:
                return list(coordinates)
            return [coordinate.abs() for coordinate in coordinates]

        assert self.level_subspace_masks is not None
        return subspace_norms(
            torch.cat(coordinates, dim=1),
            self.level_subspace_masks,
        )

    def transform_output(self, output: Mapping[str, Any]) -> Dict[str, Any]:
        """Build the minimal output consumed by the shared metric evaluator."""
        return {"logits_per_level": self.scores_from_output(output)}

    def describe(self) -> Dict[str, Any]:
        """Return a self-contained record of what this rule computed."""
        description: Dict[str, Any] = {
            "readout": self.readout,
            "hcc_projection": self.applies_hcc,
            "input": (
                "hiercos_fixed_layer_node_logits"
                if self.model_name == "hiercos"
                else "native_per_level_scores"
            ),
            "coordinate_evidence": "magnitude" if self.magnitude else "signed_value",
            "score": (
                "own_node_coordinate"
                if self.readout == NODE_SCORE
                else "l2_norm_over_ancestors_self_descendants_subspace"
            ),
            "prediction": "raw_score_argmax",
            "training": False,
        }
        if self.readout == SUBSPACE_NORM and self.model_name != "hiercos":
            description["fixed_frame"] = "identity"
        if self.applies_hcc:
            description["projection"] = "hcc_affine_least_squares_children_sum_to_parent"
            description["staging"] = "coarse_to_fine_with_detached_anchors"
            description["alpha"] = 1.0
        return description
