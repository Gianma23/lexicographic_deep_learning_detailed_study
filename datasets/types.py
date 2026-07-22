"""Small public contracts shared by dataset adapters and training."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class DatasetLabelSpace:
    """Canonical mapping and taxonomy built once and reused by every split."""

    raw_to_contiguous: Tuple[Mapping[int, int], ...]
    taxonomy: Dict[str, Any]
    levels: Tuple[str, ...]

    @property
    def depth(self) -> int:
        return len(self.raw_to_contiguous)

    @property
    def raw_ids_per_level(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(
            tuple(raw for raw, _ in sorted(mapping.items(), key=lambda item: item[1]))
            for mapping in self.raw_to_contiguous
        )

    @property
    def num_classes_per_level(self) -> Tuple[int, ...]:
        return tuple(len(mapping) for mapping in self.raw_to_contiguous)


@dataclass(frozen=True)
class DatasetMetadata:
    label_space: DatasetLabelSpace

    @property
    def num_classes_per_level(self) -> List[int]:
        return list(self.label_space.num_classes_per_level)

    @property
    def taxonomy(self) -> Dict[str, Any]:
        return deepcopy(self.label_space.taxonomy)

    @property
    def levels(self) -> List[str]:
        return list(self.label_space.levels)
