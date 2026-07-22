"""Canonical hierarchy construction shared by every dataset adapter."""

from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .types import DatasetLabelSpace

Sample = Mapping[str, Any]
Taxonomy = Mapping[str, Any]


def normalize_samples(
    samples: Sequence[Sample],
    *,
    depth: int,
    dataset_name: str,
    split: str,
) -> List[Dict[str, Any]]:
    """Validate adapter output and normalize labels and metadata in one pass."""
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("Dataset samples must be a sequence of mappings.")

    result: List[Dict[str, Any]] = []
    parents = {level: {} for level in range(1, depth)}
    for row, sample in enumerate(samples):
        context = f"dataset={dataset_name} split={split} row={row}"
        if not isinstance(sample, Mapping):
            raise TypeError(f"Malformed sample at {context}: expected a mapping.")
        if sample.get("image") is None:
            raise ValueError(f"Malformed sample at {context}: missing `image`.")

        raw_labels = sample.get("labels")
        if not isinstance(raw_labels, (list, tuple)):
            raise TypeError(f"Malformed sample at {context}: `labels` must be a list or tuple.")
        if len(raw_labels) != depth:
            raise ValueError(
                f"Malformed sample at {context}: expected {depth} labels, got {len(raw_labels)}."
            )
        labels = []
        for level, value in enumerate(raw_labels):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(
                    f"Malformed sample at {context}: label at level {level} "
                    f"must be an integer, got {value!r}."
                )
            labels.append(int(value))
            if labels[-1] < 0:
                raise ValueError(
                    f"Malformed sample at {context}: label at level {level} "
                    f"is negative ({labels[-1]})."
                )

        for level in range(1, depth):
            parent, child = labels[level - 1], labels[level]
            previous = parents[level].get(child)
            if previous is not None and previous != parent:
                raise ValueError(
                    f"Inconsistent taxonomy at {context}: child {child} at level {level} "
                    f"maps to both parent {previous} and parent {parent}."
                )
            parents[level][child] = parent

        metadata = sample.get("meta") or {}
        if not isinstance(metadata, Mapping):
            raise TypeError(f"Malformed sample at {context}: `meta` must be a mapping.")
        metadata = dict(metadata)
        if metadata.get("split") not in (None, split):
            raise ValueError(
                f"Malformed sample at {context}: metadata split={metadata['split']!r} "
                "does not match requested split."
            )
        metadata["split"] = split
        result.append({"image": sample["image"], "labels": labels, "meta": metadata})
    return result


def _parent_map(taxonomy: Taxonomy, level: int) -> Mapping[Any, Any]:
    parent_of = taxonomy.get("parent_of")
    if not isinstance(parent_of, Mapping):
        raise ValueError("Taxonomy must contain a `parent_of` mapping.")
    mapping = parent_of.get(level, parent_of.get(str(level)))
    if not isinstance(mapping, Mapping):
        raise ValueError(f"Taxonomy is missing transition mapping for level {level}.")
    return mapping


def _build_maps(
    samples: Sequence[Sample],
    taxonomy: Optional[Taxonomy],
    depth: int,
) -> Tuple[Dict[int, int], ...]:
    values = [
        {int(sample["labels"][level]) for sample in samples}
        for level in range(depth)
    ]
    if taxonomy is not None:
        parent_of = taxonomy.get("parent_of")
        if not isinstance(parent_of, Mapping):
            raise ValueError("Taxonomy must contain a `parent_of` mapping.")
        for level_key, mapping in parent_of.items():
            level = int(level_key)
            if level not in range(1, depth) or not isinstance(mapping, Mapping):
                raise ValueError(f"Invalid taxonomy transition at level {level}.")
            values[level].update(int(child) for child in mapping)
            values[level - 1].update(int(parent) for parent in mapping.values())
    if any(not level for level in values):
        raise ValueError("Every hierarchy level must contain at least one class.")
    return tuple(
        {raw: contiguous for contiguous, raw in enumerate(sorted(level))}
        for level in values
    )


def _remap_samples(
    samples: Sequence[Sample],
    maps: Sequence[Mapping[int, int]],
    *,
    dataset_name: str,
    split: str,
) -> List[Dict[str, Any]]:
    result = []
    for row, sample in enumerate(samples):
        labels = []
        for level, raw in enumerate(sample["labels"]):
            raw = int(raw)
            if raw not in maps[level]:
                raise ValueError(
                    f"Unknown raw label id {raw} at level {level} in "
                    f"dataset={dataset_name} split={split} row={row}. "
                    "Validation/test labels must use the training label space."
                )
            labels.append(int(maps[level][raw]))
        result.append(
            {"image": sample["image"], "labels": labels, "meta": dict(sample["meta"])}
        )
    return result


def _remap_taxonomy(
    taxonomy: Taxonomy,
    maps: Sequence[Mapping[int, int]],
    depth: int,
) -> Dict[str, Any]:
    remapped = {}
    for level in range(1, depth):
        mapping = _parent_map(taxonomy, level)
        child_map, parent_map = maps[level], maps[level - 1]
        fixed = {}
        for child, parent in mapping.items():
            child, parent = int(child), int(parent)
            if child not in child_map:
                raise ValueError(f"Taxonomy level {level} references unknown child id {child}.")
            if parent not in parent_map:
                raise ValueError(f"Taxonomy level {level} references unknown parent id {parent}.")
            fixed[int(child_map[child])] = int(parent_map[parent])
        remapped[level] = fixed
    return {"levels": taxonomy.get("levels"), "parent_of": remapped}


def infer_parent_of_from_samples(
    samples: Sequence[Sample],
    depth: int,
) -> Dict[int, Dict[int, int]]:
    result = {level: {} for level in range(1, depth)}
    for sample in samples:
        for level in range(1, depth):
            result[level][int(sample["labels"][level])] = int(sample["labels"][level - 1])
    return result


def taxonomy_from_parent_of(
    parent_of: Mapping[int, Mapping[int, int]],
    levels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "levels": list(levels) if levels is not None else None,
        "parent_of": {level: dict(mapping) for level, mapping in parent_of.items()},
    }


def _validate(
    samples: Sequence[Sample],
    taxonomy: Taxonomy,
    counts: Sequence[int],
    *,
    dataset_name: str,
    split: str,
) -> None:
    for level in range(1, len(counts)):
        mapping = _parent_map(taxonomy, level)
        expected = set(range(int(counts[level])))
        observed = {int(child) for child in mapping}
        if observed != expected:
            raise ValueError(
                f"Incomplete taxonomy at level {level}: "
                f"missing children {sorted(expected - observed)[:10]}, "
                f"extra children {sorted(observed - expected)[:10]}."
            )
        parent_count = int(counts[level - 1])
        if any(not 0 <= int(parent) < parent_count for parent in mapping.values()):
            raise ValueError(f"Taxonomy level {level} contains out-of-range parents.")
        for row, sample in enumerate(samples):
            child, parent = int(sample["labels"][level]), int(sample["labels"][level - 1])
            if int(mapping[child]) != parent:
                raise ValueError(
                    f"Taxonomy mismatch in dataset={dataset_name} split={split} row={row}: "
                    f"child {child} expects parent {mapping[child]}, got {parent}."
                )


def _check_split_taxonomy(
    split_taxonomy: Taxonomy,
    canonical_taxonomy: Taxonomy,
    depth: int,
    context: str,
) -> None:
    if (
        split_taxonomy.get("levels") is not None
        and canonical_taxonomy.get("levels") is not None
        and list(split_taxonomy["levels"]) != list(canonical_taxonomy["levels"])
    ):
        raise ValueError(context)
    for level in range(1, depth):
        canonical = _parent_map(canonical_taxonomy, level)
        for child, parent in _parent_map(split_taxonomy, level).items():
            if int(child) not in canonical or int(canonical[int(child)]) != int(parent):
                raise ValueError(context)


def build_training_label_space(
    samples: Sequence[Sample],
    taxonomy: Optional[Taxonomy],
    levels: Sequence[str],
    *,
    depth: int,
    dataset_name: str,
    split: str,
) -> Tuple[List[Dict[str, Any]], DatasetLabelSpace]:
    """Canonicalize training/authoritative metadata and create the shared contract."""
    maps = _build_maps(samples, taxonomy, depth)
    remapped = _remap_samples(samples, maps, dataset_name=dataset_name, split=split)
    canonical = (
        _remap_taxonomy(taxonomy, maps, depth)
        if taxonomy is not None
        else taxonomy_from_parent_of(infer_parent_of_from_samples(remapped, depth), levels)
    )
    counts = tuple(len(mapping) for mapping in maps)
    _validate(remapped, canonical, counts, dataset_name=dataset_name, split=split)
    return remapped, DatasetLabelSpace(maps, canonical, tuple(levels))


def apply_label_space(
    samples: Sequence[Sample],
    taxonomy: Optional[Taxonomy],
    label_space: DatasetLabelSpace,
    *,
    depth: int,
    dataset_name: str,
    split: str,
) -> List[Dict[str, Any]]:
    """Apply the training label space to validation/test and verify compatibility."""
    if label_space.depth != depth:
        raise ValueError(
            f"Label-space depth {label_space.depth} does not match dataset depth {depth}."
        )
    for level, mapping in enumerate(label_space.raw_to_contiguous):
        if set(int(value) for value in mapping.values()) != set(range(len(mapping))):
            raise ValueError(f"Label-space level {level} is not contiguous.")

    remapped = _remap_samples(
        samples,
        label_space.raw_to_contiguous,
        dataset_name=dataset_name,
        split=split,
    )
    if taxonomy is not None:
        context = (
            f"Taxonomy for dataset={dataset_name} split={split} is incompatible "
            "with the canonical training taxonomy."
        )
        _check_split_taxonomy(
            _remap_taxonomy(taxonomy, label_space.raw_to_contiguous, depth),
            label_space.taxonomy,
            depth,
            context,
        )
    _validate(
        remapped,
        label_space.taxonomy,
        label_space.num_classes_per_level,
        dataset_name=dataset_name,
        split=split,
    )
    return remapped
