"""Deterministic train/validation split helpers."""

import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def resolve_split_seed(cfg: Any) -> int:
    """Resolve the seed used for deterministic train/validation splitting."""
    split_seed = cfg.dataset.get("split_seed", None)
    if split_seed is not None:
        return int(split_seed)

    train_cfg = cfg.get("train", None) if hasattr(cfg, "get") else getattr(cfg, "train", None)
    if train_cfg is not None:
        seed = train_cfg.get("seed", None) if hasattr(train_cfg, "get") else getattr(train_cfg, "seed", None)
        if seed is not None:
            return int(seed)
    return 42


def resolve_val_split_ratio(cfg: Any) -> float:
    """Resolve the validation ratio; strict configs are validated before this point."""
    return float(cfg.dataset.get("val_split_ratio", 0.1))


def stratified_train_val_indices(
    stratify_labels: List[int],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Build deterministic stratified train/validation indices from labels."""
    if not stratify_labels:
        return [], []
    if val_ratio <= 0.0:
        return list(range(len(stratify_labels))), []

    per_class: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(stratify_labels):
        per_class[int(label)].append(index)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for class_id, indices in sorted(per_class.items()):
        shuffled = list(indices)
        random.Random(seed * 1_000_003 + class_id).shuffle(shuffled)
        count = len(shuffled)
        if count <= 1:
            validation_count = 0
        else:
            validation_count = min(max(int(round(count * val_ratio)), 1), count - 1)
        val_indices.extend(shuffled[:validation_count])
        train_indices.extend(shuffled[validation_count:])

    if not val_indices and len(stratify_labels) > 1:
        shuffled = list(range(len(stratify_labels)))
        random.Random(seed).shuffle(shuffled)
        val_indices = [shuffled[0]]
        train_indices = shuffled[1:]

    return sorted(train_indices), sorted(val_indices)


def split_train_val_samples(
    samples: List[Dict[str, Any]],
    split: str,
    cfg: Any,
    stratify_level: int = -1,
) -> List[Dict[str, Any]]:
    """Return split-specific samples using a stratified train/val partition."""
    if split not in {"train", "val"} or not samples:
        return samples

    val_ratio = resolve_val_split_ratio(cfg)
    if val_ratio <= 0.0:
        return samples if split == "train" else []

    stratify_labels: List[int] = []
    for sample in samples:
        labels = [int(value) for value in sample.get("labels", [])]
        if not labels:
            stratify_labels.append(0)
            continue
        level = stratify_level if stratify_level >= 0 else len(labels) + stratify_level
        if level < 0 or level >= len(labels):
            level = len(labels) - 1
        stratify_labels.append(labels[level])

    train_indices, val_indices = stratified_train_val_indices(
        stratify_labels,
        val_ratio=val_ratio,
        seed=resolve_split_seed(cfg),
    )
    chosen = train_indices if split == "train" else val_indices
    return [samples[index] for index in chosen]
