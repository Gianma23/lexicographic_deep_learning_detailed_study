from typing import Any, Dict, List, Mapping, Sequence

import torch


def normalize_parent_of(taxonomy: Dict[str, Any], owner: str = "Hier-COS") -> Dict[int, Dict[int, int]]:
    if not taxonomy or "parent_of" not in taxonomy:
        raise ValueError(f"{owner} requires taxonomy with `parent_of` mappings.")

    parent_of_raw = taxonomy["parent_of"]
    if not isinstance(parent_of_raw, Mapping):
        raise ValueError("`taxonomy['parent_of']` must be a mapping of level -> (child -> parent).")

    normalized: Dict[int, Dict[int, int]] = {}
    for level_key, mapping in parent_of_raw.items():
        if not isinstance(mapping, Mapping):
            raise ValueError(f"`taxonomy['parent_of'][{level_key}]` must be a child -> parent mapping.")
        level = int(level_key)
        normalized[level] = {int(child): int(parent) for child, parent in mapping.items()}
    return normalized


def level_offsets(num_classes_per_level: Sequence[int]) -> List[int]:
    offsets = [0]
    for classes in num_classes_per_level[:-1]:
        offsets.append(offsets[-1] + int(classes))
    return offsets


def build_topology(
    num_classes_per_level: Sequence[int],
    taxonomy: Dict[str, Any],
    owner: str = "Hier-COS",
) -> Dict[str, Any]:
    num_classes = [int(v) for v in num_classes_per_level]
    if len(num_classes) < 2:
        raise ValueError(f"{owner} requires hierarchy depth >= 2.")
    if any(v <= 0 for v in num_classes):
        raise ValueError(f"All class counts must be > 0, got {num_classes}.")

    parent_of = normalize_parent_of(taxonomy, owner=owner)
    depth = len(num_classes)
    offsets = level_offsets(num_classes)
    total_nodes = int(sum(num_classes))

    parent_global = [-1 for _ in range(total_nodes)]
    children_global: List[List[int]] = [[] for _ in range(total_nodes)]

    for level in range(1, depth):
        mapping = parent_of.get(level)
        if mapping is None:
            raise ValueError(
                f"Missing taxonomy mapping for level transition {level - 1}->{level}. "
                f"Expected `taxonomy['parent_of'][{level}]`."
            )
        num_children = int(num_classes[level])
        num_parents = int(num_classes[level - 1])
        expected_children = set(range(num_children))
        if set(mapping.keys()) != expected_children:
            missing = sorted(expected_children - set(mapping.keys()))
            extra = sorted(set(mapping.keys()) - expected_children)
            raise ValueError(
                f"Invalid taxonomy mapping at level={level}. "
                f"Missing children: {missing[:10]}, extra children: {extra[:10]}."
            )

        for child_local in range(num_children):
            parent_local = int(mapping[child_local])
            if parent_local < 0 or parent_local >= num_parents:
                raise ValueError(
                    f"Invalid parent id={parent_local} for child={child_local} at level={level}; "
                    f"expected [0, {num_parents})."
                )
            child_global = int(offsets[level] + child_local)
            parent_global_idx = int(offsets[level - 1] + parent_local)
            parent_global[child_global] = parent_global_idx
            children_global[parent_global_idx].append(child_global)

    descendants: List[set] = [set() for _ in range(total_nodes)]
    for node in range(total_nodes - 1, -1, -1):
        for child in children_global[node]:
            descendants[node].add(int(child))
            descendants[node].update(descendants[child])

    ancestors: List[List[int]] = [[] for _ in range(total_nodes)]
    for node in range(total_nodes):
        parent = parent_global[node]
        node_ancestors: List[int] = []
        while parent >= 0:
            node_ancestors.append(int(parent))
            parent = parent_global[parent]
        ancestors[node] = node_ancestors

    level_node_ids: List[torch.Tensor] = []
    level_subspace_masks: List[torch.Tensor] = []
    for level, classes in enumerate(num_classes):
        node_ids = torch.arange(offsets[level], offsets[level] + classes, dtype=torch.long)
        level_node_ids.append(node_ids)

        mask = torch.zeros((classes, total_nodes), dtype=torch.bool)
        for local_id in range(classes):
            global_id = int(offsets[level] + local_id)
            indices = set(ancestors[global_id])
            indices.add(global_id)
            indices.update(descendants[global_id])
            mask[local_id, list(sorted(indices))] = True
        level_subspace_masks.append(mask)

    num_leaf = int(num_classes[-1])
    leaf_to_level_local = torch.full((num_leaf, depth), -1, dtype=torch.long)
    for leaf_local in range(num_leaf):
        cur = int(offsets[-1] + leaf_local)
        for level in range(depth - 1, -1, -1):
            leaf_to_level_local[leaf_local, level] = int(cur - offsets[level])
            if level > 0:
                parent = parent_global[cur]
                if parent < 0:
                    raise ValueError(
                        f"Leaf node id={leaf_local} has no valid ancestor at level={level - 1}."
                    )
                cur = parent

    level_weights = torch.arange(depth, 0, -1, dtype=torch.float32)
    level_weights = torch.exp(1.0 / level_weights)
    level_weights = level_weights / torch.norm(level_weights, p=2).clamp_min(1e-12)
    level_weights = level_weights.pow(2)

    return {
        "depth": depth,
        "total_nodes": total_nodes,
        "level_node_ids": level_node_ids,
        "level_subspace_masks": level_subspace_masks,
        "leaf_to_level_local": leaf_to_level_local,
        "node_prob_weights": level_weights,
    }
