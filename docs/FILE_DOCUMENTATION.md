# File Documentation

This document describes every project file in the repository (source, configs, and root metadata; excludes `.git/` internals).

## Root Files

### `.gitignore`
- Purpose: Standard Python-focused ignore rules plus tool-specific caches (e.g., Ruff, Cursor, virtual environments).
- Notes: Prevents generated artifacts and local environment files from being committed.

### `README.md`
- Purpose: Main project overview and quickstart for the unified H-CAST + HT-CapsNet training pipeline.
- Notes: Describes unified dataset/model APIs, taxonomy schema, and reproducibility options.

### `requirements.txt`
- Purpose: Python dependency list for training/evaluation.
- Notes: Includes PyTorch, torchvision, config libraries, and `timm`; `dgl` is optional.

## Documentation

### `docs/FILE_DOCUMENTATION.md`
- Purpose: Central file-by-file project reference.
- Notes: Documents purpose and key APIs for every repository file (excluding `.git/` internals).

## Configurations

### `configs/hcast.yaml`
- Purpose: Default training config for the H-CAST model path.
- Key settings: H-CAST variant selection, BREEDS dataset options, optimization/loss/runtime knobs.

### `configs/capsnet.yaml`
- Purpose: Default training config for the HT-CapsNet model path.
- Key settings: Capsule routing parameters, taxonomy-mask controls, CUB dataset options, optimization/loss/runtime knobs.

## Dataset Package (`datasets/`)

### `datasets/__init__.py`
- Purpose: Dataset factory and dataloader assembly.
- Key APIs: `build_transforms`, `_collate_fn`, `build_dataloader`.
- Notes: Registers dataset adapters and returns `(loader, num_classes_per_level, taxonomy)`.

### `datasets/base.py`
- Purpose: Abstract base dataset with shared hierarchy/taxonomy utilities.
- Key APIs: `BaseHierDataset`, `infer_parent_of_from_samples`, `taxonomy_from_parent_of`.
- Notes: Handles annotation loading, synthetic fallback, contiguous label remap, and taxonomy inference/remap.

### `datasets/breeds.py`
- Purpose: BREEDS adapter.
- Key API: `BreedsDataset`.
- Notes: Reads optional unified JSON or H-CAST-style BREEDS txt files; provides coarse-to-species taxonomy.

### `datasets/cub.py`
- Purpose: CUB-200 adapter.
- Key API: `CUBDataset`.
- Notes: Reads optional JSON or class-folder structure; maps species to order/family using `cub_tree.py`.

### `datasets/cub_tree.py`
- Purpose: Static hierarchy table for CUB classes.
- Key data: `TREES` (species -> order/family mapping).
- Notes: Used by `CUBDataset` to construct hierarchical labels and taxonomy.

### `datasets/aircraft.py`
- Purpose: FGVC-Aircraft adapter.
- Key API: `AircraftDataset`.
- Notes: Parses official split txt files and `Air.csv`, then maps variant/family/manufacturer via `aircraft_tree.py`.

### `datasets/aircraft_tree.py`
- Purpose: Static hierarchy table for Aircraft classes.
- Key data: `TREES` (variant -> family/manufacturer mapping).
- Notes: Used by `AircraftDataset`.

### `datasets/inat.py`
- Purpose: iNaturalist adapter (iNat18 + iNat21-mini style).
- Key API: `INatDataset`.
- Notes: Supports JSON annotations, txt annotations, iNat18 tree-based decoding, and inferred taxonomy.

## Model Package (`models/`)

### `models/__init__.py`
- Purpose: Unified model/loss dispatch layer.
- Key APIs: `build_model`, `compute_loss`.
- Notes: Routes runtime to `hcast` or `ht_capsnet` implementations.

### `models/hd_capsnet/__init__.py`
- Purpose: Placeholder module for potential HD-CapsNet integration.
- Notes: Currently no runtime model implementation.

## H-CAST Modules (`models/hcast/`)

### `models/hcast/__init__.py`
- Purpose: Public H-CAST package exports.
- Key exports: `build_model`, `compute_loss`.

### `models/hcast/factory.py`
- Purpose: H-CAST model builder.
- Key API: `build_model`.
- Notes: Instantiates `HCASTModel` with variant/fallback settings from config.

### `models/hcast/model.py`
- Purpose: Runtime H-CAST wrapper plus lightweight fallback model.
- Key classes: `HCASTModel`, `HCASTLite`.
- Notes: Loads upstream CAST variants when available; falls back to ResNet18 heads if dependencies fail.

### `models/hcast/losses.py`
- Purpose: H-CAST loss composition.
- Key API: `compute_loss`.
- Notes: Combines per-level CE with hierarchy-violation loss, tree-path KL, and optional global KL.

### `models/hcast/internal/__init__.py`
- Purpose: Internal export point for upstream-style CAST constructors.
- Key exports: `cast_small`, `cast_small_deep`, `cast_base`, `cast_base_deep`.

### `models/hcast/internal/utils.py`
- Purpose: Low-level tensor/segmentation helper utilities.
- Key APIs: `segment_mean`, `segment_mean_nd`, `resize_labels`, `pca`, `one_hot`, `normalize_embedding`.
- Notes: Supports segment-wise pooling and feature manipulation used by H-CAST internals.

### `models/hcast/internal/modules.py`
- Purpose: Shared building blocks for H-CAST internals.
- Key classes: `Pooling`, `ConvStem`, `_BatchNorm1d`, `BlockFusion`.
- Notes: Implements pooling wrappers, convolutional stem patch embedding, and multi-block feature fusion.

### `models/hcast/internal/graph_pool.py`
- Purpose: Graph pooling implementation used inside CAST.
- Key classes/functions: `Attention`, `Block`, `GraphPooling`, `valid_mean`.
- Notes: Uses DGL farthest-point sampling when available, with deterministic fallback sampling otherwise.

### `models/hcast/internal/cast_deit_hier.py`
- Purpose: Core CAST architecture adapted from upstream DeiT/ViT code.
- Key class/functions: `CAST`, `cast_small`, `cast_small_deep`, `cast_base`, `cast_base_deep`.
- Notes: Implements staged transformer blocks with graph pooling and multi-level hierarchical heads.

## HT-CapsNet Modules (`models/ht_capsnet/`)

### `models/ht_capsnet/__init__.py`
- Purpose: Public HT-CapsNet package exports.
- Key exports: `build_model`, `compute_loss`.

### `models/ht_capsnet/factory.py`
- Purpose: HT-CapsNet model builder.
- Key API: `build_model`.
- Notes: Reads capsule/routing/taxonomy attention hyperparameters from config and constructs `HTCapsNet`.

### `models/ht_capsnet/model.py`
- Purpose: HT-CapsNet architecture in PyTorch.
- Key classes: `_ConvBackbone`, `HTCapsNet`.
- Notes: Builds primary capsules from CNN features, performs taxonomy-guided routing per level, applies cross-capsule attention.

### `models/ht_capsnet/routing.py`
- Purpose: Taxonomy-aware dynamic routing primitives.
- Key APIs: `squash`, `taxonomy_mask_from_matrix`, `taxonomy_guided_routing`.
- Notes: Computes routing couplings with optional parent-child mask constraints.

### `models/ht_capsnet/losses.py`
- Purpose: HT-CapsNet loss composition.
- Key API: `compute_loss`.
- Notes: Weighted multi-level margin loss plus hierarchy-consistency penalty.

## Training Package (`train/`)

### `train/__init__.py`
- Purpose: Public training engine exports.
- Key exports: `train_one_epoch`, `evaluate`.

### `train/train.py`
- Purpose: Main CLI training entrypoint.
- Key APIs: `parse_args`, `main`, config parsing helpers.
- Notes: Loads config/overrides, builds dataloaders+model, runs train/val/test loops, and saves checkpoints.

### `train/engine.py`
- Purpose: Core epoch loops.
- Key APIs: `train_one_epoch`, `evaluate`.
- Notes: Handles mixed precision, forward/loss/backprop, and batch metric aggregation.

### `train/eval.py`
- Purpose: Batch-level evaluation helpers and metric formatting.
- Key APIs: `evaluate_batch`, `pretty_metrics`.
- Notes: Adds per-level accuracy, path accuracy, and taxonomy-aware consistency metrics.

### `train/metrics.py`
- Purpose: Metric definitions and reduction helpers.
- Key APIs: `per_level_top1`, `full_path_accuracy`, `inconsistency_rate`, `tice_like_score`, `merge_metric_batches`.
- Notes: Supports hierarchy-consistency checks when taxonomy exists.

### `train/utils.py`
- Purpose: Reproducibility, optimizer/scheduler creation, checkpoint I/O, resume logic.
- Key APIs: `seed_everything`, `build_optimizer`, `build_scheduler`, `save_checkpoint`, `resume_if_available`, `metric_for_best`.
- Notes: Encapsulates reusable training utilities used by `train.py`.
