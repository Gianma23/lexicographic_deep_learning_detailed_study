# File Documentation

This document gives a high-level map of the repository by folder. For Python modules, each section describes files plus top-level functions/classes (including important class methods), without going into low-level implementation details.

Scope notes:
- Includes source, configs, docs, and runtime artifact folders under the project root.
- Excludes `.git/` internals and `__pycache__/` bytecode files.

## 1. Root Folder (`/`)

### 1.1. `.gitignore`
- Purpose: Ignore Python caches, environments, editor state, and training artifacts.

### 1.2. `README.md`
- Purpose: Project overview, setup commands, unified CLI usage, and API conventions for datasets/models/metrics.

### 1.3. `requirements.txt`
- Purpose: Core dependency list used to run training and evaluation.

### 1.4. `TODO.md`
- Purpose: Local planning notes and pending tasks for the repository.

## 2. Config Folder (`configs/`)

### 2.1. `configs/hcast.yaml`
- Purpose: Default configuration for H-CAST training/evaluation runs.

### 2.2. `configs/capsnet.yaml`
- Purpose: Default configuration for HT-CapsNet training/evaluation runs.


### 2.3. `configs/cifar100.yaml`
- Purpose: Dataset-focused run preset for CIFAR-100 hierarchy experiments.

### 2.4. `configs/cub200.yaml`
- Purpose: Dataset-focused run preset for CUB-200 hierarchy experiments.

### 2.5. `configs/aircraft.yaml`
- Purpose: Dataset-focused run preset for FGVC-Aircraft hierarchy experiments.

### 2.6. `configs/inat21mini.yaml`
- Purpose: Dataset-focused run preset for iNat21-mini hierarchy experiments.

## 3. Data Folder (`data/`)

### 3.1. `data/` (directory)
- Purpose: Local raw dataset storage used by dataset adapters.
- Notes: Contains external dataset payloads; contents can vary by machine and are not part of the code API.

## 4. Docs Folder (`docs/`)

### 4.1. `docs/FILE_DOCUMENTATION.md`
- Purpose: This full repository reference document.

## 5. Dataset Package (`datasets/`)

### 5.1. `datasets/__init__.py`
- Purpose: Dataset registry and dataloader assembly entrypoint.
- Functions:
  - `build_transforms`: Creates split-aware image transforms from config.
  - `_collate_fn`: Batch collation helper for `(image, labels, meta)` samples.
  - `build_dataloader`: Instantiates the selected dataset and returns loader plus hierarchy metadata.

### 5.2. `datasets/base.py`
- Purpose: Shared hierarchical dataset base class and common split/taxonomy utilities.
- Functions:
  - `infer_parent_of_from_samples`: Infers parent-child mapping from labeled samples.
  - `taxonomy_from_parent_of`: Converts parent mapping into normalized taxonomy payload.
  - `resolve_split_seed`: Resolves deterministic split seed from config.
  - `resolve_val_split_ratio`: Reads/validates validation split ratio from config.
  - `stratified_train_val_indices`: Builds deterministic stratified index split.
  - `split_train_val_samples`: Applies train/val partitioning logic to sample lists.
- Class `BaseHierDataset`: Base interface for hierarchical dataset adapters.
  - `__init__`: Loads split data, normalizes labels, and prepares taxonomy/metadata.
  - `load_samples`: Abstract hook implemented by concrete datasets.
  - `load_taxonomy`: Optional taxonomy override hook.
  - `_infer_taxonomy`: Internal fallback taxonomy inference path.
  - `_remap_taxonomy_ids`: Keeps taxonomy consistent after label remapping.
  - `_remap_labels_to_contiguous`: Enforces contiguous ids per hierarchy level.
  - `_compute_num_classes_per_level`: Computes class count per level.
  - `_build_synthetic_samples`: Generates synthetic fallback samples for smoke tests.
  - `__len__`: Returns dataset size.
  - `__getitem__`: Returns one transformed sample with multi-level labels and meta.
  - `_load_image`: Centralized image loading helper.
  - `_read_json_samples`: Reads optional JSON annotation format.
  - `_annotation_file_for_split`: Resolves split-specific annotation file path.

### 5.3. `datasets/breeds.py`
- Purpose: BREEDS adapter built on top of `BaseHierDataset`.
- Class `BreedsDataset`:
  - `load_samples`: Loads split annotations from supported BREEDS formats.
  - `load_taxonomy`: Returns coarse-to-fine taxonomy for consistency checks/metrics.

### 5.4. `datasets/cifar100.py`
- Purpose: CIFAR-100 adapter with canonical coarse/fine hierarchy.
- Class `CIFAR100Dataset`:
  - `__init__`: Extends base init with CIFAR-specific setup options.
  - `_label_path`: Resolves optional label metadata path.
  - `_default_parent_of`: Provides built-in coarse-to-fine class mapping.
  - `load_samples`: Reads torchvision samples and builds hierarchical labels.
  - `_load_image`: Handles CIFAR image conversion to expected format.
  - `load_taxonomy`: Returns taxonomy object for coarse/fine structure.

### 5.5. `datasets/cub.py`
- Purpose: CUB-200 adapter supporting multiple on-disk formats.
- Class `CUBDataset`:
  - `load_samples`: Main split loader with format auto-detection.
  - `_load_from_split_folders`: Reads `train/test` folder layouts.
  - `_read_folder_classes`: Builds class mapping from folder names.
  - `_species_from_class_name`: Normalizes species identifiers from raw class names.
  - `_load_from_official_files`: Reads official CUB metadata files.
  - `_read_int_str_map`: Utility parser for int-to-string metadata files.
  - `_read_int_int_map`: Utility parser for int-to-int metadata files.
  - `load_taxonomy`: Builds hierarchy using static CUB mapping table.

### 5.6. `datasets/cub_tree.py`
- Purpose: Static species-to-family/order mapping table used by `CUBDataset`.

### 5.7. `datasets/aircraft.py`
- Purpose: FGVC-Aircraft adapter with official split support and fallbacks.
- Class `AircraftDataset`:
  - `load_samples`: Loads split lists and constructs hierarchical targets.
  - `_resolve_data_root`: Locates expected Aircraft `data/` directory.
  - `_build_variant_to_id`: Creates stable variant label indexing.
  - `_read_variant_file`: Parses official split text files.
  - `load_taxonomy`: Builds manufacturer/family/variant taxonomy.

### 5.8. `datasets/aircraft_tree.py`
- Purpose: Static variant-to-family/manufacturer mapping table used by `AircraftDataset`.

### 5.9. `datasets/inat.py`
- Purpose: iNaturalist adapter supporting iNat18 and iNat21-mini style formats.
- Class `INatDataset`:
  - `load_samples`: Main routing entry that selects supported annotation format.
  - `_load_inat18_split`: Loads iNat18 JSON-based split data.
  - `_load_inat21_split`: Loads iNat21 text-list split data.
  - `_find_inat21_split_file`: Searches for available split file candidates.
  - `_find_existing`: Returns first existing path from candidate list.
  - `_read_txt_annotations`: Parses simple text annotation lines.
  - `_resolve_image_path`: Resolves absolute path for each image entry.
  - `_read_inat18`: Reads iNat18 JSON structures.
  - `load_taxonomy`: Builds normalized taxonomy payload for hierarchy metrics.

## 6. Model Package (`models/`)

### 6.1. `models/__init__.py`
- Purpose: Unified model/loss dispatcher across model families.
- Functions:
  - `build_model`: Builds selected model backend (`hcast` or `ht_capsnet`).
  - `compute_loss`: Dispatches loss computation to the selected backend.

## 7. H-CAST Package (`models/hcast/`)

### 7.1. `models/hcast/__init__.py`
- Purpose: Public exports for H-CAST model construction and loss API.

### 7.2. `models/hcast/factory.py`
- Purpose: H-CAST-specific model construction glue.
- Functions:
  - `build_model`: Builds `HCASTModel` using model config and hierarchy metadata.

### 7.3. `models/hcast/model.py`
- Purpose: Runtime wrapper for full H-CAST and lightweight fallback path.
- Class `HCASTLite`:
  - `__init__`: Creates a simple fallback backbone + per-level heads.
  - `forward`: Produces `logits_per_level` in unified output format.
- Class `HCASTModel`:
  - `__init__`: Builds upstream CAST variant when dependencies are available.
  - `_build_default_segments`: Generates default token segmentation metadata.
  - `forward`: Runs selected architecture and normalizes outputs to unified schema.

### 7.4. `models/hcast/losses.py`
- Purpose: Composite loss for hierarchical H-CAST training.
- Functions:
  - `_normalize_parent_of`: Sanitizes taxonomy parent map into a consistent shape.
  - `_project_children_to_parent`: Aggregates child probabilities into parent space.
  - `_hierarchy_violation_loss`: Penalizes invalid parent-child confidence patterns.
  - `_tree_path_kl_loss`: Encourages level-wise path consistency through KL terms.
  - `_global_kl_loss`: Adds optional global distribution regularization.
  - `compute_loss`: Combines task and hierarchy-aware terms into final training loss.

## 8. H-CAST Internal Modules (`models/hcast/internal/`)

### 8.1. `models/hcast/internal/__init__.py`
- Purpose: Internal export surface for CAST constructor variants.

### 8.2. `models/hcast/internal/cast_deit_hier.py`
- Purpose: Core CAST transformer architecture and size presets.
- Class `CAST`:
  - `__init__`: Configures hierarchical transformer stages and heads.
  - `_block_operations`: Shared stage block execution routine.
  - `forward_features`: Produces hierarchical intermediate features.
  - `forward`: Produces multi-level logits in CAST format.
- Functions:
  - `cast_small`: Small preset constructor.
  - `cast_small_deep`: Small-deep preset constructor.
  - `cast_base`: Base preset constructor.
  - `cast_base_deep`: Base-deep preset constructor.

### 8.3. `models/hcast/internal/graph_pool.py`
- Purpose: Graph attention/pooling blocks used in hierarchical token processing.
- Class `Attention`:
  - `__init__`: Builds attention projection layers.
  - `forward`: Applies attention over token/node features.
- Class `Block`:
  - `__init__`: Composes attention and MLP sublayers.
  - `forward`: Applies one transformer-style block step.
- Class `GraphPooling`:
  - `__init__`: Configures graph-based downsampling strategy.
  - `_fill_with_mean`: Fills missing pooled positions with averaged features.
  - `forward`: Pools graph tokens for next hierarchy stage.
- Function:
  - `valid_mean`: Computes masked mean used by pooling logic.

### 8.4. `models/hcast/internal/modules.py`
- Purpose: Reusable neural modules used by CAST internals.
- Class `Pooling`:
  - `__init__`: Configures pooled token readout behavior.
  - `forward`: Runs selected pooling strategy.
- Class `ConvStem`:
  - `__init__`: Builds convolutional patch/token stem.
  - `forward`: Converts image input to token embedding grid.
- Class `_BatchNorm1d`:
  - `__init__`: Wraps batch norm behavior for expected tensor layout.
  - `forward`: Applies normalization with layout handling.
- Class `BlockFusion`:
  - `__init__`: Sets up multi-stage feature fusion layers.
  - `_make_proj_block`: Builds projection sub-blocks.
  - `_unpool`: Restores pooled features for fusion alignment.
  - `_proj_block_operations`: Shared projection/fusion step routine.
  - `forward`: Fuses hierarchical block outputs before heads.

### 8.5. `models/hcast/internal/utils.py`
- Purpose: Tensor and embedding utility helpers used by CAST internals.
- Functions:
  - `resize_labels`: Resizes label maps to target resolution.
  - `calculate_principal_components`: Computes PCA basis vectors.
  - `pca`: Projects embeddings with computed/provided PCA basis.
  - `one_hot`: Builds one-hot encoded label tensors.
  - `normalize_embedding`: Applies feature normalization for stability.
  - `segment_mean`: Computes mean per segment id.
  - `segment_mean_nd`: Segment mean helper for higher-dimensional tensors.

## 9. HT-CapsNet Package (`models/ht_capsnet/`)

### 9.1. `models/ht_capsnet/__init__.py`
- Purpose: Public exports for HT-CapsNet model construction and loss API.

### 9.2. `models/ht_capsnet/factory.py`
- Purpose: HT-CapsNet-specific model constructor.
- Functions:
  - `build_model`: Builds `HTCapsNet` using routing/capsule config parameters.

### 9.3. `models/ht_capsnet/model.py`
- Purpose: PyTorch implementation of taxonomy-guided capsule model.
- Class `_ConvBackbone`:
  - `__init__`: Builds CNN feature extractor for primary capsules.
  - `forward`: Produces feature maps for capsule projection.
- Class `HTCapsNet`:
  - `__init__`: Configures capsule hierarchy, routing, and classifier heads.
  - `_normalize_parent_of`: Standardizes taxonomy map for routing constraints.
  - `_taxonomy_matrix`: Builds matrix view of parent-child relations.
  - `_build_primary_caps`: Converts backbone features into primary capsules.
  - `forward`: Runs routing per level and returns unified logits payload.

### 9.4. `models/ht_capsnet/routing.py`
- Purpose: Taxonomy-constrained routing primitives.
- Functions:
  - `squash`: Capsule squashing nonlinearity.
  - `taxonomy_mask_from_matrix`: Converts taxonomy matrix into routing mask.
  - `taxonomy_guided_routing`: Runs iterative dynamic routing with optional masks.

### 9.5. `models/ht_capsnet/losses.py`
- Purpose: Composite HT-CapsNet training loss.
- Functions:
  - `_margin_loss`: Computes capsule-style per-level margin loss.
  - `_normalize_parent_of`: Normalizes taxonomy map for consistency penalty.
  - `_hier_consistency_penalty`: Penalizes parent-child prediction conflicts.
  - `_level_weights`: Resolves level-wise loss weighting from config.
  - `compute_loss`: Combines per-level margin loss and hierarchy penalty terms.

## 10. Training Package (`train/`)

### 10.1. `train/__init__.py`
- Purpose: Public exports for epoch-level training/evaluation loops.

### 10.2. `train/train.py`
- Purpose: Main CLI entrypoint and config orchestration.
- Class `AttrDict`:
  - `__getattr__`: Attribute-style read access for config dictionaries.
  - `__setattr__`: Attribute-style write access for config dictionaries.
- Functions:
  - `_to_attr`: Recursively converts dictionaries to `AttrDict`.
  - `_coerce_scalar`: Converts CLI override strings to scalar Python types.
  - `_apply_dotlist`: Applies dot-list overrides into nested config objects.
  - `_load_config`: Loads YAML config and merges CLI overrides.
  - `_parse_args`: Defines/reads command-line arguments.
  - `main`: Full training workflow (load config, build components, run epochs, checkpoint, evaluate).

### 10.3. `train/engine.py`
- Purpose: Epoch-level optimization and evaluation loops.
- Functions:
  - `train_one_epoch`: Executes one training epoch with optimizer/scaler integration.
  - `evaluate`: Runs model on validation/test loader and aggregates metrics.

### 10.4. `train/eval.py`
- Purpose: Batch-level evaluation metric assembly and report formatting.
- Functions:
  - `evaluate_batch`: Computes metrics for one model output batch.
  - `pretty_metrics`: Formats metric dictionary for human-readable logging.

### 10.5. `train/metrics.py`
- Purpose: Shared metric calculations for hierarchical classification.
- Functions:
  - `per_level_top1`: Computes top-1 accuracy per hierarchy level.
  - `full_path_accuracy`: Computes exact path (all levels) accuracy.
  - `inconsistency_rate`: Measures taxonomy-violating prediction frequency.
  - `tice_like_score`: Computes taxonomy-informed confidence/consistency score.
  - `merge_metric_batches`: Merges batch metric dictionaries into epoch summary.

### 10.6. `train/utils.py`
- Purpose: Reproducibility, optimization setup, and checkpoint lifecycle helpers.
- Functions:
  - `seed_everything`: Sets global random seeds and deterministic runtime switches.
  - `build_optimizer`: Creates optimizer from config.
  - `build_scheduler`: Creates LR scheduler from config.
  - `save_checkpoint`: Saves model/optimizer/scheduler/scaler/training state.
  - `resume_if_available`: Restores training state from checkpoint when requested.
  - `metric_for_best`: Selects scalar metric used to track/save best checkpoint.

## 11. Outputs Folder (`outputs/`)

### 11.1. `outputs/` (directory)
- Purpose: Runtime artifacts from training runs.
- Typical contents:
  - Per-run folders (for example `cifar100/`, `smoke_capsnet/`).
  - `latest.pt`: Most recent checkpoint snapshot.
  - `best.pt`: Best-scoring checkpoint according to configured selection metric.


