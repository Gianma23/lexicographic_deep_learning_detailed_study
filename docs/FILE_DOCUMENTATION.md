# Repository Map

This document is a concise map of the tracked repository files. It describes what each area is responsible for today and is intended to stay aligned with the current code, not to duplicate every internal method.

## Root

- `README.md`: Project overview, setup, usage, datasets, metrics, and outputs.
- `requirements.txt`: Python dependencies other than the user-installed `torch` and `torchvision`.
- `TODO.md`: Local project notes.

## Configs

All experiment configs follow the same top-level sections: `model`, `dataset`, `dataloader`, `train`, `optim`, `scheduler`, and `runtime`.

### H-CAST presets

- `configs/hcast/hcast_cifar100.yaml`: Base H-CAST on CIFAR-100.
- `configs/hcast/hcast_cub200.yaml`: Base H-CAST on CUB-200-2011.
- `configs/hcast/hcast_aircraft.yaml`: Base H-CAST on FGVC-Aircraft.
- `configs/hcast/hcast_inat21mini.yaml`: Base H-CAST on iNat21-style data.
- `configs/hcast/hcast_hcc_cifar100.yaml`: H-CAST with the hard hierarchy projection block on CIFAR-100.
- `configs/hcast/hcast_hcc_cub200.yaml`: Same HCC variant on CUB.
- `configs/hcast/hcast_hcc_aircraft.yaml`: Same HCC variant on Aircraft.

### LH-DNN presets

- `configs/lhdnn/lhdnn_cifar100.yaml`
- `configs/lhdnn/lhdnn_cub200.yaml`
- `configs/lhdnn/lhdnn_aircraft.yaml`

These are paper-aligned LH-DNN presets for the supported datasets.

### HT-CapsNet presets

- `configs/capsnet/capsnet_cifar100.yaml`
- `configs/capsnet/capsnet_cub200.yaml`
- `configs/capsnet/capsnet_aircraft.yaml`

These presets configure the PyTorch HT-CapsNet port and its routing/loss parameters.

### HRN presets

- `configs/hrn/hrn_cifar100_parity.yaml`
- `configs/hrn/hrn_cub200_parity.yaml`
- `configs/hrn/hrn_aircraft_parity.yaml`

The HRN family supports exactly three hierarchy levels.
The CUB-200 and Aircraft presets mirror upstream HRN preprocessing and optimization: ImageNet-pretrained ResNet-50, 448 crops after 550x550 resize, `[0.5, 0.5, 0.5]` normalization, cosine LR, and a 0.1x LR group for the ResNet trunk. The CIFAR-100 preset is a local extrapolation because upstream HRN does not include CIFAR-100.

### Hier-COS presets

- `configs/hiercos/hiercos_cifar100_parity.yaml`
- `configs/hiercos/hiercos_cub200_parity.yaml`
- `configs/hiercos/hiercos_aircraft_parity.yaml`

These presets use the single Hier-COS implementation with a fixed orthonormal frame, taxonomy-driven subspace scores, configurable `model.loss` (`kl_reg` default, preferred lex mode `per_level_kl_reg`, or local CE ablations `per_level_ce` / `per_level_abs_node_ce`), shared per-level weighting through `model.weight_mode` (KL target-path and CE), and paper-style SGD/cosine settings for CIFAR-100 and Aircraft, plus a pragmatic CUB extrapolation. CIFAR keeps this repo hierarchy format (not the paper 5-level protocol). CIFAR uses HAFrame WideResNet from scratch; Aircraft and CUB use an ImageNet-pretrained ResNet-50.

### Templates

- `configs/templates/dataset_template.yaml`: Commented template for dataset and transform settings.
- `configs/templates/hcast_template.yaml`: Commented template for H-CAST-specific model options.
- `configs/templates/hcc_template.yaml`: Standalone optional HCC projection block template.
- `configs/templates/ht_capsnet_template.yaml`: Commented template for HT-CapsNet-specific model options.
- `configs/templates/hrn_template.yaml`: Commented template for HRN-specific model options.
- `configs/templates/hiercos_template.yaml`: Commented template for Hier-COS-specific model options.
- `configs/templates/training_template.yaml`: Shared template for train/optim/scheduler/runtime sections.

## Datasets

- `datasets/__init__.py`: Strict dataset-id registry (`cifar-100`, `cub-200-2011`, `fgvc-aircraft`, `inat21-mini`), transforms, collate function, and dataloader builder (including optional `crop_bottom_pixels` transform and `drop_last_eval` dataloader flag).
- `datasets/base.py`: Shared hierarchical dataset base class, train/val splitting, normalized JSON annotations, label remapping, and taxonomy inference.
- `datasets/cifar100.py`: CIFAR-100 adapter. Supports 2-level `coarse -> fine` and 3-level `super -> coarse -> fine` hierarchies.
- `datasets/cub.py`: CUB-200-2011 adapter for folder-based and official metadata layouts.
- `datasets/cub_tree.py`: Static order/family/species mapping used by the CUB adapter.
- `datasets/aircraft.py`: FGVC-Aircraft adapter for official variant split files and related fallbacks.
- `datasets/aircraft_tree.py`: Static manufacturer/family/variant mapping used by the Aircraft adapter.
- `datasets/inat.py`: iNat21-style adapter for JSON and JSON-in-tar annotations, with repo-specific split fallback logic.

## Models

- `models/__init__.py`: Unified `build_model` and `compute_loss` dispatcher. Supported model ids are `hcast`, `lhdnn`, `ht_capsnet`, `hrn`, and `hiercos`.

### H-CAST

- `models/hcast/__init__.py`: Public H-CAST exports.
- `models/hcast/factory.py`: Converts config sections into `HCASTModel` constructor arguments.
- `models/hcast/model.py`: Main H-CAST wrapper around the vendored CAST backbone, segmentation, and optional HCC projection.
- `models/hcast/losses.py`: H-CAST loss combining per-level task loss, optional global KL regularization, and HCC-aware probability handling.
- `models/hcast/segments.py`: OpenCV SEEDS-based segment generation helpers.
- `models/hcast/hard_hierarchy.py`: Hierarchical Constraint Cascade (HCC) affine projector used for hard hierarchy constraints.

#### Vendored H-CAST internals

- `models/hcast/internal/__init__.py`: Internal export surface.
- `models/hcast/internal/cast_deit_hier.py`: CAST architecture and size variants.
- `models/hcast/internal/graph_pool.py`: Graph pooling and attention blocks.
- `models/hcast/internal/modules.py`: Shared low-level modules used by CAST.
- `models/hcast/internal/utils.py`: Tensor and embedding helpers.

These files are the closest local equivalents of the upstream CAST internals and are usually treated as vendored backbone code.

### LH-DNN

- `models/lhdnn/__init__.py`: Public LH-DNN exports.
- `models/lhdnn/factory.py`: Builds the fixed-topology LH-DNN model from config.
- `models/lhdnn/model.py`: Paper-aligned LH-DNN implementation with shared features, projection component, and taxonomy-aware advantage topology.
- `models/lhdnn/losses.py`: LH-DNN per-level cross-entropy loss with optional soft-target support from mixup/cutmix.

### HT-CapsNet

- `models/ht_capsnet/__init__.py`: Public HT-CapsNet exports.
- `models/ht_capsnet/factory.py`: Validates config and builds the HT-CapsNet model.
- `models/ht_capsnet/model.py`: PyTorch HT-CapsNet implementation with configurable backbone, routing levels, and cross-capsule attention.
- `models/ht_capsnet/routing.py`: Routing primitives and taxonomy-guided routing helpers.
- `models/ht_capsnet/losses.py`: Capsule margin loss and hierarchy-aware penalties.

### HRN

- `models/hrn/__init__.py`: Public HRN exports.
- `models/hrn/factory.py`: Enforces the 3-level constraint and builds `HRNModel`.
- `models/hrn/model.py`: ResNet-50 HRN implementation with upstream branch blocks, residual fusion, and low-LR trunk parameter groups.
- `models/hrn/losses.py`: HRN combinatorial tree loss and fine-level cross-entropy objective, including upstream-style leaf-only CE handling.

### Hier-COS

- `models/hiercos/__init__.py`: Public Hier-COS exports.
- `models/hiercos/factory.py`: Builds `HierCosModel` from config and taxonomy metadata.
- `models/hiercos/model.py`: Hier-COS model with taxonomy-driven node subspaces and an upstream-style fixed random orthonormal frame.
- `models/hiercos/losses.py`: Hier-COS loss selection between paper-aligned KL + level regularization (`kl_reg`), lex-ready per-level KL+reg decomposition (`per_level_kl_reg`), and local weighted three-level CE ablations (`per_level_ce` on subspace scores, `per_level_abs_node_ce` on `abs(node_logits)` node slices).

## Training

- `train/__init__.py`: Package marker for training utilities.
- `train/train.py`: Main CLI entrypoint (`python -m train.train`) and run orchestration (loader/model/runtime setup, train loop, checkpoint selection, final test evaluation).
- `train/config_loader.py`: YAML config loader with optional OmegaConf support and dotlist override handling.
- `train/engine.py`: `train_one_epoch` and `evaluate` loops.
- `train/evaluation.py`: Batch metric assembly.
- `train/metric_formatting.py`: Console metric formatting (`pretty_metrics`).
- `train/metrics.py`: Shared hierarchical metrics such as per-level accuracy, weighted AP, FPA, AHD, and TICE.
- `train/mixup.py`: Mixup/CutMix helpers used by the unified training loop.
- `train/training_logger.py`: Writes `config_resolved.yaml`, `run_log.jsonl`, and `test_metrics.yaml`.
- `train/lexicographic/`: Lexicographic config/types and trunk-gradient diagnostics/projection utilities.
- `train/runtime/`: Runtime concerns split by responsibility (optimization, finetune loading, checkpoint/resume, best-checkpoint selection), imported directly from concrete modules.

## Notebooks

- `notebooks/hcast_analysis.ipynb`: H-CAST result analysis notebook.
- `notebooks/hcast_analysis_utils.py`: Shared loaders, run selection, plotting, and table utilities used by `hcast_analysis.ipynb`.
- `notebooks/hcc_internal_diagnostics.ipynb`: HCC diagnostics notebook over `run_log.jsonl` with switch-focused plots/tables.
- `notebooks/hcc_failure_examples.ipynb`: CUB/Aircraft qualitative notebook for cases where independent fails and top-down succeeds.
- `notebooks/model_comparison_all_datasets.ipynb`: Cross-model comparison notebook across datasets.

## Docs

- `docs/FILE_DOCUMENTATION.md`: This repository map.
- `docs/HCC_DIAGNOSTIC_LOGS.md`: Glossary and interpretation of HCC diagnostic metric keys.
- `docs/GRADIENT_PARAM_DIAGNOSTIC_LOGS.md`: Glossary and interpretation of trunk gradient/parameter diagnostics, including lexicographic metrics.
- `docs/hrn_hiercos_alignment.md`: HRN/Hier-COS upstream-alignment audit and intentional divergence notes.

## Runtime Artifacts

Runtime outputs are not tracked in the repository tree. Each training run writes to its configured `train.output_dir`, typically producing:

- `latest.pt`
- `best_topdown.pt`
- `best_independent.pt`
- `config_resolved.yaml`
- `run_log.jsonl`
- `test_metrics.yaml`

Dataset files are also external to the repository and are located through `dataset.root` in each config.
