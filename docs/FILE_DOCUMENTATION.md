# Repository map

This document maps the current tracked source tree. Generated experiment
outputs and external datasets are not part of the repository.

## Root

- `.env.example` — machine-local dataset roots, output root, device, and
  launcher defaults.
- `README.md` — setup, protocol, presets, datasets, metrics, launchers, and
  verification.
- `requirements.txt` — Python dependencies other than machine-specific
  PyTorch and torchvision builds.
- `AGENTS.md` — repository research, safety, coding, and reporting rules.

There is no tracked `TODO.md`.

## Configuration

Runnable experiments:

- `configs/hcast/` — four base H-CAST presets, three HCC presets, and three
  explicit lexicographic presets.
- `configs/lhdnn/` — CIFAR-100 plus CUB/Aircraft extrapolation presets.
- `configs/capsnet/` — HT-CapsNet presets for CIFAR-100, CUB, and Aircraft.
- `configs/hrn/` — HRN presets for CIFAR-100, CUB, and Aircraft.
- `configs/hiercos/` — Hier-COS presets for CIFAR-100, CUB, Aircraft, and
  iNat19.

All runnable configs contain the standard sections `model`, `dataset`,
`dataloader`, `train`, `optim`, `scheduler`, and `runtime`, and are tagged
`runtime.protocol: corrected_unified_v1`.

Commented schema fragments:

- `configs/templates/dataset_template.yaml`
- `configs/templates/hcast_template.yaml` — includes the optional top-level
  HCC block.
- `configs/templates/ht_capsnet_template.yaml`
- `configs/templates/hrn_template.yaml`
- `configs/templates/hiercos_template.yaml`
- `configs/templates/orthonormal_plugin_template.yaml`
- `configs/templates/training_template.yaml`

Templates are validated schema fragments, not standalone runnable configs.
There is no separate `hcc_template.yaml`.

## Datasets

- `datasets/__init__.py` — small stable public API for dataset contracts,
  transforms, and loader builders.
- `datasets/types.py` — `DatasetLabelSpace` and the non-duplicating
  `DatasetMetadata` view.
- `datasets/base.py` — compact PyTorch dataset lifecycle, image checks, and
  normalized JSON annotation I/O.
- `datasets/hierarchy.py` — pure canonical mapping, remapping, and taxonomy
  validation operations.
- `datasets/splitting.py` — deterministic stratified train/validation splits.
- `datasets/transforms.py` — timm/manual training and evaluation transforms.
- `datasets/loaders.py` — adapter registry, deterministic DataLoaders, and
  shared train/validation/test construction.
- `datasets/cifar100.py` — official CIFAR-100 fine/coarse ingestion and the
  explicit B-CNN 8-to-20 parent edge.
- `datasets/cub.py` — folder and official-file CUB ingestion.
- `datasets/cub_tree.py` — retained H-CAST order/family/species taxonomy.
- `datasets/aircraft.py` — official class lists and parallel
  manufacturer/family/variant annotations with completeness checks.
- `datasets/inat.py` — official COCO/JSON-in-tar and normalized explicit iNat19
  manifests projected to family/genus/species.

Validation and test datasets always reuse the training/authoritative label
space. Explicit missing annotations and previously silent malformed rows are
fatal errors.

## Models

`models/__init__.py` is the unified model/loss dispatcher.

### H-CAST

- `models/hcast/factory.py` — config adapter.
- `models/hcast/model.py` — unified H-CAST wrapper and HCC integration.
- `models/hcast/losses.py` — per-level loss and optional global KL.
- `models/hcast/segments.py` — grid/SEEDS segmentation.
- `models/hcast/hard_hierarchy.py` — HCC affine hierarchy projection and
  schedule.
- `models/hcast/internal/` — vendored CAST/H-CAST backbone implementation.

### LH-DNN

- `models/lhdnn/model.py` — paper-derived large topology, forward-pass
  projection blocks, advantage topology, and explicit optional large-image
  deterministic average pooling.
- `models/lhdnn/losses.py` — unweighted sum of level cross-entropies.
- `models/lhdnn/factory.py` — config and taxonomy adapter.

### HT-CapsNet

- `models/ht_capsnet/model.py` — PyTorch capsule architecture, backbones, and
  cross-capsule attention.
- `models/ht_capsnet/routing.py` — taxonomy-guided routing.
- `models/ht_capsnet/losses.py` — capsule margin loss and dynamic/static level
  weighting.
- `models/ht_capsnet/factory.py` — deterministic and complete-taxonomy checks.

### HRN

- `models/hrn/model.py` — ResNet-50 RFM trunk, three branches, residual
  fusion, leaf logits, and optimizer parameter groups.
- `models/hrn/losses.py` — upstream-style tree-state loss plus leaf CE.
- `models/hrn/factory.py` — exact three-level guard.

### Hier-COS and the orthonormal plugin

- `models/hiercos/model.py` — fixed-frame node-space model with WideResNet or
  ResNet-50 backbone and optional LH-projected learnable level heads, detached
  advantage baselines, and a global fixed frame.
- `models/hiercos/factory.py` — model/plugin config adapter.
- `models/hiercos/losses.py` — compatibility export for shared fixed-frame
  losses.
- `models/orthonormal_plugin/topology.py` — taxonomy node ids, subspace masks,
  and path targets.
- `models/orthonormal_plugin/head.py` — fixed identity/random orthonormal
  classifiers.
- `models/orthonormal_plugin/transforms.py` — full, BN-linear, and final-only
  transformations.
- `models/orthonormal_plugin/losses.py` — `kl_reg`,
  `global_softmax_ce_reg`, and `level_softmax_ce_reg`.
- `models/orthonormal_plugin/wrapper.py` — optional adapter for non-Hier-COS
  host models.
- `models/orthonormal_plugin/config.py` — shared plugin config helpers.

## Training and evaluation

- `train/train.py` — CLI and complete train/validation/test lifecycle.
- `train/config_loader.py` — mandatory OmegaConf loading, environment
  interpolation, dotlist overrides, and validation call.
- `train/config_validation.py` — strict allowed-key schema, numeric checks,
  and model/dataset/HCC/plugin/lex compatibility.
- `train/engine.py` — training/evaluation loops, AMP, diagnostics, and
  lexicographic switch.
- `train/evaluation.py` — per-batch metric and HCC diagnostic assembly.
- `train/metrics.py` — independent/top-down decoding, per-level accuracy,
  weighted AP, FPA, AHD, TICE, and sample-weighted aggregation.
- `train/mixup.py` — hierarchy-aware MixUp/CutMix target construction.
- `train/metric_formatting.py` — concise console formatting.
- `train/training_logger.py` — resolved config, JSONL epoch/resume events, and
  final YAML test output.

Lexicographic modules:

- `train/lexicographic/config.py` — projection-mode/rule resolution and
  compatibility checks.
- `train/lexicographic/gradients.py` — trunk detection, raw gradient
  diagnostics, projection, and custom gradient assignment.
- `train/lexicographic/types.py` — typed lexicographic config/state records.

Runtime modules:

- `train/runtime/optimization.py` — deterministic seeding, optimizers, and
  schedulers.
- `train/runtime/checkpointing.py` — v2 exact-selection state, RNG/loader
  snapshots, strict resume, and legacy selection upgrade.
- `train/runtime/selection.py` — exact `(FPA, -TICE, weighted_AP)` comparison.
- `train/runtime/finetune.py` — trusted finetune/checkpoint loading.
- `train/runtime/common.py` — configuration conversion helpers.

## Experiment and data scripts

- `scripts/load_env.sh` — shell `.env` loading with process-environment
  precedence.
- `scripts/run_seed_utils.sh` — seed matrix and nested output helpers.
- `scripts/run_matrix_utils.sh` — whitespace-separated environment matrix
  parsing and allowed-value validation.
- `scripts/hcast/` — base, HCC, lexicographic, and plugin studies.
- `scripts/lhdnn/` — paper-aligned CIFAR-100 baseline plus explicit
  CUB/Aircraft protocol extrapolations.
- `scripts/hrn/` — base and plugin studies.
- `scripts/hiercos/` — decomposed-loss baselines, two lexicographic rules,
  transform ablation, and the LH-projected learnable-head study.
- `scripts/data/prepare_inat19_mbm_splits.py` — iNat Making Better Mistakes
  manifest preparation.
- `scripts/migrate_single_seed_outputs.py` — dry-run-first historical output
  nesting migration.

Runner matrices are overridable using `DATASETS`, `START_EPOCHS`,
`LEX_PROJECTION_MODES`, and `TRANSFORM_MODES`, depending on the script. Current
narrow defaults are printed during `DRY_RUN`.

## Grid search

- `gridsearch/hiercos_cub200_optuna.py` — original CUB Hier-COS Optuna study.
- `gridsearch/hiercos_cub200_refined_optuna.py` — refined search space and
  output layout.

## Analysis notebooks and helpers

- `notebooks/hcast_analysis.ipynb`
- `notebooks/hcc_internal_diagnostics.ipynb`
- `notebooks/hiercos_analysis.ipynb`
- `notebooks/lhdnn_analysis.ipynb`
- `notebooks/hrn_analysis.ipynb`
- `notebooks/model_comparison_all_datasets.ipynb`
- `notebooks/hcast_analysis_utils.py`
- `notebooks/model_comparison_utils.py`
- `notebooks/multiseed_utils.py`

`analysis/hiercos_current_runs/hiercos_current_plots.ipynb` is a separate
working analysis notebook. Notebook files are not rewritten by repository
audits.

## Documentation

- [This repository map](FILE_DOCUMENTATION.md)
- [Dated correctness/fidelity audit](REPOSITORY_AUDIT.md)
- [Pinned upstream comparison by model](model_repo_differences.md)
- [HCC diagnostic glossary](HCC_DIAGNOSTIC_LOGS.md)
- [Gradient, parameter, and lexicographic diagnostic glossary](GRADIENT_PARAM_DIAGNOSTIC_LOGS.md)
- [HCC/H-CAST research synthesis](hcc_hcast_research_report.md)
- [Detailed HRN/Hier-COS alignment notes](hrn_hiercos_alignment.md)
- `docs/generate_hcc_internal_insights_figures.py` — generates documentation
  figures from existing run logs; outputs belong under the external output
  root.

## Tests

Tests use `python -m unittest discover -s tests -v`. They cover official
hierarchies, fixed-frame loss equivalence, multiseed helpers, shared label
spaces, strict configs, exact checkpoint selection/resume, metrics, model
contracts, launchers, and documentation paths.

## Runtime artifacts

Each seed run is written outside the repository:

```text
<experiment>/seed_<n>/
  latest.pt
  best_topdown.pt
  best_independent.pt
  config_resolved.yaml
  run_log.jsonl
  test_metrics.yaml
```
