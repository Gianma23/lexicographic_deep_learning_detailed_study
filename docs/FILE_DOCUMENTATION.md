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

- `configs/hcast/` — three base H-CAST presets only; the HCC and explicit
  lexicographic variants are CLI overrides applied by their launchers.
- `configs/lhdnn/` — CIFAR-100 plus CUB/Aircraft extrapolation presets.
- `configs/capsnet/` — HT-CapsNet presets for CIFAR-100, CUB, and Aircraft.
- `configs/hrn/` — HRN presets for CIFAR-100, CUB, and Aircraft.
- `configs/hiercos/` — Hier-COS presets for CIFAR-100, CUB, and Aircraft.

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

- `models/ht_capsnet/model.py` — PyTorch capsule architecture, final-feature
  EfficientNet wrapper, Keras-shaped SDPA attention, and checkpointed dynamic
  loss-weight state.
- `models/ht_capsnet/routing.py` — taxonomy-guided routing.
- `models/ht_capsnet/losses.py` — capsule margin loss and next-batch
  dynamic/static level weighting.
- `models/ht_capsnet/factory.py` — deterministic and complete-taxonomy checks.

### HRN

- `models/hrn/model.py` — ResNet-50 RFM trunk, three branches, residual
  fusion, leaf logits, and optimizer parameter groups.
- `models/hrn/losses.py` — upstream-style tree-state loss plus leaf CE.
- `models/hrn/factory.py` — exact three-level guard.

### Hier-COS and the orthonormal plugin

- `models/hiercos/model.py` — fixed-frame node-space model with WideResNet or
  ResNet-50 backbone, optional LH-projected learnable level heads reading the
  transform output directly or through an LH-DNN-style shared PReLU/rho
  derivative, detached advantage baselines, and an independent identity or
  per-level block-diagonal fixed frame.
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
- `train/evaluation.py` — per-batch metric and HCC diagnostic assembly. Decodes
  once per decoder and shares the predictions with every metric;
  `include_diagnostics=False` drops the level-3 diagnostics for callers that
  discard them. The ground-truth rank diagnostic is a masked comparison over the
  cached sibling mask rather than a per-sample loop, which matters because
  `train_one_epoch` calls `evaluate_batch` on every training batch.
- `train/metrics.py` — independent/top-down decoding, per-level accuracy,
  weighted AP, FPA, AHD, TICE, and sample-weighted aggregation. The per-level
  `allowed` mask and child→parent lookup are cached per taxonomy and device;
  each metric also accepts precomputed `preds` to avoid decoding again.
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

## Checkpoint-only inference evaluation

- `evaluation/README.md` — modes, command, checkpoint rule, and output
  contract for inference-only comparisons.
- `evaluation/posthoc_inference.py` — inference-only readout x transform grid:
  `node_score` or `subspace_norm` readout over the model's node coordinates,
  optionally preceded by the HCC affine hierarchy projection (`alpha=1`, no
  training). Each model's own inference is one cell of the grid.
- `evaluation/evaluate_checkpoints.py` — CLI for comparing inference rules on
  existing `best_topdown.pt` and `best_independent.pt` checkpoints without
  constructing a loss or optimizer.

## Experiment and data scripts

- `scripts/load_env.sh` — shell `.env` loading with process-environment
  precedence.
- `scripts/run_seed_utils.sh` — seed matrix and nested output helpers.
- `scripts/run_matrix_utils.sh` — whitespace-separated environment matrix
  parsing and allowed-value validation.
- `scripts/hcast/` — base, HCC, lexicographic, and plugin studies.
- `scripts/lhdnn/` — paper-aligned CIFAR-100 baseline plus explicit
  CUB/Aircraft protocol extrapolations.
- `scripts/capsnet/` — native HT-CapsNet baselines with dynamic level weights
  and lexicographic studies with unit level weights.
- `scripts/hrn/` — base, native level-marginal lexicographic, and plugin studies.
- `scripts/hiercos/` — decomposed-loss baselines, two lexicographic rules,
  transform ablation, and the LH-projected learnable-head study.
- `scripts/migrate_single_seed_outputs.py` — dry-run-first historical output
  nesting migration.

Runner matrices are overridable using `DATASETS`, `LEX_PROJECTION_MODE(S)`,
`LEX_PROJECTION_RULE`, and `TRANSFORM_MODES`, depending on the script. Current
narrow defaults are printed during `DRY_RUN`. Lexicographic run directories are
named `<model>_<dataset>[_<loss>]_lex_<rule>_<mode>[...]`.

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
- `notebooks/posthoc_hiercos_inference_comparison.ipynb`
- `notebooks/hcast_analysis_utils.py`
- `notebooks/model_comparison_utils.py`
- `notebooks/multiseed_utils.py`

Current-run trade-off analyses live directly under `analysis/current_runs/`, one
notebook per model family:

- `analysis/current_runs/hiercos_current_plots.ipynb` — the full Hier-COS
  baseline, lexicographic, projection, and ablation analysis.
- `analysis/current_runs/hcast_current_plots.ipynb` — H-CAST baseline,
  no-global-KL baseline, explicit-lexicographic, and HCC comparison.
- `analysis/current_runs/hrn_current_plots.ipynb` — HRN baseline,
  explicit-lexicographic, and HCC comparison.
- `analysis/current_runs/lhdnn_current_plots.ipynb` — the single LH-DNN baseline
  arm placed among the other model families on a shared scale.
- `analysis/current_runs/htcapsnet_current_plots.ipynb` — HT-CapsNet baseline,
  explicit-lexicographic, and HCC comparison, with the margin-collapse check
  that must precede any reading of the numbers.
- `analysis/current_runs/current_run_plot_utils.py` — shared aggregation,
  reference-model discovery, off-scale gutter layout, collision-aware labels,
  HCC-activation verification, trade-off plotting, absolute and delta
  level-accuracy plotting, and Pareto-summary helpers for all current-run
  notebooks.

Every current-run notebook reads the independently selected checkpoint and the
independent metric family. Top-down decoding is intentionally not offered there:
its predicted path is consistent by construction, so `tice_topdown` is
identically zero and `fpa_topdown` collapses onto top-down fine accuracy, leaving
a top-down trade-off view with nothing to show. Top-down results remain available
from `test_metrics.yaml` and from the older `notebooks/` analyses.

Figures are authored for the thesis, not for the screen. `use_paper_style()` in
`current_run_plot_utils.py` applies the same rcParams as
`analysis/datasets_analysis.ipynb` (DejaVu Serif, 9 pt base, 400 dpi raster,
`pdf.fonttype=42`, constrained layout), and `save_figure()` writes a PDF and a
PNG at the authored size without `bbox_inches='tight'`. Every figure is 6.3 in
wide — an A4 text block with 2.5 cm margins — and must be included at
`\linewidth` with no further scaling, since resizing in LaTeX would shrink the
fonts along with the artwork. Figure titles are omitted on purpose; they belong
in the caption.

Each figure stacks one full-width panel per dataset rather than three panels in
a row: at 6.3 in a 1×3 row leaves about 2 in per panel, too little for the direct
labels or for grouped bars once a run matrix grows. The delta figure gives each
panel its own vertical scale, because delta ranges differ by an order of
magnitude across datasets; captions should say so.

Notebook files are not rewritten by repository audits.

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
