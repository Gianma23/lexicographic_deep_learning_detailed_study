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
- `models/common/hcc.py` — HCC affine hierarchy projection and the shared
  on/off controller used by every HCC-capable model.
- `models/common/subspace_supervision.py` — mixed-precision-safe taxonomy
  subspace norms, the attainable path-energy target profile rebuilt from the
  model's own level weights, decoder-aligned tempered soft cross-entropy on a
  level-shared score scale averaged uniformly over levels, and capability
  validation.
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

### Hier-COS

- `models/hiercos/model.py` — fixed-frame node-space model with WideResNet or
  ResNet-50 backbone, optional LH-projected learnable level heads reading the
  transform output directly or through an LH-DNN-style shared PReLU/rho
  derivative, detached advantage baselines, an independent identity or
  per-level block-diagonal fixed frame, and non-persistent path-overlap tables
  exposed through the shared supervision contract.
- `models/hiercos/factory.py` — Hier-COS model config adapter.
- `models/hiercos/losses.py` — `kl_reg`, `global_softmax_ce_reg`, and
  `level_softmax_ce_reg` objectives.
- `models/hiercos/topology.py` — taxonomy node ids, subspace masks, per-leaf
  path-overlap counts, and level path weights.
- `models/hiercos/fixed_frame.py` — fixed identity/random orthonormal
  classifiers.
- `models/hiercos/transforms.py` — full, BN-linear, and final-only
  transformations.
- `models/hiercos/config.py` — small Hier-COS config parsing helpers.

## Training and evaluation

- `train/train.py` — CLI and complete train/validation/test lifecycle.
- `train/config_loader.py` — mandatory OmegaConf loading, environment
  interpolation, dotlist overrides, and validation call.
- `train/config_validation.py` — strict allowed-key schema, numeric checks,
  and model/dataset/HCC/lex/direct-subspace compatibility.
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

- `train/lexicographic/config.py` — priority-mode and selected-support-block
  resolution plus compatibility checks.
- `train/lexicographic/gradients.py` — exact `p...` gradient-support detection,
  block-selected diagnostics and projection, and custom gradient assignment.
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
- `scripts/hcast/` — base, HCC, and lexicographic studies.
- `scripts/lhdnn/` — paper-aligned CIFAR-100 baseline plus explicit
  CUB/Aircraft protocol extrapolations.
- `scripts/capsnet/` — native HT-CapsNet baselines with dynamic level weights
  and lexicographic studies with unit level weights.
- `scripts/hrn/` — base and native level-marginal lexicographic studies.
- `scripts/hiercos/` — decomposed-loss baselines, lexicographic priority modes,
  transform ablation, the LH-projected learnable-head study, and the
  direct-subspace supervision launcher.
- `scripts/stage_thesis_figures.py` — stages the selected baseline and
  direct-subspace analysis PDFs under `docs/images/experiments/` and removes
  raster duplicates from those destination directories.
- `scripts/migrate_single_seed_outputs.py` — dry-run-first historical output
  nesting migration.

Runner matrices are overridable using `DATASETS`, `LEX_PROJECTION_MODE(S)`, and
`TRANSFORM_MODES`, depending on the script. Current
narrow defaults are printed during `DRY_RUN`. Lexicographic run directories are
named `<model>_<dataset>[_<loss>]_lex_<mode>[...]`.

## Grid search

- `gridsearch/hiercos_cub200_optuna.py` — original CUB Hier-COS Optuna study.
- `gridsearch/hiercos_cub200_refined_optuna.py` — refined search space and
  output layout.

## Analysis notebooks and helpers

All notebooks and their shared helper modules live under `notebooks/`, split
into four areas: cross-cutting notebooks directly under `notebooks/`,
per-model finished-run analyses under `notebooks/model_analysis/`, post-hoc
inference comparisons under `notebooks/inference_analysis/`, live-run
trade-off plots under `notebooks/tradeoff_analysis/`, and their shared Python
helpers under `notebooks/utils/`.

- `notebooks/datasets_analysis.ipynb` — dataset-level analysis and the figures
  exported for the thesis dataset section.
- `notebooks/model_comparison_all_datasets.ipynb` — cross-model, cross-dataset
  comparison: one run per family, each family's baseline, on the same axes. It
  is the transpose of the family notebooks below and shares their code and
  figure style through `HCastAnalysis.for_baseline_comparison`, which reads each
  baseline out of the same `FAMILY_PROFILES` registry they use. It renders the
  subset of their sections that is well defined across architectures - run
  matrix, validation curves, per-level validation accuracy, per-run level
  objectives, test tables - and omits the aggregate objective, the
  gradient-support blocks, since `total` is a different functional per family and
  the blocks are a property of the architecture. Its second half is the
  cross-model mechanism comparison, driven by `MechanismDeltaComparison`: for
  each mechanism registered in `FamilyProfile.mechanism_arms` (`hcc`,
  `lex_coarse_first`, `lex_fine_first`) it pairs every family's arm against that
  family's own baseline, per seed, and plots and tabulates the difference. It
  never puts two architectures on a shared axis there, because the gap between
  models is far larger than the effect of a mechanism on either. It also diffs
  the two resolved configs of every pair, which is what surfaces a second change
  travelling with the mechanism - H-CAST's lexicographic mode requires
  `model.loss.globalkl: false`, and the HRN arms also switch the objective to
  `level_conditional` (which is `native` regrouped, not a different objective).
- `notebooks/posthoc_hiercos_inference_comparison.ipynb` — the original
  single-family post-hoc inference notebook, superseded by
  `notebooks/inference_analysis/`.

`notebooks/inference_analysis/` holds the post-hoc inference comparisons. All of
them drive the CLI documented in `evaluation/README.md`, read the per-run
`posthoc_inference_test_metrics.yaml` it writes, and share
`notebooks/utils/posthoc_inference_utils.py`:

- `notebooks/inference_analysis/baseline.ipynb`,
  `lexmode.ipynb`, `hcc.ipynb`, `lhprojection.ipynb`,
  `subspace_supervision.ipynb` — one notebook per training mechanism, each
  reporting the within-checkpoint gain of every readout x transform cell against
  that mechanism's own native inference.
- `notebooks/inference_analysis/all_mechanics.ipynb` — every mechanism on one
  grid at every inference cell, for the best (mechanism, inference) combination
  overall. Its cross-mechanism deltas pair two different training runs by seed
  only, so they mix the mechanism with the readout and are not the
  within-checkpoint gains the per-mechanism notebooks report.

`notebooks/model_analysis/` holds one notebook per model family for finished,
selected-checkpoint analysis. The five family notebooks are deliberately
identical: the same eleven sections in the same order, driven by the same code,
so a section of one can be read directly against the same section of another.
The only line that differs is `FAMILY` in the setup cell, and the run matrix,
labels and aggregate loss terms that selects come from `FAMILY_PROFILES` in
`notebooks/utils/hcast_analysis_utils.py`. A section that comes out shorter for
one family is a fact about the family - a loss term it never logs, a gradient-support
block its architecture does not contain, a diagnostic that is constant or that
duplicates another panel exactly - and each of those prints a line saying so. The
degenerate cases are detected from the values rather than listed, so a quantity that
stops being degenerate reappears on its own.

- `notebooks/model_analysis/hcast_analysis.ipynb`
- `notebooks/model_analysis/lhdnn_analysis.ipynb`
- `notebooks/model_analysis/ht_capsnet_analysis.ipynb` — plus one
  family-specific appendix, the per-seed margin-collapse screen.
- `notebooks/model_analysis/hrn_analysis.ipynb`
- `notebooks/model_analysis/hiercos_analysis.ipynb`
- `notebooks/model_analysis/hcc_internal_diagnostics.ipynb` — not part of that
  set: a standalone cross-family baseline-versus-HCC notebook with its own
  loaders and plotting helpers.

`notebooks/utils/` holds the shared Python helpers imported by the notebooks
above:

- `notebooks/utils/hcast_analysis_utils.py` — the family registry, the run
  loaders, and every figure used by `notebooks/model_analysis/` and by
  `notebooks/model_comparison_all_datasets.ipynb`.
- `notebooks/utils/thesis_style.py` — the single print style, page geometry, and
  display-name tables the exported figures share.
- `notebooks/utils/model_comparison_utils.py` — the previous, separate
  implementation of the cross-model comparison. No notebook imports it since
  that comparison moved onto `hcast_analysis_utils.py`.
- `notebooks/utils/multiseed_utils.py`
- `notebooks/utils/posthoc_inference_utils.py` — run discovery per training
  mechanism, the evaluator sweep, the long-form result loader, the paired and
  cross-mechanism gain tables, and every figure used by
  `notebooks/inference_analysis/`.

Current-run trade-off analyses live under `notebooks/tradeoff_analysis/`, one
notebook per model family:

- `notebooks/tradeoff_analysis/hiercos_current_plots.ipynb` — the full Hier-COS
  baseline, lexicographic, projection, and ablation analysis.
- `notebooks/tradeoff_analysis/hcast_current_plots.ipynb` — H-CAST baseline,
  no-global-KL baseline, explicit-lexicographic, and HCC comparison.
- `notebooks/tradeoff_analysis/hrn_current_plots.ipynb` — HRN baseline,
  explicit-lexicographic, and HCC comparison.
- `notebooks/tradeoff_analysis/lhdnn_current_plots.ipynb` — the single LH-DNN
  baseline arm placed among the other model families on a shared scale.
- `notebooks/tradeoff_analysis/htcapsnet_current_plots.ipynb` — HT-CapsNet
  baseline, explicit-lexicographic, and HCC comparison, with the
  margin-collapse check that must precede any reading of the numbers.
- `notebooks/utils/current_run_plot_utils.py` — shared aggregation,
  reference-model discovery, the visual-encoding layer, off-scale gutter layout,
  collision-aware labels, HCC-activation verification, trade-off plotting,
  absolute and delta level-accuracy plotting, and Pareto-summary helpers for all
  current-run notebooks.
- `tests/test_current_run_encoding.py` — unit tests for the encoding layer:
  channel resolution, collision detection, palette separation, legend mode, and
  the point-label policy.

### Visual encoding

`encode_rows()` resolves visual channels from row properties; a notebook declares
which property drives which channel, and the legend is generated to match:

```python
ENCODING = encode_rows(rows, hue=('mechanism', MECHANISM_COLORS),
                             shape=('variant', VARIANT_MARKERS))
```

**Colour is globally semantic**: it encodes the `mechanism` — the intervention a
run applies (`baseline`, `lex`, `hcc`, `projection`) — and
means the same thing in every model's figure, so HCC `step@0` is the same green
everywhere. **Shape is locally semantic**: each notebook declares what it
separates (the variant within a mechanism for the single-family notebooks, the
loss family for Hier-COS) and the legend spells it out. `fill` (hollow) and
`ring` (a charcoal halo) carry one boolean each; anything left over goes to the
direct labels. Cross-model references give up colour entirely and are drawn in
neutral grey, told apart by marker shape, so the focal family owns the colour
dimension. Bars reuse the same encoding with hatch standing in for marker shape.

`check_encoding()` reports runs that resolve to one glyph within a dataset panel;
those pairs are separated only by their direct labels, which is why Hier-COS runs
with `point_labels='all'` while the others use the default.

`plot_tradeoff(point_labels=...)` selects which direct labels are drawn:
`'off_scale'` (the default) keeps only the gutter-pinned references, whose drawn
position is deliberately false and whose label is therefore the only record of
the true value; `'all'` keeps everything; `'none'` keeps nothing and warns;
`'auto'` labels focal runs only when the legend is not a complete key and
in-range references only on an unfrozen panel. A panel that never freezes has no
pinned references, so under the default it carries no direct labels at all and
relies on the legend and the axes.

Every current-run notebook reads the independently selected checkpoint and the
independent metric family. Top-down decoding is intentionally not offered there:
its predicted path is consistent by construction, so `tice_topdown` is
identically zero and `fpa_topdown` collapses onto top-down fine accuracy, leaving
a top-down trade-off view with nothing to show. Top-down results remain available
from `test_metrics.yaml` and from the `notebooks/model_analysis/` analyses.

Figures are authored for the thesis, not for the screen. `use_paper_style()` in
`current_run_plot_utils.py` applies the same rcParams as
`notebooks/datasets_analysis.ipynb` (DejaVu Serif, 9 pt base, 400 dpi raster,
`pdf.fonttype=42`, constrained layout), and `save_figure()` writes a PDF and a
PNG at the authored size without `bbox_inches='tight'` for runtime outputs;
destinations under `docs/` receive PDF only. Every figure is 6.3 in
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
contracts, direct subspace targets/losses/gradients, launchers, and
documentation paths.

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
