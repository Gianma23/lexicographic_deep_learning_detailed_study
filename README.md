# Lexicographic Deep Learning for Hierarchical Image Classification

A unified PyTorch framework for a master-thesis study of hierarchical image
classification. It runs five published families — H-CAST, LH-DNN, HT-CapsNet,
HRN and Hier-COS — through one training, evaluation and checkpoint-selection
lifecycle, then adds four controlled interventions on top of that common
substrate:

| Intervention | Where it acts | Config switch |
|---|---|---|
| HCC (Hierarchical Constraint Cascade) | output space, affine hierarchy projection | `hcc.enabled` |
| Direct subspace supervision | the objective | `train.subspace_supervision.enabled` |
| Lexicographic optimisation | parameter gradients | `train.lexicographic.enabled` |
| LH-projection | the forward graph, backward-only effect | `model.projection.enabled` |

The four are distinct mechanisms and are not interchangeable. The thesis text
lives in this repository as `docs/04-methodology.tex` (methods) and
`docs/05-experiments.tex` (protocol, fidelity and results).

The training entrypoint applies the corrected unified protocol to all shipped
presets. It fixes one canonical taxonomy per dataset, evaluates every validation
and test sample, and selects checkpoints on validation data only — deliberately
departing from reference codebases that drop evaluation batches or select on
test data. The protocol is enforced by configuration validation rather than by
a `runtime.protocol` field in each preset.

## Repository structure

```text
configs/        runnable presets (one per model x dataset) and commented templates
datasets/       adapters, taxonomy construction, splits, transforms
models/         hcast/ lhdnn/ ht_capsnet/ hrn/ hiercos/ and shared components
train/          training CLI, engine, config validation, metrics, diagnostics
evaluation/     checkpoint-only inference grid (no training, no optimiser)
scripts/        experiment launchers, one per model family and mechanism
notebooks/      analysis notebooks that turn run outputs into tables and figures
tests/          unittest suite
docs/           thesis chapters (.tex) and implementation reference (.md)
gridsearch/     Optuna studies (exploratory, not part of the reported results)
```

Model families are selected by `model.name`: `hcast`, `lhdnn`, `ht_capsnet`,
`hrn`, `hiercos`. Every family supports CIFAR-100, CUB-200-2011 and
FGVC-Aircraft, but "supported" means a runnable preset exists — not that the
original paper reported that pair. The unsupported family–dataset extrapolations
are listed in `docs/05-experiments.tex`, section *Baseline comparability*.

## Setup

Python 3.10+ and PyTorch 2.0+ (HT-CapsNet uses native scaled-dot-product
attention). Install a PyTorch build matching the machine's CUDA first; PyTorch
is intentionally unpinned in `requirements.txt` because those builds are
machine-specific.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

Then point the framework at the datasets and the output root:

```bash
cp .env.example .env
```

```text
# Local machine paths
CIFAR100_ROOT=/path/to/cifar100
CUB200_ROOT=/path/to/CUB_200_2011
AIRCRAFT_ROOT=/path/to/fgvc-aircraft-2013b
OUTPUTS_ROOT=/path/to/outputs

# Shared launcher defaults; individual commands can still override these
PYTHON_BIN=.venv/bin/python
TRAIN_DEVICE=cuda
MAX_PARALLEL=1
MAX_RESUME_RETRIES=1
NUM_RUNS=1
BASE_SEED=0
SPLIT_SEED=0
```

Existing process environment variables take precedence over `.env`, so the
reproduction commands below override these defaults rather than requiring the
file to be edited.

Datasets are read from their native metadata: CIFAR-100 from the Python archive
plus the published B-CNN 8→20 edge, CUB from `images.txt` /
`image_class_labels.txt` / `train_test_split.txt`, and Aircraft only from a
complete official download. Only `OUTPUTS_ROOT` is written to.

## Quick start

```bash
python -m train.train --config configs/hcast/hcast_cifar100.yaml
```

Any config key can be overridden on the command line as a dotlist, which is how
every mechanism is switched on:

```bash
python -m train.train --config configs/hcast/hcast_cifar100.yaml \
  train.epochs=1 dataloader.batch_size=4 \
  train.output_dir=$OUTPUTS_ROOT/smoke/hcast_cifar100
```

Configuration is fail-fast: unknown keys, mismatched level counts and
model/mechanism incompatibilities are rejected before training starts. Resume is
strict — only `train.resume`, `train.output_dir` and `train.stop_epoch` may
differ from the checkpoint's own resolved config:

```bash
python -m train.train --config configs/hcast/hcast_cifar100.yaml \
  train.resume=$OUTPUTS_ROOT/hcast_cifar100/seed_0/latest.pt
```

See [docs/PRESETS.md](docs/PRESETS.md) for what each preset sets and why.

## Reproducing the thesis results

Results come from the launchers under `scripts/`, at three seeds and a fixed
dataset split. Launcher defaults are deliberately narrow so that invoking a
script cannot accidentally start the full grid, **so the reproduction commands
below always pass the matrix explicitly.** Prefix any command with `DRY_RUN=1` to
print the exact `python -m train.train` invocations without running them.

Shared run-control variables, at the values the reported campaign used:

```text
DATASETS="cifar100 cub200 aircraft"   whitespace-separated, validated
NUM_RUNS=3                            seeds per arm
BASE_SEED=0                           first training seed (0,1,2)
SPLIT_SEED=0                          fixed dataset split seed, never varied
MAX_PARALLEL=1                        concurrent training processes
MAX_RESUME_RETRIES=3                  resume-from-latest.pt attempts on failure
DRY_RUN=1                             print commands only
RUN_PREFLIGHT=strict                  skip completed seeds, resume interrupted ones
```

Each launcher prints the matrix it selected before starting. `DATASETS` is
ignored by the two single-dataset launchers (the CIFAR-100 backbone ladder and
the Aircraft pretraining ablation); the rest honour it. `NUM_RUNS`,
`BASE_SEED`, `SPLIT_SEED`, `MAX_PARALLEL` and `MAX_RESUME_RETRIES` are also read
from `.env`, so the values there apply to anything the commands below leave
unset.

Re-running a launcher is safe and idempotent. Every launcher shares one resume
policy: a seed whose `test_metrics.yaml` already exists is skipped, one with a
`latest.pt` resumes from it, and a directory holding other artifacts but no
resumable checkpoint stops the campaign rather than being overwritten. Re-running
a partly finished campaign therefore fills in only what is missing. To retrain
something deliberately, remove its run directory first, or pass
`RUN_PREFLIGHT=none` to launch regardless of what is already there.

`SPLIT_SEED=0` must stay fixed: the CIFAR-100 and CUB validation splits are built
once with it and reused by every arm, which is what makes seed-matched
differences meaningful.

### Step 0 — verify the installation

```bash
python -m unittest discover -s tests -v
DRY_RUN=1 scripts/hcast/run_hcast_baselines.sh
python -m train.train --config configs/hcast/hcast_cifar100.yaml \
  train.epochs=1 dataloader.batch_size=4 \
  train.output_dir=$OUTPUTS_ROOT/smoke/hcast_cifar100
```

The smoke run should produce a complete seed directory (below). If it does, the
dataset paths, taxonomy and device assumptions are all correct.

### Step 1 — unified baselines

These populate the *Unified baselines* section and are the matched reference for
every later comparison. Run all of them before anything else.

```bash
export DATASETS="cifar100 cub200 aircraft" NUM_RUNS=3 BASE_SEED=0 SPLIT_SEED=0

scripts/hcast/run_hcast_baselines.sh              # -> hcast_<ds>
scripts/lhdnn/run_lhdnn_baselines.sh              # -> lhdnn_<ds>
scripts/capsnet/run_ht_capsnet_baselines.sh       # -> capsnet_<ds>
scripts/hrn/run_hrn_baselines.sh                  # -> hrn_<ds>_level_conditional

LOSS_MODE=global_softmax_ce_reg WEIGHT_MODE=kl_leaf \
FIXED_FRAME_MODE=orthonormal_random FIXED_FRAME_PER_LEVEL=false \
  scripts/hiercos/run_hiercos_baselines.sh
  # -> hiercos_<ds>_global_softmax_ce_reg_baseline_kl_leaf
```

Two baselines are not the family's own default and must be set as shown. HRN
reports `level_conditional`, which splits the native tree objective into three
conditional terms that sum back to it — same gradient, but three level
objectives for the mechanism experiments to project. Hier-COS reports the
matched global-softmax CE reference on the published **dense** frame
(`FIXED_FRAME_PER_LEVEL=false`), not the launcher's default block frame.

### Step 2 — trained mechanisms

Each arm is compared against its Step 1 baseline at matched seeds, backbone and
schedule.

```bash
export DATASETS="cifar100 cub200 aircraft" NUM_RUNS=3 BASE_SEED=0 SPLIT_SEED=0

# HCC — output-space constraint, active from the first batch
scripts/hcast/run_hcast_hcc.sh                    # -> hcast_<ds>_hcc
scripts/capsnet/run_ht_capsnet_hcc.sh             # -> capsnet_<ds>_hcc
scripts/hrn/run_hrn_hcc.sh                        # -> hrn_<ds>_level_conditional_hcc
scripts/hiercos/run_hiercos_hcc.sh                # -> hiercos_<ds>_<loss>_hcc

# Gradient-space lexicographic optimisation
LEX_PROJECTION_MODE=coarse_first scripts/hcast/run_hcast_lex.sh
LEX_PROJECTION_MODE=fine_first   scripts/hcast/run_hcast_lex.sh
scripts/capsnet/run_ht_capsnet_lex.sh             # coarse_first by default
scripts/hrn/run_hrn_lex.sh                        # coarse_first by default

LOSS_MODE=global_softmax_ce_reg WEIGHT_MODE=kl_leaf \
FIXED_FRAME_MODE=orthonormal_random FIXED_FRAME_PER_LEVEL=false \
LEX_PROJECTION_MODES="coarse_first fine_first" \
  scripts/hiercos/run_hiercos_lex.sh
  # -> hiercos_<ds>_global_softmax_ce_reg_lex_<mode>_kl_leaf

# Direct supervision of taxonomy-subspace scores (Hier-COS only)
SUBSPACE_TAU=0.1 scripts/hiercos/run_hiercos_subspace.sh
  # -> hiercos_<ds>_subspace
```

The Hier-COS lex launcher defaults to `global_softmax_ce_reg` and `kl_leaf`, so
its objective and level weights already match the Step 1 baseline: enabling the
mechanism does not silently change the loss mode or the weighting alongside it,
which on the weights alone would be worth roughly 1–2 pp of fine accuracy. Its
frame default is still the block frame, so `FIXED_FRAME_PER_LEVEL=false` is
passed above to put both sides of the comparison on the published dense,
globally normalised configuration. Everything up to Step 3 stays there.

Priority order is an axis, not a default: H-CAST's launcher defaults to
`fine_first` while HT-CapsNet's and HRN's default to `coarse_first`, so pass
`LEX_PROJECTION_MODE` explicitly for both arms of that comparison. Lex arms are
named `<model>_<ds>_lex_<mode>`; HCC arms append `_hcc`. Lexicographic
projection has no onset control — when enabled it is active for the whole run,
so a directory name alone never establishes when a mechanism became active.
Confirm activation from `config_resolved.yaml`, and for HCC from
`proj_constraint_alpha` in `run_log.jsonl`.

`SUBSPACE_TAU=0.1` is the campaign value; the launcher's own default temperature
is also `0.1`, and earlier runs under the same directory name used superseded
objectives.

### Step 3 — Hier-COS substrate and LH-projection

This step changes the Hier-COS substrate once, in two stages, and then runs the
LH-projection on the result. Both stages hold `kl_leaf` weights and no mechanism
fixed, varying only the named axis.

```bash
export NUM_RUNS=3 BASE_SEED=0 SPLIT_SEED=0

# (a) Frame ladder at global softmax. The dense row is already Step 1.
DATASETS="cifar100 cub200 aircraft" LOSS_MODE=global_softmax_ce_reg \
FIXED_FRAME_MODE=identity FIXED_FRAME_PER_LEVEL=false \
  scripts/hiercos/run_hiercos_baselines.sh
  # -> hiercos_<ds>_global_softmax_ce_reg_baseline_kl_leaf_identity
DATASETS="cifar100 cub200 aircraft" LOSS_MODE=global_softmax_ce_reg \
FIXED_FRAME_MODE=orthonormal_random FIXED_FRAME_PER_LEVEL=true \
  scripts/hiercos/run_hiercos_baselines.sh
  # -> hiercos_<ds>_global_softmax_ce_reg_baseline_kl_leaf_block

# (b) Softmax scope, at each dataset's selected frame from (a)
DATASETS=cifar100 LOSS_MODE=level_softmax_ce_reg \
FIXED_FRAME_MODE=orthonormal_random FIXED_FRAME_PER_LEVEL=true \
  scripts/hiercos/run_hiercos_baselines.sh
  # -> hiercos_cifar100_level_softmax_ce_reg_baseline_kl_leaf_block
DATASETS="cub200 aircraft" LOSS_MODE=level_softmax_ce_reg \
FIXED_FRAME_MODE=identity FIXED_FRAME_PER_LEVEL=false \
  scripts/hiercos/run_hiercos_baselines.sh
  # -> hiercos_<ds>_level_softmax_ce_reg_baseline_kl_leaf_identity

# (c) LH-projection on that substrate
DATASETS=cifar100 FIXED_FRAME_MODE=orthonormal_random FIXED_FRAME_PER_LEVEL=true \
  scripts/hiercos/run_hiercos_lhdnn_projection.sh
DATASETS="cub200 aircraft" FIXED_FRAME_MODE=identity FIXED_FRAME_PER_LEVEL=false \
  scripts/hiercos/run_hiercos_lhdnn_projection.sh
  # -> hiercos_<ds>_level_softmax_ce_reg_projection_d512_kl_leaf_<frame>

# (d) CIFAR-100 backbone-capacity ladder (3 seeds, sensitivity only)
NUM_RUNS=3 WRN_SIZES="16-8 28-4" \
  scripts/hiercos/run_hiercos_cifar100_backbone_ladder.sh
```

The frame is dataset-dependent and is a *result* of stage (a), not a convention:
block for CIFAR-100, identity for CUB-200 and Aircraft. It matters downstream
because a dense frame is rejected under `projection.enabled=true`, so the LH arms
in (c) must sit on the LH-compatible frame their dataset selected. Hold the frame
fixed within a dataset and carry the substrate column on every mechanism table.

`FEATURE_DIM` selects the shared representation width (`0` = the dataset's
taxonomy width; the projection launcher defaults to `512`) and appends `_d<dim>`
to the run name. `WEIGHT_MODE` appends its own suffix for anything other than
`equal`. Set both explicitly when reproducing a specific row.

The LH-projection on HRN has no launcher — it is driven by CLI overrides on the
HRN preset, and it requires zero dropout and is mutually exclusive with HCC:

```bash
python -m train.train --config configs/hrn/hrn_cifar100.yaml \
  model.projection.enabled=true model.dropout=0 \
  train.output_dir=$OUTPUTS_ROOT/hrn_cifar100_projection/seed_0 \
  dataset.split_seed=0 train.seed=0
```

### Step 4 — post-hoc inference grid

Inference-only. It re-reads frozen checkpoints through a readout × transform
grid and changes no weights.

```bash
python -m evaluation.evaluate_checkpoints \
  --run-dir $OUTPUTS_ROOT/<run>/seed_0 \
  --inference-mode all \
  --checkpoint-mode both
```

|  | `node_score` | `subspace_norm` |
|---|---|---|
| **no transform** | rank each node by its own coordinate | rank it by the L2 norm over its ancestors+self+descendants subspace |
| **HCC projection** | `hcc_node_score` | `hcc_subspace_norm` |

Each checkpoint's own inference is one cell — `node_score` for
H-CAST/HRN/LH-DNN/HT-CapsNet, `subspace_norm` for native Hier-COS, and the
`hcc_`-prefixed cell of the same readout for an HCC-trained run. That cell is
recorded as `native_inference_mode` and is the paired reference for the others.
Results are written to `posthoc_inference_test_metrics.yaml` inside the run
directory and preserved unless `--overwrite` is passed, so the notebooks in Step
5 reuse the work. Full contract: [evaluation/README.md](evaluation/README.md).

### Step 5 — tables and figures

The notebooks read the run directories and emit the reported tables and figures.
They do not train.

| Notebook | Produces |
|---|---|
| `notebooks/model_comparison_all_datasets.ipynb` | cross-family baseline tables |
| `notebooks/model_analysis/*.ipynb` | per-family training dynamics and diagnostics |
| `notebooks/inference_analysis/*.ipynb` | the inference-grid study (see its [README](notebooks/inference_analysis/README.md)) |
| `notebooks/tradeoff_analysis/*.ipynb` | accuracy–consistency trade-off figures |
| `notebooks/lex_gradient_conflict.ipynb` | lexicographic gradient-conflict analysis |
| `notebooks/datasets_analysis.ipynb` | dataset and taxonomy statistics |

Each notebook exposes its run selection in the setup cell; re-run the discovery
cell and everything below it after changing that selection. Figures are written
under `$OUTPUTS_ROOT/analysis/`, never into the repository.

Two rules govern reading the results. Compare top-down rows only against the
top-down-selected checkpoint and independent rows only against the
independent-selected checkpoint — the two best epochs differ. And use
`test_metrics.yaml` for final numbers, `run_log.jsonl` only for epoch dynamics.

## Outputs

Each seed directory under `$OUTPUTS_ROOT/<experiment>/seed_<n>/` contains:

```text
latest.pt              rolling checkpoint, used for resume
best_topdown.pt        best under top-down decoding
best_independent.pt    best under independent decoding
config_resolved.yaml   the fully resolved config actually used
run_log.jsonl          per-epoch losses, metrics and diagnostics
test_metrics.yaml      final test metrics, per decoding mode
```

`test_metrics.yaml` records the checkpoint, epoch, primary metric and full
selection key for each mode. Version-2 checkpoints store `best_metrics` for
compatibility and `best_selection_keys` for exact resume.

Run logs also carry model-specific diagnostics, documented key by key in
[HCC](docs/HCC_DIAGNOSTIC_LOGS.md) and
[gradient/parameter/lexicographic](docs/GRADIENT_PARAM_DIAGNOSTIC_LOGS.md)
glossaries.

## Metrics, decoding and checkpoint selection

All metrics are ratios in `[0, 1]`:

| Metric | Meaning | Direction |
|---|---|---|
| per-level top-1 accuracy | accuracy at one hierarchy level | higher is better |
| weighted AP (`weighted_ap_*`) | class-count-weighted mean of per-level accuracies | higher is better |
| FPA | exact full-path accuracy | higher is better |
| AHD | average prefix/LCA-equivalent hierarchy distance | lower is better |
| TICE | taxonomy inconsistency rate | lower is better |

**Independent** decoding takes an argmax at every level. **Top-down** decoding
picks the coarse class first, then restricts each later argmax to children of
the selected parent. Never mix the two in one comparison.

Checkpoints are ranked by the exact tuple `(FPA, -TICE, weighted_AP)`, compared
lexicographically rather than packed into a single float, so no TICE change can
override a real FPA improvement. If hierarchy metrics are unavailable,
deepest-level accuracy is the sole primary value.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q datasets models train gridsearch notebooks scripts docs
for f in scripts/*.sh scripts/*/*.sh; do bash -n "$f"; done
git diff --check
```

The suite covers the ported capsule equations and attention shapes, the taxonomy
mask, the HRN objective and schedule, lexicographic projection, the Hier-COS
advantage and the model presets. It checks equations, shapes and required
properties — it does not show that a trained model is numerically identical to a
published checkpoint.

## Documentation

Thesis text (LaTeX, in `docs/`):

- `03-nonstandard.tex`, `04-methodology.tex` — theory and methods
- `05-experiments.tex` — protocol, baseline comparability and all results
- `appendiceA.tex`, `thesis_datasets_section.tex`, `thesis_appendix_figures.tex`

Implementation reference (Markdown, in `docs/`):

- [Presets and mechanism configuration](docs/PRESETS.md)
- [Repository map](docs/FILE_DOCUMENTATION.md)
- [Lexicographic mode: per-model adaptation and constraints](docs/LEX_MODEL_ADAPTATION.md)
- [HCC diagnostic keys](docs/HCC_DIAGNOSTIC_LOGS.md)
- [Gradient, parameter and lexicographic diagnostic keys](docs/GRADIENT_PARAM_DIAGNOSTIC_LOGS.md)

Planning (working documents, not a description of finished work):

- [Experiment matrix](docs/experiment_matrix.md) — the trial space by research question
- [Hier-COS run checklist](docs/HIERCOS_RUN_CHECKLIST.md) — outstanding run queue

Model fidelity against the published sources is discussed in
`docs/05-experiments.tex`, section *Baseline comparability*, which supersedes the
earlier standalone fidelity notes.
