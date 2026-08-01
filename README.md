# Lexicographic Deep Learning for Hierarchical Image Classification

This repository is a unified PyTorch framework for a master-thesis study of
hierarchical image classification. It compares H-CAST, LH-DNN, HT-CapsNet,
HRN, and Hier-COS under one training and evaluation lifecycle, and adds
controlled HCC, gradient-space lexicographic, and orthonormal-frame studies.

The main entrypoint is:

```bash
python -m train.train --config <config.yaml> [key=value ...]
```

## Corrected unified protocol

All shipped runnable presets are tagged:

```yaml
runtime:
  protocol: corrected_unified_v1
```

The protocol fixes the comparison rules that are shared across model families:

- one canonical label mapping and taxonomy is built from training/authoritative
  metadata and reused for validation and test;
- CIFAR-100 uses the 8 → 20 → 100 B-CNN hierarchy;
- CUB uses 13 orders → 38 families → 200 species;
- FGVC-Aircraft uses the official 30 manufacturers → 70 families → 100
  variants annotations;
- validation and test never drop incomplete batches;
- checkpoints are selected on validation data, separately for top-down and
  independent decoding;
- selection is exactly lexicographic: higher FPA, then lower TICE, then higher
  weighted AP;
- final top-down and independent test rows come from their corresponding best
  checkpoints.

These choices intentionally differ from original repositories that drop
evaluation batches, choose checkpoints on test data, or use a different
hierarchy depth. Existing historical results are not silently relabeled as
`corrected_unified_v1`.

## Supported matrix

| Model id | CIFAR-100 | CUB-200 | Aircraft | Main qualification |
|---|---:|---:|---:|---|
| `hcast` | yes | yes | yes | upstream core plus local HCC/lex extensions |
| `lhdnn` | yes | yes | yes | paper-derived; CUB/Aircraft are extrapolations |
| `ht_capsnet` | yes | yes | yes | TensorFlow-to-PyTorch port; Aircraft extrapolation |
| `hrn` | yes | yes | yes | exactly three levels; CIFAR extrapolation |
| `hiercos` | yes | yes | yes | fixed-frame core; local three-level protocol |

“Supported” means that a runnable preset exists. It does not mean that the
dataset/model pair was reported by the original paper.

## Installation

Use Python 3.10 or newer and PyTorch 2.0 or newer (HT-CapsNet uses native
scaled-dot-product attention). Install a PyTorch/torchvision build appropriate for
the machine first, then install the repository dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

PyTorch is intentionally not pinned in `requirements.txt` because CUDA builds
are machine-specific.

Copy the environment template and edit the paths:

```bash
cp .env.example .env
```

The relevant variables are:

```text
CIFAR100_ROOT=/path/to/cifar100
CUB200_ROOT=/path/to/CUB_200_2011
AIRCRAFT_ROOT=/path/to/fgvc-aircraft-2013b
OUTPUTS_ROOT=/path/to/outputs
TRAIN_DEVICE=cuda
```

Existing process environment variables take precedence over `.env`.

## Quick start

Run H-CAST on CIFAR-100:

```bash
python -m train.train --config configs/hcast/hcast_cifar100.yaml
```

Run HRN on CUB:

```bash
python -m train.train --config configs/hrn/hrn_cub200.yaml
```

Run one stopped training stage without changing the scheduler horizon:

```bash
python -m train.train \
  --config configs/hcast/hcast_cifar100.yaml \
  train.stop_epoch=1 \
  train.output_dir=/scratch/$USER/outputs/smoke/hcast_cifar100
```

Resume only from a configuration-compatible checkpoint:

```bash
python -m train.train \
  --config configs/hcast/hcast_cifar100.yaml \
  train.resume=/path/to/latest.pt
```

Strict resume validation permits only `train.resume`, `train.output_dir`, and
`train.stop_epoch` to differ. Use the checkpoint’s saved
`config_resolved.yaml` when resuming a historical protocol.

## Configuration rules

Runnable configs use these required top-level sections:

```text
model, dataset, dataloader, train, optim, scheduler, runtime
```

Optional top-level sections are `hcc` and `orthonormal_plugin`. OmegaConf
environment interpolation and command-line dotlist overrides are resolved
before validation.

Configuration is fail-fast:

- unknown keys are rejected;
- level-name count must equal `dataset.hierarchy_depth`;
- model/dataset/HCC/plugin/lexicographic incompatibilities are rejected;
- `dataloader.drop_last_eval: true` is rejected by the corrected protocol;
- explicit annotation paths are mandatory and missing files fail at dataset
  construction;
- HT-CapsNet, LH-DNN, Hier-COS, and plugin paths require a complete taxonomy.

The commented fragments under `configs/templates/` document the accepted
fields. They are not standalone runnable experiments.

## Presets

### H-CAST

Base:

- `configs/hcast/hcast_cifar100.yaml`
- `configs/hcast/hcast_cub200.yaml`
- `configs/hcast/hcast_aircraft.yaml`

HCC:

- `configs/hcast/hcast_hcc_cifar100.yaml`
- `configs/hcast/hcast_hcc_cub200.yaml`
- `configs/hcast/hcast_hcc_aircraft.yaml`

Explicit gradient-space lexicographic training:

- `configs/hcast/hcast_lex_cifar100.yaml`
- `configs/hcast/hcast_lex_cub200.yaml`
- `configs/hcast/hcast_lex_aircraft.yaml`

HCC is an output-space affine hierarchy constraint. It changes the objective’s
logits but does not explicitly project parameter gradients. Explicit
lexicographic training is enabled by `train.lexicographic.enabled` and projects
lower-priority gradients.

### LH-DNN

- `configs/lhdnn/lhdnn_cifar100.yaml`
- `configs/lhdnn/lhdnn_cub200.yaml`
- `configs/lhdnn/lhdnn_aircraft.yaml`

The CIFAR-100 preset uses the paper’s large topology and 15-epoch schedule.
CUB and Aircraft retain that topology but reduce the 14×14 final feature map
to 2×2 with deterministic 7×7 average pooling. For this divisible geometry it
is equivalent to 2×2 adaptive average pooling, while supporting strict
deterministic CUDA backward. It preserves the paper’s pre-head geometry without
creating an unintended ~51-million-parameter dense layer at 224 px. Those two
presets remain local extrapolations.

### HT-CapsNet

- `configs/capsnet/capsnet_cifar100.yaml`
- `configs/capsnet/capsnet_cub200.yaml`
- `configs/capsnet/capsnet_aircraft.yaml`

The presets use the upstream runner’s 200-epoch horizon, taxonomy temperature
`0.5`, 16×32 Keras-shaped attention, per-example MixUp, the exact source
exponential schedule, deterministic execution, capsule margin loss, and
next-batch dynamic level weights. CIFAR uses 32 px; CUB and Aircraft use 64 px.
The CUB preset deliberately retains this repository's 13/38/200 taxonomy, so
it is not an exact reproduction of the paper's 39/123/200 run. Aircraft is a
64 px extrapolation from the paper datasets. `train.resume` is empty by
default; runs never silently reuse an old checkpoint.
Older local HT-CapsNet checkpoints predate the corrected attention projections
and loss-weight buffer and must not be resumed for these fidelity runs.

### HRN

- `configs/hrn/hrn_cifar100.yaml`
- `configs/hrn/hrn_cub200.yaml`
- `configs/hrn/hrn_aircraft.yaml`

HRN supports exactly three levels. CUB and Aircraft preserve the upstream
ResNet-50/RFM architecture, 448 px preprocessing, tree loss, leaf CE, and
trunk LR scaling. The unified protocol deliberately evaluates every sample and
selects checkpoints on validation data. CIFAR-100 is a dataset-native 32 px
extrapolation for comparability with the other CIFAR presets; it is not an HRN
paper setting. All HRN presets are full-label only, and requested ImageNet
initialization never falls back to random weights.

### Hier-COS

- `configs/hiercos/hiercos_cifar100.yaml`
- `configs/hiercos/hiercos_cub200.yaml`
- `configs/hiercos/hiercos_aircraft.yaml`

Baseline presets default to the upstream-aligned `model.loss: kl_reg` and
`model.weight_mode: kl_leaf`. The local
`global_softmax_ce_reg` and `level_softmax_ce_reg` modes expose three
differentiable level objectives and are selected explicitly by lexicographic
runner overrides. CUB is an extrapolation; CIFAR uses this repository’s
three-level hierarchy instead of the upstream full-depth protocol.

The optional `model.projection.enabled: true` path gives each level an
LH-DNN-style projected learnable FC head. It concatenates the three head
outputs and applies an identity or per-level block-diagonal frozen Hier-COS
frame to the combined vector; a dense global frame is rejected because it
would mix the independent LH-DNN branches.
The projection retains the complete transformation, including its PReLU
activations and both residual skips in `full` mode. By default, the level
branches read the transform output directly and the projection matrix stacks
the preceding detached head weights, making `A` shared by the whole batch.
Set `model.projection.rho_enabled: true` to insert a shared channel-wise PReLU
before the level heads and use the LH-DNN form
`A[b] = W_previous * rho_prime(k[b])`; this makes the projection matrix
sample-dependent. The launcher names this variant `projection_rho` and uses
plain `projection` for the direct-head form.
Set `model.projection.advantage_enabled: true` to additionally propagate
detached parent-class logits as LH-DNN advantage baselines. This path requires
`model.loss: level_softmax_ce_reg`.

## Dataset behavior

Native metadata is used unless a config explicitly supplies annotations:

- CIFAR-100 reads official fine/coarse labels from the Python archive and adds
  the published B-CNN 8-to-20 edge.
- CUB reads common train/test folders or official `images.txt`,
  `image_class_labels.txt`, and `train_test_split.txt`; order/family parents
  come from the retained H-CAST mapping.
- Aircraft accepts only a complete official download and joins the parallel
  variant/family/manufacturer files for every official split.

Every row must provide an existing image and exactly the configured number of
non-negative integer labels. A child may have only one parent. Validation/test
labels that are absent from the canonical training label space are rejected
instead of being remapped independently.

## Metrics and decoding

Metrics are reported as ratios in `[0, 1]`:

- per-level top-1 accuracy: higher is better;
- weighted AP (`weighted_ap_*`): H-CAST’s class-count-weighted mean of
  per-level top-1 accuracies; higher is better;
- FPA: exact full-path accuracy; higher is better;
- AHD: average prefix/LCA-equivalent hierarchy distance; lower is better;
- TICE: taxonomy inconsistency rate; lower is better.

Independent decoding takes an argmax at every level. Top-down decoding first
chooses the coarse class and restricts each later argmax to children of the
selected parent. Do not mix decoding modes in comparisons.

### Checkpoint-only inference comparison

Existing checkpoints can be tested without retraining using the tools in the
top-level `evaluation/` package:

```bash
INFERENCE_MODE=both  # normal | hiercos | node_softmax | both
python -m evaluation.evaluate_checkpoints \
  --run-dir /scratch/$USER/outputs/<run>/seed_0 \
  --inference-mode "$INFERENCE_MODE"
```

For H-CAST, HRN, and other non-Hier-COS models, `both` compares normal
inference with post-hoc identity-frame Hier-COS distance inference. For native
Hier-COS, `both` compares its normal distance inference with direct values from
one global `softmax(abs(node_logits))`, selected by hierarchy level without
constructing subspace scores.

By default the command evaluates both `best_topdown.pt` and
`best_independent.pt` on the test split. Use
`--checkpoint-mode topdown` or `--checkpoint-mode independent` to select only
one. Results are saved to `posthoc_inference_test_metrics.yaml` inside the run
directory; an existing file is preserved unless `--overwrite` is passed.

The post-hoc rule concatenates the model's native raw logits in taxonomy-node
order, assumes an identity orthonormal frame, computes the Hier-COS L2 norm of
each ancestors+self+descendants subspace, and predicts directly from those raw
scores. It does not load an orthonormal plugin, change the loss, update model
parameters, or claim that the source model was trained in a Hier-COS space.

The paired multi-model analysis notebook is
`notebooks/posthoc_hiercos_inference_comparison.ipynb`. It runs matched H-CAST,
HRN, and Hier-COS baselines across CIFAR-100, CUB, and Aircraft; preserves
top-down/independent checkpoint selection; and reports the mean, sample
standard deviation, and seed count for absolute metrics and direction-aware
paired gains. Its execution cell invokes
`python -m evaluation.evaluate_checkpoints` directly for every completed seed.
Each seed's result remains in its own run directory beside its checkpoints.

Checkpoint ranking uses the exact tuple:

```text
(FPA, -TICE, weighted_AP)
```

If hierarchy metrics are unavailable, deepest-level accuracy is the sole
primary value. No decimal packing or tolerance can allow TICE to override a
real FPA improvement.

## Outputs

Each seed directory contains:

```text
latest.pt
best_topdown.pt
best_independent.pt
config_resolved.yaml
run_log.jsonl
test_metrics.yaml
```

Version-2 checkpoints store `best_metrics` for compatibility and
`best_selection_keys` for exact resume behavior. `test_metrics.yaml` records
the checkpoint, epoch, primary metric, full selection key, and final metrics
for each decoding mode.

Run logs also contain model-specific loss and diagnostic fields. See:

- [HCC diagnostic keys](docs/HCC_DIAGNOSTIC_LOGS.md)
- [gradient, parameter, and lexicographic diagnostic keys](docs/GRADIENT_PARAM_DIAGNOSTIC_LOGS.md)

## Experiment launchers

Launchers are under `scripts/hcast/`, `scripts/lhdnn/`, `scripts/hrn/`, and
`scripts/hiercos/`. They support:

```text
DRY_RUN=1
NUM_RUNS=<positive integer>
BASE_SEED=<first training seed>
SPLIT_SEED=<fixed dataset split seed>
MAX_PARALLEL=<parallel processes>
MAX_RESUME_RETRIES=<retry count>
```

Matrix variables are whitespace-separated and validated:

```bash
DATASETS="cifar100 aircraft" \
START_EPOCHS="0 80" \
DRY_RUN=1 \
scripts/hcast/run_hcast_lex_grid.sh
```

Hier-COS and plugin launchers similarly accept `LEX_PROJECTION_MODES` and
`TRANSFORM_MODES`. Each launcher prints the selected matrix. Narrow defaults
remain narrow so invoking a script cannot unexpectedly start the full
expensive grid.

Run the paper-aligned LH-DNN CIFAR-100 preset and the two explicitly
extrapolated large-image presets with:

```bash
scripts/lhdnn/run_lhdnn_baselines.sh
```

Run the Hier-COS model with projected learnable level heads and the optional
shared PReLU/rho derivative with:

```bash
scripts/hiercos/run_hiercos_lhdnn_projection.sh
```

Projected Hier-COS uses `model.projection.feature_dim` for the shared backbone
and transform width (model default `256`; the projection launcher defaults to
`0`, which selects the dataset taxonomy width).
It applies only when `model.projection.enabled: true`; with the projection off
the width stays at `sum(num_classes_per_level)`. Each level head reduces that feature
vector to its class count before the outputs enter the taxonomy-width fixed
frame. Set it to `0` to restore the previous taxonomy width,
`sum(num_classes_per_level)`. The launcher exposes the setting as
`FEATURE_DIM`, so `FEATURE_DIM=0` works across datasets with different
hierarchy widths.

## Verification

The repository uses the standard library test runner:

```bash
python -m unittest discover -s tests -v
```

Useful fast checks:

```bash
python -m compileall -q datasets models train gridsearch notebooks scripts docs
for f in scripts/*.sh scripts/*/*.sh; do bash -n "$f"; done
git diff --check
```

The test suite covers official hierarchies, cross-split label stability,
strict config parsing, metric/selection oracles, source-equation contracts,
checkpoint compatibility, launcher matrices, and documentation paths.

## Documentation

- [Repository map](docs/FILE_DOCUMENTATION.md)
- [Pinned upstream fidelity and divergence log](docs/model_repo_differences.md)
- [Dated correctness audit](docs/REPOSITORY_AUDIT.md)
- [HCC/H-CAST research synthesis](docs/hcc_hcast_research_report.md)
- [Detailed HRN and Hier-COS alignment notes](docs/hrn_hiercos_alignment.md)
