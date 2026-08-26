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

The optional top-level section is `hcc`. OmegaConf
environment interpolation and command-line dotlist overrides are resolved
before validation.

Configuration is fail-fast:

- unknown keys are rejected;
- level-name count must equal `dataset.hierarchy_depth`;
- model/dataset/HCC/lexicographic incompatibilities are rejected;
- `dataloader.drop_last_eval: true` is rejected by the corrected protocol;
- explicit annotation paths are mandatory and missing files fail at dataset
  construction;
- HT-CapsNet, LH-DNN, and Hier-COS require a complete taxonomy.

The commented fragments under `configs/templates/` document the accepted
fields. They are not standalone runnable experiments.

## Presets

### H-CAST

`configs/hcast/` holds only the three base presets:

- `configs/hcast/hcast_cifar100.yaml`
- `configs/hcast/hcast_cub200.yaml`
- `configs/hcast/hcast_aircraft.yaml`

The HCC and lexicographic variants are not separate configs. Their launchers
start from these base presets and add the whole variant block as CLI overrides:

- HCC: `scripts/hcast/run_hcast_hcc.sh` adds the `hcc.*` block
  (`enabled`, `eps`). HCC is a binary switch: when `hcc.enabled: true` the
  projection is fully applied from the first batch onwards, with no onset,
  alpha-schedule, or temperature knobs.
- Lexicographic: `scripts/hcast/run_hcast_lex.sh` adds the
  `train.lexicographic.*` block and the required `model.loss.globalkl=false`.

HCC is an output-space affine hierarchy constraint. It changes the objective’s
logits but does not explicitly project parameter gradients. Explicit
lexicographic training is enabled by `train.lexicographic.enabled` and projects
lower-priority gradients. Native lexicographic training is supported for
H-CAST, HT-CapsNet, HRN, and decomposed-loss Hier-COS. LH-DNN is excluded.
`train.gradient_blocks` selects the exact gradient-support blocks used by
both diagnostics and projection. Blocks use compact support names (`p123`,
`p12`, `p23`, `p1`, and so on); omitting the field preserves the historical
`[p123, p12, p1]` behavior. HT-CapsNet launchers select
`[p123, p23, p3]` for its coarse-to-fine capsule cascade.

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

The presets port the released TensorFlow architecture and callback behavior:
one squashed primary-capsule tensor is reshaped at later levels, parent capsule
lengths are softmaxed before taxonomy masking, and dynamic loss weights use the
released callback's parentheses. Experimental hyperparameters follow the paper except for
the training budget, taxonomy temperature `0.5`, 16×32
Keras-shaped attention with rank-three Keras Glorot initialization, per-example
MixUp, Keras 2.8 Adam update, deterministic
execution, capsule margin loss, and next-batch dynamic level weights. The
presets run 100 epochs rather than the paper's 200, under the repository-wide
budget shared by every family except LH-DNN; the learning-rate decay rate is
unchanged, so the schedule is truncated rather than rescaled.
Checkpointing deliberately departs from the released finest-output monitor and
uses the repository-wide `(FPA, -TICE, weighted_AP)` ranking, so every family is
selected by the same rule. The last partial training batch is retained. The EfficientNet preset pins
the Keras-compatible
`tf_efficientnet_b7.aa_in1k` conversion and restores Keras BatchNorm/drop-connect
training semantics. CIFAR uses native 32 px inputs and split-wide scalar
standardization; CUB uses the source path-loader sequence (resize to 512 px,
then to the paper's 64 px input) and batch-wide scalar standardization. Aircraft
uses the same path preprocessing as an explicit local extrapolation.
The CUB preset deliberately retains this repository's 13/38/200 taxonomy, so
it is not an exact reproduction of the paper's 39/123/200 run. Aircraft is a
64 px extrapolation from the paper datasets. `train.resume` is empty by
default; runs never silently reuse an old checkpoint.
Older local HT-CapsNet checkpoints predate the source-aligned primary-capsule,
routing-mask, and dynamic-weight behavior as well as corrected attention
initialization, backbone semantics, optimizer, and loss-weight buffering. They
must not be resumed for these fidelity runs. The paper and released TensorFlow
file contain material contradictions; the exact source-reproduction
alternatives and preprocessing/LayerNorm diagnostics are recorded in
`docs/model_repo_differences.md`.
Native lexicographic HT-CapsNet uses the same three capsule margin losses. The
lex launcher selects `model.loss.weight_mode: none`, which is the existing
unit-weight mode; the paper-baseline configs retain their dynamic weighting.

### HRN

- `configs/hrn/hrn_cifar100.yaml`
- `configs/hrn/hrn_cub200.yaml`
- `configs/hrn/hrn_aircraft.yaml`

HRN supports exactly three levels. CUB and Aircraft preserve the upstream
ResNet-50/RFM architecture, 448 px preprocessing, tree loss, leaf CE, and
trunk LR scaling. The unified protocol deliberately evaluates every sample and
selects checkpoints on validation data. CIFAR-100 is an explicit HRN-WRN-28-8
adaptation: it trains from scratch on native 32 px inputs and uses the same
CIFAR-style backbone geometry as Hier-COS, preserving an 8 x 8 feature map for
HRN's RFM branches. It is not an HRN paper setting. All HRN presets are
full-label only, and requested ResNet-50 ImageNet initialization never falls
back to random weights.
When omitted, `model.loss` defaults to the paper-aligned `native` objective
(leaf-observed joint tree marginal plus leaf CE). The local
`model.loss: level_conditional` mode splits that same objective into three
conditional negative log-likelihoods - the coarse subtree, the middle subtree
given the coarse one, and the leaf given the middle one - with the unit-weight
leaf CE added to the fine term. The three terms sum to the `native` total and
carry the same gradient, so without lexicographic projection the two modes train
identically; the split exists so lexicographic mode has three level objectives
to project. HRN lexicographic training requires this mode.

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

Native Hier-COS additionally supports two taxonomy-size-derived modes. With
`model.weight_mode: cumulative_branching`, level weights are
`C_l^beta / sum_k C_k^beta`, where `beta` is configured by
`model.weight_beta` and defaults to `0.5`. With
`model.weight_mode: marginal_branching`, the unnormalised scores are
`[1, C_1/C_0, ..., C_l/C_(l-1)]` and are normalised to sum to one. These two
mode weights the path/CE component while level regularisation remains
unweighted.

The optional `model.projection.enabled: true` path gives each level an
LH-DNN-style projected learnable FC head. It concatenates the three head
outputs and applies an identity or per-level block-diagonal frozen Hier-COS
frame to the combined vector; a dense global frame is rejected because it
would mix the independent LH-DNN branches.
The projection retains the complete transformation, including its PReLU
activations and both residual skips in `full` mode. It then inserts a shared
channel-wise PReLU before the level heads and builds the projection matrix in
the LH-DNN form `A[b] = W_previous * rho_prime(k[b])`, so the protected
subspace is sample-dependent. The PReLU derivative is part of the LH method and
is always applied; there is no config switch for it, and the earlier
batch-shared `A` variant has been removed. The launcher names these runs
`projection`.
Set `model.projection.advantage_enabled: true` to additionally propagate
detached parent-class logits as LH-DNN advantage baselines. This path requires
`model.loss: level_softmax_ce_reg`.

Direct subspace-norm supervision is enabled with
`train.subspace_supervision.enabled: true`. It replaces the model's native loss
with the mean of the per-level soft cross-entropies over the taxonomy-subspace
norms. Training consumes `subspace_scores_per_level` directly and evaluation
ranks the same scores, so the objective is aligned with the deployed
subspace-norm argmax.

Ground truth is **not** a one-hot label. A subspace spans a node's ancestors,
itself and its descendants, so two classes sharing an ancestor share those
coordinates: the score of a sibling of the correct class is bounded below by the
energy on the shared ancestors and a one-hot target is unreachable. Pushing
toward it is minimized by putting all energy on the leaf coordinate, which
collapses the coarse node coordinates the readout is supposed to accumulate.
The target used instead is the profile that the intended geometry induces --
unit energy spread evenly over the ground-truth path, pushed through the same
subspace masks -- so class `c` is targeted at the square root of the fraction of
that path its subspace contains. It depends on the taxonomy alone and is read
off by the leaf label.

`model.weight_mode` is spent where the native Hier-COS softmax losses spend it:
on the **scalarisation**. Its level weights, normalized to sum to one, are the
per-level coefficients of the total, and the target geometry carries no
weighting at all. Separating the two roles leaves `tau` as the only control over
the target's margins and keeps `model.weight_mode` meaning the same thing here
as in the baseline the arm is measured against; under `equal` the scalarisation
is the uniform mean, so the loss stays on the scale of the Hier-COS baselines'
convex combination and learning rates transfer. `loss_weight_level_*` logs the
coefficients in use and `loss_level_*` each level's contribution to the total;
the `subspace_*_level_*` diagnostics stay unweighted, so the geometry is
comparable across weight modes.

Score and target are compared through `softmax(./tau)` on each side after the
scores are divided by the norm of the node coordinate vector -- **one scale
shared by every level**, not one per level. Under the intended geometry the
correct class scores `||u||` at every level and the target is already 1 there,
so the shared scale puts both sides in the same units while giving up only the
single global degree of freedom the score map is homogeneous in. Normalizing per
level instead frees one scale per level and leaves the node energies
unidentified: the recovered geometry then drifts off the ground-truth path
whenever the target is not sharply peaked. `tau -> 0` recovers the unreachable
one-hot target; larger values flatten it.

The loss dispatch is capability-based rather than tied to `model.name`. A model
opting in must return `subspace_scores_per_level` (`[B, C_l]`),
`subspace_path_overlap_by_level` (`[num_leaf, C_l]` integer counts of how many
levels of a leaf's path each class subspace contains), `node_logits` (`[B, N]`)
for the shared scale, and the fields the level-weight resolution needs. The
mechanism rejects soft targets, mixup/cutmix, HCC, lexicographic training, and the
Hier-COS LH projection. Its loss log includes the per-level soft
cross-entropies, the level coefficients, unnormalized score norms,
and `subspace_target_kl*` -- the residual above the target entropy, which is
zero at the intended geometry and is the part the optimizer can remove. The
accepted config block is:

```yaml
train:
  subspace_supervision:
    enabled: true
    tau: 0.25                 # > 0; target/prediction temperature
    eps: 1.0e-12              # numerical epsilon for the shared scale
```

There is no loss selector and no level-weight setting in this block. Enabling
the mechanism always selects the tempered soft cross-entropy against the induced
profile, and the level weights come from **`model.weight_mode`**, which the arm
spends on the target geometry alone. The target is rebuilt from those weights
each step rather than tabulated, so any `model.weight_mode` gives a target the
uniformly averaged level losses are consistent with.

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
INFERENCE_MODE=all  # node_score | subspace_norm | hcc_node_score | hcc_subspace_norm | both | all
python -m evaluation.evaluate_checkpoints \
  --run-dir /scratch/$USER/outputs/<run>/seed_0 \
  --inference-mode "$INFERENCE_MODE"
```

An inference rule is a readout — `node_score` (rank each taxonomy node by its
own coordinate) or `subspace_norm` (rank it by the L2 norm over its
ancestors+self+descendants subspace) — optionally preceded by the HCC affine
hierarchy projection, giving a 2x2 grid. Both readouts consume the same node
coordinates: the fixed-layer `node_logits` for native Hier-COS, the native
per-level scores for classifier-head models, which `subspace_norm` treats as
coordinates in an identity taxonomy frame. `all` evaluates the four cells from
one shared forward pass; `both` evaluates the two untransformed readouts.

Every cell is defined for every model, and each checkpoint's own inference is one
of them: `node_score` for H-CAST/HRN/LH-DNN/HT-CapsNet, `subspace_norm` for
native Hier-COS, and the `hcc_`-prefixed cell of the same readout for a run
trained with `hcc.enabled: true`. That cell is recorded as
`native_inference_mode` and used as the paired reference for the others. Nothing
here changes the loss or updates parameters.

Two properties shape the deltas: `hcc_node_score` shifts each sibling group by
one constant, so it cannot change any top-down metric for a signed readout (it
can for Hier-COS, whose readout takes the magnitude), and `subspace_norm`
squares its inputs, so it discards the sign of a signed logit. The previous mode
names — `normal`, `hiercos`, `node_softmax`, `hcc` — still work and map onto the
grid, which also lets results computed before the rename be read without
recomputation. See `evaluation/README.md` for the full contract.

By default the command evaluates both `best_topdown.pt` and
`best_independent.pt` on the test split. Use
`--checkpoint-mode topdown` or `--checkpoint-mode independent` to select only
one. Results are saved to `posthoc_inference_test_metrics.yaml` inside the run
directory; an existing file is preserved unless `--overwrite` is passed.

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
- [lexicographic mode per-model adaptation, constraints, and quirks](docs/LEX_MODEL_ADAPTATION.md)

## Experiment launchers

Launchers are under `scripts/hcast/`, `scripts/lhdnn/`, `scripts/capsnet/`,
`scripts/hrn/`, and `scripts/hiercos/`. They support:

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
LEX_PROJECTION_MODE=fine_first \
DRY_RUN=1 \
scripts/hcast/run_hcast_lex.sh
```

The always-on HCC launchers are:

```bash
scripts/hcast/run_hcast_hcc.sh
scripts/capsnet/run_ht_capsnet_hcc.sh
scripts/hrn/run_hrn_hcc.sh
scripts/hiercos/run_hiercos_hcc.sh
```

Each defaults to all three datasets. HCC output directories use
`<model>_<dataset>[_<special-setting>]_hcc`; for example,
`hcast_cifar100_hcc`, `capsnet_cub200_hcc`,
`hrn_aircraft_level_conditional_hcc`, and
`hiercos_cifar100_global_softmax_ce_reg_hcc`.

Hier-COS launchers similarly accept `LEX_PROJECTION_MODES` and
`TRANSFORM_MODES`. Each launcher prints the selected matrix. Narrow defaults
remain narrow so invoking a script cannot unexpectedly start the full
expensive grid.

Run Hier-COS with direct subspace supervision using:

```bash
DATASETS=cifar100 BASE_SEED=0 NUM_RUNS=1 \
scripts/hiercos/run_hiercos_subspace.sh
```

The launcher sets `alpha=0`, enables the tempered soft cross-entropy against the
induced path-energy profile, and forces the Hier-COS LH projection, HCC,
mixup/cutmix, label smoothing, and lexicographic training off. Override
`SUBSPACE_TAU` (default `0.25`) to sweep the temperature; level weights follow
`model.weight_mode`. Run directories are named
`hiercos_<dataset>_subspace`; earlier runs under that name used the hard-label
or normalized-profile objectives and are not comparable.

Run native HT-CapsNet baselines on the three dataset configurations with:

```bash
NUM_RUNS=3 scripts/capsnet/run_ht_capsnet_baselines.sh
```

The runner preserves each config's dynamic margin-loss weighting. The model
follows the released TensorFlow behavior and the experiment values follow the
paper; CUB uses this repository's unified taxonomy, and Aircraft is a local
extrapolation. Use `DATASETS`, `NUM_RUNS`, `BASE_SEED`, or `SPLIT_SEED` to select
a reproducible subset.

Run native HT-CapsNet and HRN lexicographic training on the three baseline
dataset configurations with:

```bash
scripts/capsnet/run_ht_capsnet_lex.sh
scripts/hrn/run_hrn_lex.sh
```

Both default to coarse-first. Lexicographic projection is always active for the
whole run. Override `DATASETS` or `LEX_PROJECTION_MODE` to select another
validated run without adding a dataset config. The HT-CapsNet and HRN lex
launchers pass an explicit `train.epochs=100`, which now agrees with their
baseline presets, so baseline-versus-lex comparisons in these two families are
training-budget matched (they remain not compute-matched, since a lexicographic
step is more expensive than a baseline step).
The HRN launcher also enforces hard targets.

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
and transform width (model and projection-launcher default `0`, which selects
the dataset taxonomy width).
It applies only when `model.projection.enabled: true`; with the projection off
the width stays at `sum(num_classes_per_level)`. Each level head reduces that feature
vector to its class count before the outputs enter the taxonomy-width fixed
frame. Keep it at `0` to use the taxonomy width,
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
