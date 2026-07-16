# Hierarchical Image Classification

Unified PyTorch training code for five hierarchical image classifiers:

- `hcast`
- `lhdnn`
- `ht_capsnet`
- `hrn`
- `hiercos`

The main user-facing entrypoint is `python -m train.train`. It handles config loading, training, validation, checkpointing, and final test evaluation from separate top-down and independent best checkpoints.

## What Is In Scope

- One config-driven training pipeline for all supported models
- Dataset adapters for CIFAR-100, CUB-200-2011, FGVC-Aircraft, and iNaturalist 2019 data
- Shared hierarchical metrics and logging
- Optional H-CAST Hierarchical Constraint Cascade (HCC) variants

## Setup

The current development and experiment environment uses:

- Python `3.10.12`
- PyTorch `2.10.0` and torchvision `0.25.0`
- CUDA `12.8` PyTorch wheels on the GPU machine

Other Python and PyTorch versions have not been systematically tested. The virtual environment itself is not committed; it is created with Python's standard `venv` module in the ignored `.venv/` directory.

Create and activate the environment from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the same CUDA build used for the experiments, then install the remaining dependencies:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

For a CPU-only environment, replace the PyTorch command with:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

Verify that the expected interpreter and core packages are active:

```bash
python --version
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

The expected output starts with `Python 3.10.12` and reports PyTorch `2.10.0` and torchvision `0.25.0` (with a platform-specific build suffix such as `+cu128` or `+cpu`). When using the shell runners under `scripts/`, set `PYTHON_BIN=.venv/bin/python` in `.env` so they use this environment even when it is not activated.

Notes:

- `timm` is required for H-CAST and timm-backed schedulers.
- `opencv-contrib-python` is required for H-CAST `segments.mode: seeds`.
- `requirements.txt` currently specifies compatible version ranges rather than a fully frozen cross-platform lock file. Record `python -m pip freeze` with archived experiment artifacts when exact package-level reconstruction is required.

### Local Environment

Create the ignored, machine-local settings file once:

```bash
cp .env.example .env
```

Edit `.env` when dataset or output storage moves. The active presets and runner scripts share these values:

- `CIFAR100_ROOT`, `CUB200_ROOT`, `AIRCRAFT_ROOT`, `INAT19_ROOT`: complete dataset paths
- `OUTPUTS_ROOT`: parent directory for run artifacts, grid searches, and analysis outputs
- `PYTHON_BIN`: Python executable used by shell runners
- `MAX_PARALLEL`, `MAX_RESUME_RETRIES`: shared runner resource/retry defaults
- `NUM_RUNS`: repetitions per experiment condition (default `1`)
- `BASE_SEED`: first training seed; later repetitions use consecutive seeds
- `SPLIT_SEED`: fixed train/validation split seed shared by all repetitions
- `TRAIN_DEVICE`: default config device (`cuda` or `cpu`)

The Python config loader reads `.env` automatically. Shell runners load it through `scripts/load_env.sh`. Variables already exported by the calling process take precedence, and `PROJECT_ENV_FILE=/path/to/file` selects a different env file.

## Quick Start

1. Configure `.env` as described above.
2. Pick a preset config from `configs/`.
3. Run training:

```bash
python -m train.train --config configs/hcast/hcast_cifar100.yaml
```

CLI dotlist overrides are supported:

```bash
python -m train.train --config configs/hrn/hrn_cub200_parity.yaml train.epochs=1 dataloader.batch_size=4
```

There is no separate standalone evaluation CLI at the moment. End-of-run test evaluation is part of `train/train.py`.

## Available Presets

H-CAST:

- `configs/hcast/hcast_cifar100.yaml`
- `configs/hcast/hcast_cub200.yaml`
- `configs/hcast/hcast_aircraft.yaml`
- `configs/hcast/hcast_inat19.yaml`
- `configs/hcast/hcast_hcc_cifar100.yaml`
- `configs/hcast/hcast_hcc_cub200.yaml`
- `configs/hcast/hcast_hcc_aircraft.yaml`

LH-DNN:

- `configs/lhdnn/lhdnn_cifar100.yaml`
- `configs/lhdnn/lhdnn_cub200.yaml`
- `configs/lhdnn/lhdnn_aircraft.yaml`

HT-CapsNet:

- `configs/capsnet/capsnet_cifar100.yaml`
- `configs/capsnet/capsnet_cub200.yaml`
- `configs/capsnet/capsnet_aircraft.yaml`

HRN:

- `configs/hrn/hrn_cifar100_parity.yaml`
- `configs/hrn/hrn_cub200_parity.yaml`
- `configs/hrn/hrn_aircraft_parity.yaml`

Hier-COS:

- `configs/hiercos/hiercos_cifar100.yaml`
- `configs/hiercos/hiercos_cub200.yaml`
- `configs/hiercos/hiercos_aircraft.yaml`
- `configs/hiercos/hiercos_inat19.yaml`

Reusable templates:

- `configs/templates/dataset_template.yaml`
- `configs/templates/hcast_template.yaml`
- `configs/templates/hcc_template.yaml`
- `configs/templates/ht_capsnet_template.yaml`
- `configs/templates/hrn_template.yaml`
- `configs/templates/hiercos_template.yaml`
- `configs/templates/orthonormal_plugin_template.yaml`
- `configs/templates/training_template.yaml`

## Config Structure

Experiment YAMLs use the same top-level sections:

- `model`
- `orthonormal_plugin` (optional)
- `dataset`
- `dataloader`
- `train`
- `optim`
- `scheduler`
- `runtime`

Common fields worth checking first:

- `dataset.root`
- `dataset.annotations.{train,val,test}`
- `dataset.hierarchy_depth`
- `train.output_dir`
- `train.device`
- `train.resume`
- `dataloader.drop_last_train` / `dataloader.drop_last_eval`
- `dataset.transforms.manual.crop_bottom_pixels`
- `model.loss` (Hier-COS: `kl_reg`, `global_softmax_ce_reg`, or `level_softmax_ce_reg`)
- `model.weight_mode` (Hier-COS shared path/CE weighting in both non-lex and lex modes: `equal`, `kl_leaf`, or `kl_coarse`)

Lexicographic upper-bound mode (3-level):

- `train.lexicographic.enabled`: enables gradient-space lexicographic updates
- `train.lexicographic.start_epoch`: internal zero-based epoch index where projected gradients start
- `train.lexicographic.projection_mode`: projection composition mode (`coarse_first`, `fine_first`, `pairwise_orthogonal`)
- projection mode semantics: `coarse_first` prioritizes coarse then mid then fine; `fine_first` reverses that order; `pairwise_orthogonal` enforces sequential pairwise orthogonalization among level gradients
- `train.lexicographic.projection_rule`: projection rule (`orthogonalize_all` or `conflict_only`)
- `train.lexicographic.eps`: projection denominator epsilon
- `train.lexicographic.log_metrics`: logs projection diagnostics under `train_metrics`
- H-CAST requires exactly 3 level losses and `model.loss.globalkl: false`
- Hier-COS lexicographic mode requires per-level losses (`model.loss: global_softmax_ce_reg` or `level_softmax_ce_reg`); plain `model.loss: kl_reg` does not expose differentiable per-level losses. Lexicographic updates honor the configured `model.weight_mode`.
- `orthonormal_plugin.enabled: true` uses the shared post-logit orthonormal taxonomy-frame loss for any model that emits `orthonormal_plugin_scores_per_level`; lexicographic mode requires `orthonormal_plugin.loss` to be `global_softmax_ce_reg` or `level_softmax_ce_reg`.

The config loader supports positional dotlist overrides such as `train.epochs=10` or `optim.lr=1e-4`.

Hier-COS runners expose `LOSS_MODE`, `WEIGHT_MODE`, and `FIXED_FRAME_MODE`. Set
`FIXED_FRAME_MODE=identity` to use the identity frame, or
`FIXED_FRAME_MODE=orthonormal_random` to use the fixed random orthonormal frame:

```bash
LOSS_MODE=level_softmax_ce_reg \
WEIGHT_MODE=kl_leaf \
FIXED_FRAME_MODE=identity \
./scripts/hiercos/run_hiercos_lex_orthogonalize_all.sh
```

Runner scripts are grouped by family under `scripts/hcast/`, `scripts/hrn/`, `scripts/hiercos/`, and `scripts/data/`. Plugin runs use hard targets, so the runners disable MixUp/CutMix and smoothing:

```bash
DRY_RUN=1 ./scripts/hcast/run_hcast_orthonormal_plugin.sh
DRY_RUN=1 ./scripts/hcast/run_hcast_lex_orthonormal_plugin.sh
DRY_RUN=1 ./scripts/hrn/run_hrn_baselines.sh
DRY_RUN=1 ./scripts/hrn/run_hrn_orthonormal_plugin.sh
```

Runner-managed experiments preserve the existing experiment name and store
each repetition in its own seed directory:

```text
${OUTPUTS_ROOT}/<experiment_name>/seed_<train.seed>/
```

For example, this launches three repetitions with training seeds 0, 1, and 2
while keeping the train/validation split fixed:

```bash
NUM_RUNS=3 BASE_SEED=0 SPLIT_SEED=0 \
./scripts/hiercos/run_hiercos_baselines.sh
```

`MAX_PARALLEL` limits the total number of simultaneous jobs across datasets,
experiment variants, and seeds. Each seed directory has independent
checkpoints and retry/resume state.

Historical direct-output folders can be previewed and migrated with:

```bash
python scripts/migrate_single_seed_outputs.py
python scripts/migrate_single_seed_outputs.py --apply
```

The utility only considers immediate `OUTPUTS_ROOT` children that directly
contain both `config_resolved.yaml` and `run_log.jsonl`, and refuses target
collisions.

## Supported Datasets

Strict dataset ids:

- `cifar-100`
- `cub-200-2011`
- `fgvc-aircraft`
- `inat19`

Adapter behavior:

- CIFAR-100 loads through `torchvision.datasets.CIFAR100` and supports `hierarchy_depth: 2` or `3`. The 3-level layout is `super -> coarse -> fine`.
- CUB supports `train/` and `test/` folder splits, `images_split/{train,test}`, or the official `images.txt` / `image_class_labels.txt` / `train_test_split.txt` layout.
- FGVC-Aircraft reads the official `images_variant_{train,val,test,trainval}.txt` files under `data/`-style roots.
- iNat19 uses official iNaturalist 2019 COCO-style JSON or JSON-in-tar annotations with a local 3-level `family -> genus -> species` projection. The active iNat19 configs use fixed Making Better Mistakes / Hier-COS train/val/test manifests over the labeled official `train_val2019` image pool.

Default dataset roots in `.env.example`:

```text
/scratch/g.saggini1/datasets/cifar100
/scratch/g.saggini1/datasets/CUB_200_2011
/scratch/g.saggini1/datasets/fgvc-aircraft-2013b
/scratch/g.saggini1/datasets/inat19
```

To prepare CIFAR-100 from the official Toronto download through torchvision, export the local paths first:

```bash
set -a
source .env
set +a
mkdir -p "$CIFAR100_ROOT"
python - <<'PY'
import os
from torchvision.datasets import CIFAR100
root = os.environ["CIFAR100_ROOT"]
CIFAR100(root=root, train=True, download=True)
CIFAR100(root=root, train=False, download=True)
PY
```

To prepare CUB-200-2011 from CaltechDATA:

```bash
set -a; source .env; set +a
mkdir -p "$(dirname "$CUB200_ROOT")"
cd "$(dirname "$CUB200_ROOT")"
curl -L -o CUB_200_2011.tgz "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
tar -xzf CUB_200_2011.tgz
```

Expected layout:

```text
<CUB200_ROOT>/
  images/
  images.txt
  image_class_labels.txt
  train_test_split.txt
```

To prepare FGVC-Aircraft from Oxford VGG:

```bash
set -a; source .env; set +a
mkdir -p "$(dirname "$AIRCRAFT_ROOT")"
cd "$(dirname "$AIRCRAFT_ROOT")"
curl -L -O https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
tar -xzf fgvc-aircraft-2013b.tar.gz
```

Expected layout:

```text
<AIRCRAFT_ROOT>/
  data/
    images/
    images_variant_train.txt
    images_variant_val.txt
    images_variant_test.txt
    variants.txt
```

To prepare iNaturalist 2019 with the Making Better Mistakes / Hier-COS fixed split:

```bash
set -a; source .env; set +a
mkdir -p "$INAT19_ROOT"
cd "$INAT19_ROOT"

curl -L -O https://ml-inat-competition-datasets.s3.amazonaws.com/2019/train_val2019.tar.gz
curl -L -O https://ml-inat-competition-datasets.s3.amazonaws.com/2019/train2019.json.tar.gz
curl -L -O https://ml-inat-competition-datasets.s3.amazonaws.com/2019/val2019.json.tar.gz
curl -L -o splits_inat19.zip https://github.com/fiveai/making-better-mistakes/raw/master/dataset_splits/splits_inat19.zip

tar -xzf train_val2019.tar.gz
tar -xzf train2019.json.tar.gz
tar -xzf val2019.json.tar.gz

cd -
python scripts/data/prepare_inat19_mbm_splits.py
```

Expected layout:

```text
<INAT19_ROOT>/
  train_val2019/
  train2019.json
  val2019.json
  splits_inat19.zip
  annotations_mbm/
    train.json
    val.json
    test.json
```

`test2019.tar.gz` is not required for this repo because it has no public ground-truth labels. `categories.json.tar.gz` is optional and not required for training. Without `dataset.split_policy: explicit`, the iNat19 adapter can still fall back to splitting official `train2019` into repo train/val and using official `val2019` as repo test.

Official dataset sources:

- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- CUB-200-2011: https://data.caltech.edu/records/65de6-vp158
- FGVC-Aircraft: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
- iNaturalist 2019: https://github.com/visipedia/inat_comp/tree/master/2019
- Making Better Mistakes splits: https://github.com/fiveai/making-better-mistakes

Optional normalized JSON annotations are also supported for all datasets:

```json
{
  "samples": [
    {"image": "relative/path.jpg", "labels": [0, 1, 2]}
  ]
}
```

`labels` must match `dataset.hierarchy_depth`.

## Model-Specific Constraints

- `hcast` requires at least 2 hierarchy levels.
- `hcast` `hcc.enabled: true` requires exactly 3 levels and a valid taxonomy.
- `hcast` `hcc.final_test_only: true` keeps HCC off during train/val and enables it only for final test evaluation from `best_topdown.pt` and `best_independent.pt` (with full projection effect).
- `lhdnn` requires at least 2 levels and always requires taxonomy.
- `ht_capsnet` requires at least 2 levels. The builder also enforces `train.seed` and `runtime.deterministic: true`.
- `hrn` supports exactly 3 levels.
- `hrn` follows upstream HRN for CUB-200 and Aircraft parity presets: ImageNet-pretrained ResNet-50, 1024-d branch bottlenecks, 512-d classifiers, 448 crops from 550x550 resized images, `[0.5, 0.5, 0.5]` normalization, and SGD parameter groups with the ResNet trunk at 0.1x LR.
- `hrn` parity loss does not support mixup/cutmix soft targets, so the shipped HRN presets keep them disabled.
- `hrn_cifar100_parity` is a local extrapolation because the upstream HRN repo does not include CIFAR-100; it keeps this repo CIFAR hierarchy.
- `hiercos` requires taxonomy (`taxonomy.parent_of`) and at least 2 levels.
- `hiercos` does not support mixup/cutmix soft targets. Keep `dataset.transforms.mixup/cutmix: 0.0`.
- `hiercos` uses a single fixed orthonormal Hier-COS frame with taxonomy-driven subspace scores. `model.loss: kl_reg` is the default paper-aligned exact KL + level regularization objective. The lex-ready `global_softmax_ce_reg` and `level_softmax_ce_reg` modes both optimize weighted target CE plus the same level regularizer and use leaf-derived taxonomy paths. They differ only in normalization scope: `global_softmax_ce_reg` uses one softmax across every taxonomy node, while `level_softmax_ce_reg` uses one softmax inside each level. `model.weight_mode` supports `equal` (exact `1/depth` per level), leaf-heavy `kl_leaf`, and reversed/coarse-heavy `kl_coarse`; it defines the target-path distribution for `kl_reg` and weights CE in both decomposed modes. Regularization remains unweighted in every mode. Both decomposed modes log `ce`, `reg`, `ce_level_*`, `reg_level_*`, and exact optimization objectives `loss_level_*`, with `total = sum(loss_level_*) = ce + alpha * reg`. CLI examples: `model.loss=global_softmax_ce_reg`, `model.loss=level_softmax_ce_reg model.weight_mode=equal`. CIFAR-100 uses `haframe_wide_resnet` from scratch; Aircraft, CUB-200, and iNat19 use ImageNet-pretrained `haframe_resnet50`.
- `orthonormal_plugin` is an optional post-logit reuse of the Hier-COS orthonormal taxonomy-frame head/loss for non-Hier-COS models. Enabled runs require taxonomy, hard targets, disabled MixUp/CutMix, and an explicit model output key `orthonormal_plugin_scores_per_level`.
- `hiercos_cifar100` keeps this repo CIFAR hierarchy (3-level), not the paper 5-level CIFAR protocol.
- `hiercos_cub200` is a pragmatic extrapolation preset (paper does not report CUB experiments).
- `hiercos_inat19` follows the upstream iNaturalist19-224 Hier-COS recipe where compatible (ResNet-50, ImageNet pretraining, average pooling, SGD/cosine, batch size 256, `model.loss: kl_reg`, `model.weight_mode: kl_leaf`, `alpha: 0.001`, low-LR transform/backbone groups, and iNat19 normalization), but uses this repo's local 3-level `family -> genus -> species` projection rather than the upstream full 7-level taxonomy. The Hier-COS runner scripts override the loss mode when launching the local lex-ready CE studies.

Parity notes:

- HRN config filenames keep the `_parity` suffix; Hier-COS config filenames use the shorter dataset names.
- Parity presets retain this repo's clean validation/test workflow and FPA/TICE checkpoint ranking.
- See `docs/hrn_hiercos_alignment.md` for the detailed alignment audit and intentional divergences.

If no explicit taxonomy file is provided by an adapter, the dataset base class tries to infer `taxonomy["parent_of"]` from the labels.

## Outputs And Metrics

Each run writes to `train.output_dir`:

- `latest.pt`
- `best_topdown.pt`
- `best_independent.pt`
- `config_resolved.yaml`
- `run_log.jsonl`
- `test_metrics.yaml`

When launched through `scripts/`, these files live under
`<experiment_name>/seed_<seed>/`. The analysis notebooks group the seed
directories by experiment and report the run-to-run mean and sample standard
deviation.

`best_topdown.pt` is selected from validation metrics using:

1. `fpa_topdown` higher is better
2. `tice_topdown` lower is better
3. `weighted_ap_topdown` higher is better

`best_independent.pt` uses the same ranking with independent metric keys:

1. `fpa_independent` higher is better
2. `tice_independent` lower is better
3. `weighted_ap_independent` higher is better

Final test evaluation runs once from each best checkpoint. `test_metrics.yaml` stores separate `topdown` and `independent` sections, each with `best_checkpoint`, `best_epoch`, `best_metric`, and `test_metrics`.

Each epoch event in `run_log.jsonl` stores model loss values separately from evaluation metrics under `train_losses`, `train_metrics`, `val_losses`, and `val_metrics`. Validation losses use hard labels without Mixup/CutMix. Checkpoint selection uses only `val_metrics`; final test outputs remain metric-only.

Standard logged metric keys:

- `acc_level_independent_<level>`
- `acc_level_topdown_<level>`
- `weighted_ap_independent`
- `weighted_ap_topdown`
- `fpa_independent`
- `fpa_topdown`
- `ahd_independent`
- `ahd_topdown`
- `tice_independent` when taxonomy is available
- `tice_topdown` when taxonomy is available

`topdown` metrics use taxonomy-constrained decoding. `independent` metrics use per-level argmax without hierarchy enforcement.

Model-specific losses and diagnostics can add extra keys such as `loss_level_*`, `gk_loss`, `hier_loss`, `ce_loss_leaf`, or projection diagnostics. For Hier-COS, `loss_level_*` and gradient/cosine diagnostics are emitted when `model.loss` exposes per-level losses (`global_softmax_ce_reg`, `level_softmax_ce_reg`); plain `kl_reg` does not emit per-level losses. Both decomposed modes log `ce`, `reg`, `ce_level_*`, `reg_level_*`, and `loss_level_*`; the level losses are already weighted and exactly match the tensors used by lexicographic optimization.

For the full HCC diagnostic key glossary and interpretation guide, see `docs/HCC_DIAGNOSTIC_LOGS.md`.

## Repository Guide

See `docs/FILE_DOCUMENTATION.md` for a concise map of the codebase.
