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
- Dataset adapters for CIFAR-100, CUB-200-2011, FGVC-Aircraft, and iNat21-style data
- Shared hierarchical metrics and logging
- Optional H-CAST Hierarchical Constraint Cascade (HCC) variants

## Setup

Install a compatible `torch` and `torchvision` build for your machine first, then install the repo dependencies:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

Notes:

- `timm` is required for H-CAST and timm-backed schedulers.
- `opencv-contrib-python` is required for H-CAST `segments.mode: seeds`.
- Preset configs in `configs/` use machine-specific dataset/output paths. Update them before running.

## Quick Start

1. Pick a preset config from `configs/`.
2. Edit at least `dataset.root` and `train.output_dir`.
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
- `configs/hcast/hcast_inat21mini.yaml`
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

- `configs/hiercos/hiercos_cifar100_parity.yaml`
- `configs/hiercos/hiercos_cub200_parity.yaml`
- `configs/hiercos/hiercos_aircraft_parity.yaml`

Reusable templates:

- `configs/templates/dataset_template.yaml`
- `configs/templates/hcast_template.yaml`
- `configs/templates/hcc_template.yaml`
- `configs/templates/ht_capsnet_template.yaml`
- `configs/templates/hrn_template.yaml`
- `configs/templates/hiercos_template.yaml`
- `configs/templates/training_template.yaml`

## Config Structure

Experiment YAMLs use the same top-level sections:

- `model`
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
- `model.loss` (Hier-COS: `kl_reg`, `per_level_kl_reg`, `per_level_ce`, or `per_level_abs_node_ce`)
- `model.weight_mode` (Hier-COS shared weighting for KL target-path and CE: `equal`, `kl_leaf`, or `kl_coarse`)

Lexicographic upper-bound mode (3-level):

- `train.lexicographic.enabled`: enables gradient-space lexicographic updates
- `train.lexicographic.start_epoch`: internal zero-based epoch index where projected gradients start
- `train.lexicographic.eps`: projection denominator epsilon
- `train.lexicographic.log_metrics`: logs projection diagnostics under `train_metrics`
- H-CAST requires exactly 3 level losses and `model.loss.globalkl: false`
- Hier-COS lexicographic mode requires per-level losses (`model.loss: per_level_kl_reg`, `per_level_ce`, or `per_level_abs_node_ce`); plain `model.loss: kl_reg` does not expose differentiable per-level losses

The config loader supports positional dotlist overrides such as `train.epochs=10` or `optim.lr=1e-4`.

## Supported Datasets

Strict dataset ids:

- `cifar-100`
- `cub-200-2011`
- `fgvc-aircraft`
- `inat21-mini`

Adapter behavior:

- CIFAR-100 loads through `torchvision.datasets.CIFAR100` and supports `hierarchy_depth: 2` or `3`. The 3-level layout is `super -> coarse -> fine`.
- CUB supports `train/` and `test/` folder splits, `images_split/{train,test}`, or the official `images.txt` / `image_class_labels.txt` / `train_test_split.txt` layout.
- FGVC-Aircraft reads the official `images_variant_{train,val,test,trainval}.txt` files under `data/`-style roots.
- iNat uses official iNat21-style JSON or JSON-in-tar annotations. For the repo `test` split it prefers official validation labels, then `public_test` / `test`, then a deterministic fallback split if needed.

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
- `lhdnn` requires at least 2 levels and always requires taxonomy.
- `ht_capsnet` requires at least 2 levels. The builder also enforces `train.seed` and `runtime.deterministic: true`.
- `hrn` supports exactly 3 levels.
- `hrn` follows upstream HRN for CUB-200 and Aircraft parity presets: ImageNet-pretrained ResNet-50, 1024-d branch bottlenecks, 512-d classifiers, 448 crops from 550x550 resized images, `[0.5, 0.5, 0.5]` normalization, and SGD parameter groups with the ResNet trunk at 0.1x LR.
- `hrn` parity loss does not support mixup/cutmix soft targets, so the shipped HRN presets keep them disabled.
- `hrn_cifar100_parity` is a local extrapolation because the upstream HRN repo does not include CIFAR-100; it keeps this repo CIFAR hierarchy.
- `hiercos` requires taxonomy (`taxonomy.parent_of`) and at least 2 levels.
- `hiercos` does not support mixup/cutmix soft targets. Keep `dataset.transforms.mixup/cutmix: 0.0`.
- `hiercos` uses a single fixed orthonormal Hier-COS frame with taxonomy-driven subspace scores. `model.loss: kl_reg` is the default paper-aligned KL + level regularization objective. `model.loss: per_level_kl_reg` exposes differentiable per-level losses (`coarse`, `mid`, `fine`) and uses their sum as `total`, making lex and non-lex comparisons use the same reported objective in this mode. `model.loss: per_level_ce` is a local ablation that optimizes CE on level subspace scores (`logits_per_level`). `model.loss: per_level_abs_node_ce` is a local ablation that optimizes CE on `node_logits.abs()[:, level_node_ids]` per level. `model.weight_mode` supports `equal` (exact `1/depth` per level), leaf-heavy `kl_leaf`, and reversed/coarse-heavy `kl_coarse`; the same per-level weights are used for KL target-path node weights (`kl_reg`/`per_level_kl_reg`) and CE level-loss weighting (`per_level_ce`/`per_level_abs_node_ce`). `per_level_kl_reg` additionally logs `kl_level_*` and `reg_level_*`. CLI examples: `model.loss=per_level_kl_reg`, `model.loss=per_level_ce model.weight_mode=equal`, `model.loss=per_level_ce model.weight_mode=kl_leaf`, `model.loss=per_level_abs_node_ce model.weight_mode=equal`. CIFAR-100 uses `haframe_wide_resnet` from scratch; Aircraft and CUB-200 use ImageNet-pretrained `haframe_resnet50`.
- `hiercos_cifar100_parity` keeps this repo CIFAR hierarchy (3-level), not the paper 5-level CIFAR protocol.
- `hiercos_cub200_parity` is a pragmatic extrapolation preset (paper does not report CUB experiments).

Parity notes:

- HRN and Hier-COS config folders now contain only `*_parity.yaml` presets.
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

`best_topdown.pt` is selected from validation metrics using:

1. `fpa_topdown` higher is better
2. `tice_topdown` lower is better
3. `weighted_ap_topdown` higher is better

`best_independent.pt` uses the same ranking with independent metric keys:

1. `fpa_independent` higher is better
2. `tice_independent` lower is better
3. `weighted_ap_independent` higher is better

Final test evaluation runs once from each best checkpoint. `test_metrics.yaml` stores separate `topdown` and `independent` sections, each with `best_checkpoint`, `best_epoch`, `best_metric`, and `test_metrics`.

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

Model-specific losses and diagnostics can add extra keys such as `loss_level_*`, `gk_loss`, `hier_loss`, `ce_loss_leaf`, or projection diagnostics. For Hier-COS, `loss_level_*` and gradient/cosine diagnostics are emitted when `model.loss` exposes per-level losses (`per_level_kl_reg`, `per_level_ce`, `per_level_abs_node_ce`); plain `kl_reg` does not emit per-level losses. In Hier-COS, `model.weight_mode` is shared across KL target-path weighting and CE level weighting; `per_level_kl_reg` also logs `kl_level_*` and `reg_level_*`.

For the full HCC diagnostic key glossary and interpretation guide, see `docs/HCC_DIAGNOSTIC_LOGS.md`.

## Repository Guide

See `docs/FILE_DOCUMENTATION.md` for a concise map of the codebase.
