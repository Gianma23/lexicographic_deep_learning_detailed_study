# HRN + Hier-COS Alignment Audit

Date: 2026-05-12

This note audits local `hrn` and `hiercos` implementations against upstream repositories:

- HRN: https://github.com/MonsterZhZh/HRN
- Hier-COS: https://github.com/Depanshu-Sani/Hier-COS

The implementation choices below follow the selected clean-protocol constraints for this repo:

- Keep this repo train/val/test workflow and final dual-checkpoint test evaluation.
- Keep this repo CIFAR hierarchy depth for Hier-COS (`hierarchy_depth: 3`).
- HRN scope is full-label experiments only.
- Keep checkpoint ranking by `FPA`/`TICE`/`weighted AP`.

## HRN Alignment Status

### Architecture
- **Aligned**: local `models/hrn/model.py` mirrors upstream `CUB_Aircraft/RFM.py` core design.
- Evidence:
  - ResNet-50 trunk.
  - 3 branch conv blocks (`1x1 -> 3x3`, BN+ReLU).
  - FC heads `2048 -> 1024 -> 512` with BN/ELU.
  - Residual fusion path `order`, `family+order`, `species+family+order`.
  - Separate species logits for CE (`classifier_3_1`) and tree branch (`classifier_3` with sigmoid).

### Loss
- **Aligned (full-label mode)**: local `models/hrn/losses.py` matches upstream tree loss + species CE behavior.
- Evidence:
  - Tree-state-space objective on concatenated sigmoid scores.
  - Species CE applied to leaf-observed samples.
  - Mixup/cutmix soft-targets intentionally disabled for HRN parity.

### Optimizer + Scheduler
- **Aligned for full-label branch**:
  - SGD with 10 head groups at LR and trunk at `0.1x` LR.
  - Cosine schedule with the same functional form used in upstream (`0.5 * lr * (cos(pi*t/T)+1)`).
- **Intentional omission**:
  - Upstream partial-label branch (`proportion == 0.1`) and its StepLR path are out of scope in this pass.

### Data Transforms
- **Aligned for CUB/Aircraft parity presets**:
  - `Resize(550,550) -> RandomCrop(448,padding=8) -> RandomHorizontalFlip -> Normalize(0.5,0.5,0.5)` for train.
  - `Resize(550,550) -> CenterCrop(448) -> Normalize(0.5,0.5,0.5)` for eval.

### Split/Evaluation Protocol
- **Intentionally different**:
  - Upstream trains with train/test directly; this repo keeps a train/val split and selects best checkpoints from validation metrics.
  - This is deliberate for consistent, cleaner cross-model evaluation.

## Hier-COS Alignment Status

### Architecture
- **Aligned**:
  - CIFAR path: HAFrame WideResNet (`haframe_wide_resnet`) with hierarchical node-space backbone.
  - Aircraft path: HAFrame ResNet-50 (`haframe_resnet50`) with ImageNet-pretrained trunk.
  - iNat19 path: HAFrame ResNet-50 (`haframe_resnet50`) with ImageNet-pretrained trunk, average pooling, `kl_reg`/`kl_leaf`, `alpha=0.001`, batch size 256, 100 epochs, SGD/cosine, and upstream iNaturalist19-224 hyperparameters where compatible.
  - A single fixed orthonormal Hier-COS frame with taxonomy-driven subspace scores is supported.

### Loss
- **Aligned**:
  - Local KL + level-regularization objective matches upstream `HierCOS_Loss` structure.
- **Local lex extension**:
  - `model.loss: global_softmax_ce_reg` uses weighted target CE under one global taxonomy-node softmax plus the existing level regularizer. It exposes three differentiable level losses (`coarse`, `mid`, `fine`), and `total` is their sum.
- **Local ablation**:
  - `model.loss: level_softmax_ce_reg` uses the same path targets, CE weights, regularizer, and aggregation as `global_softmax_ce_reg`, but normalizes logits independently inside each hierarchy level. `model.weight_mode` supports equal weights (`1/depth`), leaf-heavy KL-style weights, and reversed/coarse-heavy weights. CE is weighted and regularization is not. This mode is not paper-faithful Hier-COS; it isolates the effect of softmax normalization scope.

### Optimizer + Scheduler
- **Aligned for parity presets**:
  - SGD with backbone at `0.1x` LR and cosine schedule equivalent to upstream custom-sgd cosine behavior.
  - The iNat19 preset additionally matches upstream `--larger-backbone` behavior by training the transform head at `0.1x` LR.

### Data Transforms
- **Aligned + explicit FGVC parity option**:
  - CIFAR parity: reflect-padded random crop + horizontal flip + CIFAR normalization.
  - FGVC-Aircraft parity: `Resize(224) -> RandomCrop(224,padding=4) -> HFlip -> FGVC normalization`.
  - iNat19 parity: upstream iNat19 normalization with 224-pixel random resized crops for training and 224-pixel resize for validation/test.
  - Added shared transform option `dataset.transforms.manual.crop_bottom_pixels`; parity FGVC config uses `20` to reproduce upstream preprocessing that removes the bottom banner.

### Split/Evaluation Protocol
- **Intentionally different**:
  - Upstream protocol uses its own train/val/test scripts and accuracy-centric selection.
  - This repo keeps unified val-based model selection and final test evaluation from top-down and independent best checkpoints.

## Intentional Divergences Kept In This Pass

1. Hier-COS CIFAR remains 3-level in this repo (not upstream 5-level CIFAR protocol).
2. Hier-COS iNat19 uses this repo's local 3-level family/genus/species projection, not the upstream full 7-level iNaturalist19 taxonomy.
3. HRN partial-label (`proportion`) experiments are not implemented in this pass.
4. Checkpoint selection remains `FPA/TICE/weighted AP` to preserve repository-wide comparability.

## New Parity Presets Added

- `configs/hrn/hrn_cifar100_parity.yaml` (local extrapolation)
- `configs/hrn/hrn_cub200_parity.yaml`
- `configs/hrn/hrn_aircraft_parity.yaml`
- `configs/hiercos/hiercos_cifar100.yaml`
- `configs/hiercos/hiercos_cub200.yaml` (local extrapolation)
- `configs/hiercos/hiercos_aircraft.yaml`
- `configs/hiercos/hiercos_inat19.yaml` (local 3-level iNat19 projection)

These are the HRN/Hier-COS presets kept in their folders for this alignment pass.
