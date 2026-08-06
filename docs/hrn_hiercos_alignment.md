# HRN + Hier-COS Alignment Audit

Updated: 2026-08-05

This note audits local `hrn` and `hiercos` implementations against upstream repositories:

- HRN: https://github.com/MonsterZhZh/HRN
- Hier-COS: https://github.com/Depanshu-Sani/Hier-COS

The implementation choices below follow the selected clean-protocol constraints for this repo:

- Keep this repo train/val/test workflow and final dual-checkpoint test evaluation.
- Keep this repo CIFAR hierarchy depth for Hier-COS (`hierarchy_depth: 3`).
- HRN scope is full-label experiments only.
- Keep exact checkpoint ranking by `(FPA, -TICE, weighted AP)`.
- Evaluate every validation/test sample (`drop_last_eval: false`).

## HRN Alignment Status

### Architecture
- **Aligned for CUB/Aircraft**: the ResNet-50 path in local
  `models/hrn/model.py` mirrors upstream `CUB_Aircraft/RFM.py` core design.
- Evidence:
  - ResNet-50 trunk.
  - 3 branch conv blocks (`1x1 -> 3x3`, BN+ReLU).
  - FC heads `2048 -> 1024 -> 512` with BN/ELU.
  - Residual fusion path `order`, `family+order`, `species+family+order`.
  - Separate species logits for CE (`classifier_3_1`) and tree branch (`classifier_3` with sigmoid).
  - Effective evaluation scores use sigmoid at order/family and softmax at species.

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
- **Dataset-native CIFAR extrapolation**:
  - 32 px reflect-padded random crop and CIFAR normalization, matching the
    repository's other CIFAR-100 baselines rather than the HRN FGVC recipe.

### CIFAR Backbone Choice

The CIFAR-100 preset uses a WideResNet-28-8 trunk trained from scratch at the
dataset-native 32 px resolution. This is a deliberate local adaptation, not an
HRN paper setting. A standard ResNet-50 at this resolution would produce an
approximately `1 x 1` final spatial feature map before HRN's three RFM
branches. Consequently, the branches' subsequent `3 x 3` convolutions would
have essentially no remaining spatial neighbourhoods to process.

HRN and Hier-COS instantiate the same shared CIFAR WideResNet trunk
implementation, with identical depth, width, residual blocks, initialization,
and downsampling geometry. HRN consumes the shared trunk's spatial
`512 x 8 x 8` output directly, whereas Hier-COS appends its existing node-space
projection and global pooling to obtain a feature vector. This choice controls
the backbone geometry when comparing their hierarchical mechanisms and avoids
interpolating CIFAR images to 224 px, which would add no genuine visual detail.

The CIFAR model must therefore be reported explicitly as **HRN-WRN-28-8**, a
local backbone-controlled adaptation, and must not be described as the
paper-aligned HRN architecture. Historical HRN-ResNet50 CIFAR results are not
directly interchangeable with new HRN-WRN-28-8 runs and should retain their
original run identifiers if used as a separate backbone ablation.

### Split/Evaluation Protocol
- **Intentionally different**:
  - Upstream trains with train/test directly; this repo keeps a train/val split and selects best checkpoints from validation metrics.
  - Upstream drops incomplete evaluation batches; this repo evaluates all samples.
  - This is deliberate for consistent, cleaner cross-model evaluation.

## Hier-COS Alignment Status

### Architecture
- **Aligned**:
  - CIFAR path: HAFrame WideResNet (`haframe_wide_resnet`) with hierarchical node-space backbone.
  - Aircraft path: HAFrame ResNet-50 (`haframe_resnet50`) with ImageNet-pretrained trunk.
  - A single fixed orthonormal Hier-COS frame with taxonomy-driven subspace scores is supported.
  - Inference takes `argmax` directly over the raw taxonomy-subspace projection
    norms, matching upstream `get_distances(...).topk(...)`; softmax is used by
    the loss and probability diagnostics, not as a prediction transform.

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

### Data Transforms
- **Aligned + explicit FGVC parity option**:
  - CIFAR parity: reflect-padded random crop + horizontal flip + CIFAR normalization.
  - FGVC-Aircraft parity: `Resize(224) -> RandomCrop(224,padding=4) -> HFlip -> FGVC normalization`.
  - Added shared transform option `dataset.transforms.manual.crop_bottom_pixels`; parity FGVC config uses `20` to reproduce upstream preprocessing that removes the bottom banner.

### Split/Evaluation Protocol
- **Intentionally different**:
  - Upstream protocol uses its own train/val/test scripts and accuracy-centric selection.
  - This repo keeps unified val-based model selection and final test evaluation from top-down and independent best checkpoints.

## Intentional Divergences Kept In This Pass

1. Hier-COS CIFAR remains 3-level in this repo (not upstream 5-level CIFAR protocol).
2. HRN partial-label (`proportion`) experiments are not implemented in this pass.
3. Checkpoint selection remains `FPA/TICE/weighted AP` to preserve repository-wide comparability.
4. CIFAR-100 is an explicit native-32 px HRN-WRN-28-8 adaptation because
   upstream HRN does not report CIFAR-100; it prioritizes CIFAR-appropriate
   spatial geometry and backbone control with Hier-COS over the paper's
   ResNet-50/448 px FGVC geometry.

## Active presets

- `configs/hrn/hrn_cifar100.yaml` (local HRN-WRN-28-8 adaptation)
- `configs/hrn/hrn_cub200.yaml`
- `configs/hrn/hrn_aircraft.yaml`
- `configs/hiercos/hiercos_cifar100.yaml`
- `configs/hiercos/hiercos_cub200.yaml` (local extrapolation)
- `configs/hiercos/hiercos_aircraft.yaml`

These are the HRN/Hier-COS presets kept in their folders for this alignment pass.
