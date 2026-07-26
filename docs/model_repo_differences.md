# Model fidelity and upstream delta log

Audit date: 20 July 2026

This document records which behavior is source-aligned and which behavior is a
framework adaptation, verified correction, unified-protocol choice, local
extension, or unsupported extrapolation. “Aligned” does not imply identical
results across frameworks, library versions, seeds, or hardware.

## Source anchors

| Family | Primary source | Pinned revision | Comparison |
|---|---|---|---|
| H-CAST | <https://github.com/pseulki/HCAST> | `b1a222bb32da5caf48691b5987d56b7483801907` | PyTorch repository |
| HT-CapsNet | <https://github.com/tasrif-khondaker/HT-CapsNet> | `8a0ea23f3e6b68b75d8add07674b4b0288380417` | TensorFlow repository |
| HRN | <https://github.com/MonsterZhZh/HRN> | `59944e48fcbf41cc475402c8b9cb6af301006399` | PyTorch repository |
| Hier-COS | <https://github.com/Depanshu-Sani/Hier-COS> | `122b01dff393d3b562ad2daac494496fda65131c` | PyTorch repository |
| LH-DNN | <https://arxiv.org/abs/2409.16956> | arXiv `2409.16956v2` | paper; no official code found |

The revisions are evidence anchors. They are not fetched during normal
training.

## Shared corrected-unified choices

These are deliberate protocol choices, not claims about an original model:

- one coarse-to-fine three-level label convention;
- one canonical label space reused across train, validation, and test;
- deterministic CIFAR/CUB partitions with fixed split seed;
- complete validation/test batches;
- validation-only best-checkpoint selection;
- separate top-down and independent best checkpoints;
- exact ranking by FPA, then TICE, then weighted AP;
- common metrics, output artifacts, AMP gate, and strict resume checks.

The common protocol improves comparison discipline but prevents exact
reproduction of repositories that use a different hierarchy, drop evaluation
samples, or choose models on test data.

## H-CAST

### Source-aligned

- CAST/H-CAST internals remain vendored under `models/hcast/internal/`.
- Local class outputs are reordered to the repository-wide
  coarse-to-fine convention at the wrapper boundary.
- Grid/SEEDS segmentation and H-CAST level/global-KL loss paths follow the
  upstream design.
- H-CAST training presets retain the source family’s CAST variant, augmentation
  controls, and LR scaling.

### Verified corrections and adaptations

- The upstream Aircraft helper relies on a local CSV/hard-coded hierarchy.
  This repository instead joins the official parallel
  manufacturer/family/variant annotations and validates the complete
  30/70/100 taxonomy.
- Local graph/backbone guards handle tensor shapes and current timm/PyTorch
  integration. Each retained divergence must have a shape or forward contract
  test; vendored files are not normalized merely for style.
- Shared validation selection, checkpointing, metrics, and logging are
  framework adaptations.

### Local extensions

- HCC is an output-space affine hierarchy projection with scheduled blending
  and diagnostics. It is absent from upstream H-CAST.
- `train.lexicographic.enabled` is an explicit gradient-space projection path.
  HCC alone is not an explicit lexicographic optimizer.
- The orthonormal plugin is an additional local taxonomy-frame study.

## LH-DNN

### Source-aligned

- The CIFAR-100 hierarchy is the paper’s 8 → 20 → 100 hierarchy.
- The large topology uses four pairs of 3×3 convolutional layers, four 2×2
  pools, a shared 512-unit ReLU layer, and linear level heads.
- Forward projection blocks detach the removed higher-priority component so
  the forward scores remain unchanged while the backward subspace is
  restricted.
- Coarser predictions are propagated as detached baselines by the advantage
  topology.
- The objective is an unweighted sum of per-level cross-entropies.
- CIFAR-100 uses the large-network 15-epoch paper schedule and the LR switch at
  epoch 11.

### Extrapolations

- The paper evaluates CIFAR-10, CIFAR-100, and Fashion-MNIST, not CUB or
  Aircraft.
- CUB/Aircraft keep the large topology but reduce the 14×14 final feature map
  to 2×2 with a deterministic 7×7 average pool. For this geometry it is
  equivalent to 2×2 adaptive average pooling, preserves the paper’s pre-head
  geometry at 224 px, and avoids an accidental dense-layer parameter explosion.
- CUB/Aircraft transforms and optimizer regularization are local choices and
  must not be reported as paper settings.

## HT-CapsNet

### Source-aligned

- Primary capsules, per-level secondary capsules, hierarchical skip input,
  routing-by-agreement, taxonomy masking, cross-capsule attention, capsule
  lengths, margin loss, and dynamic level weighting were ported from the
  TensorFlow source.
- The local taxonomy temperature `0.5` is supported by the upstream saved run
  arguments even though upstream constructor defaults are internally
  inconsistent.
- Secondary dimensions `[64, 32, 16]`, three routing iterations, per-image
  standardization, and MixUp alpha `0.2` follow the source experiment family.
- The shipped horizon is 200 epochs, matching the upstream launcher rather
  than the shorter README example.

### Framework adaptations and limits

- Keras/TensorFlow layers and callbacks are represented by native PyTorch
  modules and batch-local loss computations.
- A complete taxonomy and deterministic mode are mandatory locally; missing
  taxonomy cannot silently degrade the model into an ordinary capsule network.
- CIFAR-100 and CUB are upstream datasets. Aircraft is a local extrapolation;
  the upstream repository reports Stanford Cars rather than Aircraft.
- Backbone fallback paths are practical compatibility behavior, not source
  parity. Any fallback is warned.

## HRN

### Source-aligned

- ResNet-50 trunk, three RFM branches, 1024 branch features, 512 embeddings,
  sigmoid tree scores, leaf CE logits, and residual fusion follow upstream.
- The local state-space/tree objective and leaf-only CE reproduce full-label
  HRN semantics.
- CUB/Aircraft preprocessing uses the source-style 550 resize, 448 crop,
  `[0.5, 0.5, 0.5]` normalization, SGD/cosine schedule, and 0.1× trunk LR.

### Unified-protocol choices and extrapolations

- Upstream drops incomplete loaders, including evaluation. The corrected
  protocol deliberately keeps every validation/test sample.
- Upstream scripts can select/report on test data. Local checkpoints are
  selected only on validation data.
- Partial-label HRN branches are out of scope.
- CIFAR-100 is a local extrapolation.

## Hier-COS

### Source-aligned core

- Taxonomy nodes define orthogonal coordinate directions and hierarchical
  subspaces.
- A random orthonormal fixed classifier represents the upstream
  `opts.orthonormal_basis_vectors`; subspace projection masks operate in the
  resulting coordinate space.
- `kl_reg` combines the global absolute-score path-distribution objective with
  the levelwise cosine regularizer.
- ResNet-50/WideResNet families, fixed frames, Aircraft bottom crop, and
  SGD/cosine parameter-group behavior are retained where compatible.

### Important source ambiguity

- The upstream Aircraft script named `hier-cos.sh` passes
  `--feature_space haf++`, unlike the CIFAR/iNat scripts that pass
  `--feature_space hier-cos`. Aircraft behavior therefore cannot be described
  as an unambiguous exact Hier-COS recipe. The local Aircraft preset is a
  consistent Hier-COS adaptation and is labeled accordingly.

### Local extensions and depth choices

- `global_softmax_ce_reg` and `level_softmax_ce_reg` are local decompositions
  that expose exact per-level objectives for diagnostics and lexicographic
  projection. Baseline presets remain `kl_reg`; launchers override the loss
  explicitly for these studies.
- The repository uses three CIFAR levels instead of upstream’s five and three
  iNat levels instead of upstream’s seven.
- CUB is not an upstream Hier-COS experiment and is an extrapolation.
- The shared orthonormal plugin generalizes the fixed-frame objective to other
  host models; it is not part of upstream Hier-COS.

## Threats to validity

- TensorFlow/PyTorch kernel, initialization, attention, and optimizer details
  can prevent numerical identity even when equations match.
- Upstream repositories may contain accidental behavior. The audit preserves
  method-defining behavior but does not reproduce test leakage, incomplete
  evaluation, or silent input fallbacks.
- LH-DNN is reconstructed from prose, equations, and figures without official
  implementation tests.
- Local dataset/model extrapolations have not inherited paper-level
  hyperparameter validation.
- Corrected split and checkpoint behavior makes new runs incomparable with old
  results unless the protocol difference is stated.
- One seed is not evidence of stable superiority; report seed count and sample
  standard deviation.
- Decoder dependence is material. Every claim must identify top-down or
  independent decoding and use that mode’s selected checkpoint.
