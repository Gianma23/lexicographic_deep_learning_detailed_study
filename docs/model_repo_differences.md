# Model fidelity and upstream delta log

Audit date: 1 August 2026

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

Paper run settings were checked against the
[HRN CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Label_Relation_Graphs_Enhanced_Hierarchical_Residual_Network_for_Hierarchical_Multi-Granularity_CVPR_2022_paper.html)
and the
[HT-CapsNet Knowledge-Based Systems paper](https://doi.org/10.1016/j.knosys.2025.114444).

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
- Secondary dimensions `[64, 32, 16]`, three routing iterations, and MixUp alpha
  `0.2` follow the source experiment family.
- Standard-scaler scope is batch-scoped on CUB/Aircraft
  (`dataset.transforms.normalization_scope: batch`). Upstream reduces a single
  scalar mean/std over whatever tensor it receives: for path-loaded datasets
  that is the tensor *after* `padded_batch`, so the statistic spans the batch,
  not the image. An earlier revision of this document described the port as
  per-image standardization and called that source-aligned; that was incorrect.
  CIFAR reaches the upstream `image_type='array'` branch, which standardizes the
  whole split array once, and remains per-image locally — a known open gap.
- CUB/Aircraft downsample without antialiasing
  (`dataset.transforms.fixed_resize_antialias: false`), matching
  `tf.image.resize`'s default: a fixed 2x2 source kernel regardless of the
  downsample ratio, applied after the float conversion. At 64 px this discards
  most of a 1–2 MP source. The source's intermediate 512 px resize is a
  `padded_batch` artifact and is deliberately not reproduced; measured against
  the antialiased single-stage pipeline it contributes about a third as much
  pixel difference as the antialias setting does. The paper specifies only
  "64 x 64 pixels", so the code is the sole authority here.
- Aircraft crops the 20 px copyright banner
  (`dataset.transforms.manual.crop_bottom_pixels: 20`). This is a documented
  local decision, not a reproduction: HT-CapsNet reports Stanford Cars, which
  has no banner, so the source pipeline has no defined behaviour for it. The
  FGVC-Aircraft README instructs that the banner be removed, and every other
  model in this repository removes it — Hier-COS explicitly, HRN and H-CAST via
  their evaluation centre crops. Leaving it in would have made HT-CapsNet the
  only Aircraft row trained and evaluated on a constant artifact. LH-DNN also
  uses the fixed-resize path on Aircraft and does not yet crop. (`16` heads with independent
  `key_dim=value_dim=32`) through PyTorch scaled-dot-product attention; Q/K/V
  width is therefore 512 at every level rather than being divided from the
  capsule dimension.
- Primary features are flattened in NHWC order and `tf_efficientnet_b7`
  supplies the final spatial `forward_features` map. The Keras
  `EfficientNetB7` stem is reproduced in full: `Rescaling(1/255)` followed by
  the `Normalization` layer, whose ImageNet checkpoint state is
  `mean = [0.485, 0.456, 0.406]` and `variance = [0.229, 0.224, 0.225]`
  (Keras stores the standard deviations in the variance slot, and the layer
  divides by `sqrt(variance)`).
- Dynamic loss weights are checkpointed model buffers: each batch uses the
  weights produced after the preceding batch, matching the Keras callback.
  Where the paper's written weighting equation and callback differ, the port
  follows the source callback: `tau_i = 1 - acc_i * initial_i`, then
  `(1 - dynamic_weight) * tau_i / sum(tau)`. The source callback runs in
  `on_train_batch_end` and reads `acc_i` from the Keras `logs` dict, which
  carries the metric accumulated since the start of the epoch; the port
  therefore keeps epoch-to-date accuracy accumulators that reset on each
  `set_epoch` call, rather than using the current batch's accuracy.
- MixUp samples one beta coefficient per example with random pairing. The
  epoch-indexed source schedule holds `0.001` through epoch index 10 and uses
  `0.00095` at index 11.
- The shipped horizon is 200 epochs, matching the upstream launcher rather
  than the shorter README example.

### Framework adaptations and limits

- Keras/TensorFlow layers and callbacks are represented by native PyTorch
  modules; PyTorch's optimized scaled-dot-product primitive performs attention.
- A complete taxonomy and deterministic mode are mandatory locally; missing
  taxonomy cannot silently degrade the model into an ordinary capsule network.
- CIFAR-100 and CUB are upstream datasets. Aircraft is a local extrapolation
  ported from the paper's 64 px fine-grained recipe; the paper reports Stanford
  Cars rather than Aircraft.
- Local CUB keeps the unified 13/38/200 taxonomy, not the paper's 39/123/200
  construction, so it is a protocol adaptation rather than an exact paper run.
- Requested ImageNet backbones are mandatory. Missing weights fail clearly and
  never silently produce a random-initialized paper-labeled run.
- Corrected attention projection shapes and checkpointed loss-weight buffers
  intentionally make older local HT-CapsNet checkpoints incompatible; fidelity
  runs must start from fresh initialization.

## HRN

### Source-aligned

- ResNet-50 trunk, three RFM branches, 1024 branch features, 512 embeddings,
  sigmoid tree scores, leaf CE logits, and residual fusion follow upstream.
- Inference exposes sigmoid order/family scores and softmax species scores,
  matching the upstream evaluation path.
- The local state-space/tree objective and leaf-only CE reproduce full-label
  HRN semantics.
- CUB/Aircraft use the source-style 550 resize, 448 crop,
  `[0.5, 0.5, 0.5]` normalization, SGD/cosine schedule, and 0.1× trunk LR.

### Unified-protocol choices and extrapolations

- Upstream drops incomplete loaders, including evaluation. The corrected
  protocol deliberately keeps every validation/test sample.
- Upstream scripts can select/report on test data. Local checkpoints are
  selected only on validation data.
- Partial-label HRN branches are out of scope.
- CIFAR-100 is a local extrapolation that retains the repository-wide native
  32 px CIFAR preprocessing for cross-model comparability. Consequently it is
  not an input-resolution reproduction of an HRN paper run.
- Requested ImageNet initialization is mandatory; unavailable weights fail.

## Hier-COS

### Source-aligned core

- Taxonomy nodes define orthogonal coordinate directions and hierarchical
  subspaces.
- A random orthonormal fixed classifier represents the upstream
  `opts.orthonormal_basis_vectors`; subspace projection masks operate in the
  resulting coordinate space.
- Independent inference ranks the raw subspace projection norms directly,
  matching upstream `get_distances(...)` followed by `topk`; it does not
  softmax those scores before prediction.
- `kl_reg` combines the global absolute-score path-distribution objective with
  the levelwise cosine regularizer.
- ResNet-50/WideResNet families, fixed frames, Aircraft bottom crop, and
  SGD/cosine parameter-group behavior are retained where compatible.

### Important source ambiguity

- The upstream Aircraft script named `hier-cos.sh` passes
  `--feature_space haf++`, unlike the CIFAR script that passes
  `--feature_space hier-cos`. Aircraft behavior therefore cannot be described
  as an unambiguous exact Hier-COS recipe. The local Aircraft preset is a
  consistent Hier-COS adaptation and is labeled accordingly.

### Local extensions and depth choices

- `global_softmax_ce_reg` and `level_softmax_ce_reg` are local decompositions
  that expose exact per-level objectives for diagnostics and lexicographic
  projection. Baseline presets remain `kl_reg`; launchers override the loss
  explicitly for these studies.
- The repository uses three CIFAR levels instead of upstream’s five.
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
