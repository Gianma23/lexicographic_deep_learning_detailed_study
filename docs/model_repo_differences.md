# Model fidelity and upstream delta log

Audit date: 14 August 2026

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

### Shared paper/source core

- Per-level secondary capsules, hierarchical skip input, routing-by-agreement,
  taxonomy masking, hierarchical agreement, cross-capsule attention, capsule
  lengths, and margin loss are present in both the paper and TensorFlow source.
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
  whole split array once. The CIFAR preset now uses
  `dataset.transforms.normalization_scope: dataset`, computing one scalar
  mean/std over each concrete train/validation/test split before batching.
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
  uses the fixed-resize path on Aircraft and does not yet crop.
- Cross-capsule attention uses `16` heads with independent
  `key_dim=value_dim=32` through PyTorch scaled-dot-product attention; Q/K/V
  width is therefore 512 at every level rather than being divided from the
  capsule dimension. Its projections use the Keras rank-three `EinsumDense`
  Glorot fan calculation; ordinary Xavier initialization on flattened PyTorch
  `Linear` shapes is not equivalent.
- Primary features are flattened in NHWC order and `tf_efficientnet_b7`
  supplies the final spatial `forward_features` map. The explicit
  `tf_efficientnet_b7.aa_in1k` tag is used rather than timm's version-dependent
  untagged alias; its stem convolution matches the Keras
  `efficientnetb7_notop.h5` tensor exactly. A mechanical audit of all 1,035
  weight-bearing arrays (all MBConv, SE, convolution, and BatchNorm tensors)
  found bit-for-bit equality after the required layout transposes. PyTorch
  BatchNorm momentum is `0.01`
  (the equivalent of Keras `0.99`) and progressive drop path reaches `0.2`,
  matching the Keras application. The Keras
  `EfficientNetB7` stem is reproduced in full: `Rescaling(1/255)` followed by
  the `Normalization` layer, whose ImageNet checkpoint state is
  `mean = [0.485, 0.456, 0.406]` and `variance = [0.229, 0.224, 0.225]`
  (Keras stores the standard deviations in the variance slot, and the layer
  divides by `sqrt(variance)`).
- Dynamic loss weights are checkpointed model buffers: each batch uses the
  weights produced after the preceding batch, matching the Keras callback.
  The source callback runs in
  `on_train_batch_end` and reads `acc_i` from the Keras `logs` dict, which
  carries the metric accumulated since the start of the epoch; the port
  therefore keeps epoch-to-date accuracy accumulators that reset on each
  `set_epoch` call, rather than using the current batch's accuracy.
- MixUp samples one beta coefficient per example with random pairing. The
  epoch-indexed paper-reported schedule holds `0.001` through epoch index 10
  and uses `0.00095` at index 11 (the released parser instead defaults to a
  `0.9` decay factor). `keras_adam` implements the Keras 2.8 epsilon-hat
  update instead of assuming that PyTorch Adam with `eps=1e-7` is identical.
- During training, margin loss consumes raw capsule lengths. During validation
  and test it consumes the softmax output produced by the source `LengthLayer`;
  argmax predictions are unchanged.
- The shipped horizon is 200 epochs, matching the upstream launcher rather
  than the shorter README example.

### Paper/source contradictions and selected behavior

The public paper and its released TensorFlow file are not one executable
specification. The baseline presets now select the published equations where
the two materially disagree, while the released behavior remains available by
an explicit config value:

- `primary_capsule_mode: paper_independent` implements Eq. 4/Fig. 2: each
  hierarchy level independently reshapes the shared backbone feature map to its
  own primary-capsule dimension and then applies squash. The released file's
  `source_reuse` mode builds one squashed 8-D tensor and reinterprets its memory
  as 64-D and 32-D capsules at later levels.
- `routing_parent_activation: norm` implements Eq. 13's direct parent capsule
  length. `softmax_norm` reproduces the released layer, where nearly equal
  LayerNorm capsule lengths become an almost uniform parent vector and greatly
  weaken the taxonomy mask.
- `dynamic_weight_formula: paper` implements Eq. 21,
  `(1 - accuracy_i) * initial_i`. `released_source` implements the callback's
  different parentheses, `1 - accuracy_i * initial_i`.
- `drop_last_train: false` retains the last partial batch as `padded_batch`
  does. Checkpoint ranking deliberately remains the same repository-wide
  FPA/TICE/weighted-AP policy used for all models. The local run also keeps a
  genuine validation split and a fixed 200-epoch horizon: it does not reuse
  CIFAR test labels as validation data or enable the source's undocumented
  early stopping.

Two high-impact upstream reproducibility defects are exposed as diagnostics
rather than silently changed. First, public preprocessing applies a
split-wide StandardScaler and then EfficientNet's embedded `/255`, reducing
CIFAR image variation by roughly 60x. To test the plausible
raw-input recipe, override `dataset.transforms.normalization=none`,
`dataset.transforms.normalization_scope=image`, and
`model.backbone_preprocessing=keras_unit_range`. Second, both Eq. 18 and the
released layer compute capsule lengths after a per-capsule LayerNorm. At its
initial affine parameters those lengths are almost constant at `sqrt(D)` and
outside the margin loss's `[0, 1]` design range. The existing
`attn_postprocess: squash` option is therefore a useful ablation, but is not the
paper/source default and is not presented as a verified correction.

The paper also says the hierarchical-agreement gate and transform are
taxonomy-biased at initialization, but supplies no edge/non-edge values; the
released code initializes every gate to `0.5` and the transform from a normal
distribution. The port retains those executable values instead of inventing an
unpublished initialization. Likewise, the source cyclically tiles semantic
taxonomy rows over primary as well as previous-level capsules, while the paper
does not specify that mapping. This remains source-aligned and explicitly
unresolved rather than adding another unsupported routing mode.

Other released-recipe contradictions were audited but do not justify silent
changes: the shell launcher inherits `mask_threshold_high=0.9` and learning-rate
decay `0.9`, whereas the paper/notebook use `0.99` and `0.95`; the notebook adds
fine-accuracy early stopping despite the paper's fixed 200 epochs; CIFAR's
default zero validation split reuses the official test set for validation; and
the published multi-seed launcher contains malformed dataset/variable
arguments. The presets retain the paper values and an honest validation split,
so published-table equivalence cannot be guaranteed from the public artifacts.

The locally supplied `HierarchicalClassification-main` student archive is not
an independent HT-CapsNet port. Its capsule model is the older HD-CapsNet
family: a custom convolutional encoder with ordinary routing and none of
HT-CapsNet's EfficientNet, taxonomy mask, hierarchical-agreement, or
cross-capsule-attention components. Moreover, its default CIFAR runner selects a
non-capsule three-head projection network. That archive also contains material
input-conversion, MixUp, dynamic-weight, optimizer, and schedule defects, so it
is useful for confirming the 8/20/100 label taxonomy but not as a numerical
oracle for this model.

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
- Paper-aligned primary capsules, routing activation, dynamic weighting, plus
  corrected attention initialization, backbone training semantics, optimizer
  state, and loss-weight buffering intentionally make older local HT-CapsNet
  training checkpoints
  incompatible; fidelity runs must start from fresh initialization.

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
- CIFAR-100 is a local HRN-WRN-28-8 adaptation that retains the
  repository-wide native 32 px preprocessing and uses the same CIFAR-style
  backbone geometry as Hier-COS. The WideResNet trunk preserves an `8 x 8`
  spatial map for the HRN branches, whereas a standard ResNet-50 would reduce
  the native CIFAR input to approximately `1 x 1`. Consequently this preset is
  neither an architecture nor an input-resolution reproduction of an HRN paper
  run; CUB and Aircraft remain the ResNet-50/448 px source-aligned presets.
- For the ResNet-50 presets, requested ImageNet initialization is mandatory;
  unavailable weights fail rather than silently falling back to random weights.

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
