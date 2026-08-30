# HT-CapsNet: differences from the released Keras implementation

Complete list of the ways this repository's HT-CapsNet baseline departs from the
official TensorFlow/Keras implementation of *Taxonomy-guided routing in capsule
network for hierarchical image classification* (Noor et al., Knowledge-Based
Systems 329:114444, 2025).

**Reference source.** `https://github.com/tasrif-khondaker/HT-CapsNet`, pinned at
`8a0ea23f3e6b68b75d8add07674b4b0288380417`, cloned locally to
`/scratch/g.saggini1/outputs/ht_capsnet_upstream_reference`. Upstream line
numbers below refer to that clone. Where the paper and the released code
disagree, the repository rule is: architecture, routing, loss-update and
preprocessing **behaviour** follow the released code; experiment
**hyperparameters** follow the paper. This document records what differs from
the code; it does not attempt to re-derive the paper's tables.

**Local files.** `models/ht_capsnet/{model,routing,losses,factory}.py`,
`configs/capsnet/capsnet_{cifar100,cub200,aircraft}.yaml`,
`datasets/{transforms,loaders,cifar100,cub,aircraft}.py`, `train/mixup.py`,
`train/runtime/{optimization,selection}.py`, `train/metrics.py`.

**Classification used below**

| Tag | Meaning |
| --- | --- |
| **[protocol]** | Deliberate deviation for cross-family comparability inside this thesis |
| **[port]** | Framework-level difference forced by the TF→PyTorch move |
| **[extrapolation]** | No upstream reference exists for this setting |
| **[extension]** | Capability this repository adds; absent upstream |
| **[open]** | Difference whose upstream justification could not be verified |

Last verified: 2026-08-23.

---

## 1. Training protocol

### 1.1 Epoch budget — 100 instead of 200 **[protocol]**

* Local: `train.epochs: 100` in all three presets.
* Upstream: `run-HT-CapsNet.sh:32` sets `epochs=200`; the notebook's own parser
  default (`main.ipynb`, parser cell) is `--epochs 100`.
* The repository-wide budget shared by every family except LH-DNN, adopted so
  cross-family and baseline-versus-lex rows are epoch-matched. Only the count
  changed: `decay_rate` and `start_epoch` are untouched, so the LR schedule is
  **truncated, not rescaled**, and training ends near `lr ≈ 9.9e-6` rather than
  `5.9e-8`.

### 1.2 Learning-rate decay rate — 0.95 instead of 0.9 **[open]**

* Local: `scheduler.decay_rate: 0.95`, `start_epoch: 10`.
* Upstream: `--lr_decay_rate` defaults to **0.9** (parser cell), and the only
  saved run arguments in the clone
  (`logs/CIFAR10/MixUp/HTR_CapsNet/backbone-custom-None-Test/args.json`) also
  record `0.9`. `--lr_decay_exe` is 10 in both.
* `docs/model_repo_differences.md` attributes 0.95 to the paper's experiment
  table. That attribution is not verifiable from the pinned source, which says
  0.9 in both places it is written down. The schedule *shape* is identical:
  `lr(e) = lr0 · rate^max(e − 10, 0)` (upstream `src/models.py:124`; local
  `HTCapsNetExponentialScheduler` in `train/runtime/optimization.py:158`).

### 1.3 Optimizer — stock PyTorch Adam instead of a Keras 2.8 Adam replica **[protocol]**

* Local: `optim.name: adam`, `lr 0.001`, `eps 1e-8`, betas `(0.9, 0.999)`,
  no weight decay.
* Upstream: `keras.optimizers.Adam()` with all defaults (`main.ipynb`, model
  cell) — `lr 0.001`, `eps 1e-7`, and the Keras 2.8 "epsilon-hat" update, in
  which epsilon is added outside the square root after bias correction.
* Earlier revisions used a `keras_adam` optimizer replicating that update
  exactly. It was dropped: the two forms differ only in where epsilon enters,
  which is not a meaningful experimental variable at these gradient scales. The
  implementation has since been removed from `train/runtime/optimization.py`,
  and `optim.name` now accepts only `sgd`, `adam`, and `adamw`.

### 1.4 No early stopping **[protocol]**

* Local: none. Every run trains the full horizon.
* Upstream: `keras.callbacks.EarlyStopping(monitor='val_<finest>_accuracy',
  patience=20, mode='max', restore_best_weights=True)`, active by default —
  `--NoEarlyStop` is a `store_true` flag that the launcher never passes.
* Consequence: upstream runs may stop at an unreported epoch and restore
  earlier weights; local runs have a fixed, reportable horizon.

### 1.5 Checkpoint selection — repository-wide rule **[protocol]**

* Local: both saved checkpoints are ranked by the exact tuple
  `(FPA, −TICE, weighted AP)`; the top-down row is evaluated from the
  top-down-selected checkpoint and the independent row from the
  independently selected one (`train/runtime/selection.py`).
* Upstream: one `ModelCheckpoint` per output monitoring `val_<output>_accuracy`
  (`main.ipynb`, training cell), and after `fit` the weights of the **finest**
  output's best epoch are reloaded for evaluation.
* Cross-family comparability of the selection rule was judged to outweigh
  fidelity to one family's callback. No per-model selection flag exists.

### 1.6 Batch size **[protocol]**

* Local: CUB-200-2011 and FGVC-Aircraft use `32`, matching upstream. CIFAR-100
  currently carries `dataloader.batch_size: 256` in the working tree.
* Upstream: `batch_size=32` everywhere (`run-HT-CapsNet.sh:33`, parser default).
* The CIFAR-100 value is a live local edit, not a documented decision; with
  Adam at a fixed `lr=0.001` an 8× larger batch changes the effective step
  count and noise scale. `tests/test_fidelity_configs.py` still asserts 32 and
  currently fails on this. Either the config or the test needs to move.

### 1.7 Seeds **[protocol]**

* Local: presets carry `train.seed: 42` (upstream's default), but multi-seed
  sweeps in `scripts/capsnet/run_ht_capsnet_baselines.sh` derive seeds from
  `BASE_SEED=0`, giving 0, 1, 2.
* Upstream: `--seed` defaults to 42; the README's reproducibility recipe uses
  101, 202, 303.
* Only the seed values differ, not the protocol.

### 1.8 Determinism **[protocol]**

* Local: `runtime.deterministic: true` is *required* by the HT-CapsNet factory
  (`models/ht_capsnet/factory.py`), cuDNN is put in deterministic mode, and
  dataloader workers are seeded individually.
* Upstream: sets `PYTHONHASHSEED`, `random`, `numpy` and `tf` seeds only; TF op
  determinism is not enabled.

Not a difference: both train in fp32 (`train.amp: false`), and neither clips
gradients.

---

## 2. Datasets, splits and taxonomies

### 2.1 CIFAR-100 validation set **[protocol]**

* Local: stratified 10% carve-out of the training split
  (`val_split_ratio: 0.1`, `split_seed: 0`); the test set is never seen during
  training or selection.
* Upstream: `CIFAR_100(...)` is called without `training_validation_split`, so
  it keeps its default `0.0` (`src/hierarchical_dataset.py:1368`), and
  `make_dataset` then sets **validation = test**
  (`src/hierarchical_dataset.py:944-946`). Every reported CIFAR checkpoint is
  therefore selected on test labels upstream.

### 2.2 CUB-200-2011 validation set **[protocol]**

* Local: 15% stratified carve-out of the official *training* split
  (`val_split_ratio: 0.15`).
* Upstream: `training_validation_split=0.499`, `validation_split_from='testing'`
  (`src/hierarchical_dataset.py:1573-1574`) — validation is 49.9% of the test
  set, and the remaining 50.1% is the reported test set.

### 2.3 CUB-200-2011 images and taxonomy **[protocol]**

* Local: the official CUB-200-2011 release with this repository's unified
  13/38/200 order → family → species taxonomy, shared with every other model
  family here.
* Upstream: a `CUB_200_2011_v0.2` tarball fetched from a private host, with
  `class_level_0/1/2` labels downloaded from
  `rbouadjenek.github.io/datasets/cu_birds_{train,test}_labels.csv`; the level
  definitions live in those CSVs and are not in the repository.
  `docs/model_repo_differences.md:264` records the paper's grouping as
  39/123/200. The level cardinalities could not be confirmed from the pinned
  source, which downloads the labels at runtime.
* Taxonomy *representation* is equivalent: upstream derives a binary
  parent-child incidence matrix from the training labels
  (`src/hierarchical_dataset.py:1003`); the port builds the same matrix from
  `taxonomy['parent_of']`.

### 2.4 FGVC-Aircraft **[extrapolation]**

* The paper and the released code cover Fashion-MNIST, Marine-Tree, CIFAR-10,
  CIFAR-100, CUB-200-2011 and Stanford Cars. **FGVC-Aircraft does not exist
  upstream.** The whole preset is a local extrapolation: it reuses the released
  path-loader preprocessing sequence and the CUB hyperparameters at 64 px.
* The preset additionally crops the 20 px FGVC copyright banner
  (`transforms.manual.crop_bottom_pixels: 20`). No upstream behaviour exists
  for this; every other model in this repository removes the banner, explicitly
  or through an evaluation centre crop.

### 2.5 CIFAR-100 hierarchy — *not* a difference

Both use 8 / 20 / 100. The upstream `coarse2_coarse1` map
(`src/hierarchical_dataset.py:1404-1414`) and the local
`B_CNN_COARSE_TO_SUPER` (`datasets/cifar100.py:22`) are the same 20-entry
B-CNN grouping, entry for entry, and both take fine → coarse from the official
CIFAR-100 archive. Native 32 px resolution also matches: `_load_` forces
CIFAR-100 to `(32, 32, 3)` regardless of `--input_size`
(`src/hierarchical_dataset.py:1832-1836`).

---

## 3. Preprocessing and augmentation

### 3.1 MixUp partner sampling **[port]**

* Local: per-example λ ~ Beta(0.2, 0.2), partner chosen by a random permutation
  **within the batch** (`mixup_mode: elem`, `mixup_pairing: random`,
  `train/mixup.py:190`), applied to every training batch; the same λ and the
  same partner map are applied to the image and to all three label levels.
* Upstream: two independently shuffled copies of the whole training pipeline
  are zipped and mixed (`src/hierarchical_dataset.py:581-585`, `mix_up` at
  `:158`), so the partner may come from anywhere in the dataset, not only the
  current batch. λ, label mixing and per-example granularity are identical.
* Both apply MixUp to 100% of training batches, and neither applies label
  smoothing (`train.smoothing: 0.0`).

### 3.2 Shuffling **[port]**

* Local: `DataLoader(shuffle=True)` — a full permutation each epoch.
* Upstream: `tf.data` `.shuffle(batch_size * 100)` — a sliding 3 200-element
  buffer, reshuffled each iteration.

### 3.3 Not differences (verified equal)

* **Standardization scope.** CIFAR reaches the upstream array branch, which
  z-scores the entire split with one scalar mean/std
  (`src/hierarchical_dataset.py:128`, applied at `:570`); the port matches with
  `normalization_scope: dataset`. CUB/Aircraft reach the path branch, where the
  same function is mapped over the *batched* tensor
  (`src/hierarchical_dataset.py:692-694`); the port matches with
  `normalization_scope: batch` via a collate-time normalizer. Computing the
  statistic on `[0,1]` tensors (torchvision `ToTensor`) instead of `[0,255]`
  arrays is exactly equivalent under z-scoring.
* **Resize chain.** Path-loaded images are decoded, converted to float, resized
  to 512 (`src/hierarchical_dataset.py:898`), batched, then resized to 64.
  Reproduced by `fixed_resize_intermediate_size: 512` plus
  `fixed_resize_antialias: false`, matching `tf.image.resize`'s default fixed
  2×2 kernel applied after the float conversion.
* **No geometric augmentation.** Neither pipeline flips, crops or jitters;
  MixUp is the only augmentation.
* **Partial batches.** `drop_last_train: false` matches `padded_batch`.

---

## 4. Model and numerics

### 4.1 Framework **[port]**

TensorFlow 2.8 / Keras → PyTorch. Consequences that required explicit work:

* **Attention.** `keras.layers.MultiHeadAttention(num_heads=16, key_dim=32,
  value_dim=32)` → `KerasMultiHeadAttention` (`models/ht_capsnet/model.py:100`),
  which keeps Keras projection semantics (Q/K/V width 512 at every level,
  independent of capsule dimension; output projected back to the query width)
  but runs `torch.nn.functional.scaled_dot_product_attention` underneath.
  Initialization reproduces Keras' rank-three `EinsumDense` Glorot fan
  calculation; plain `xavier_uniform_` on flattened `Linear` shapes is not
  equivalent.
* **Backbone.** `keras.applications.EfficientNetB7(include_top=False,
  weights='imagenet')` → timm `tf_efficientnet_b7.aa_in1k` with
  `forward_features`, `bn_momentum=0.01` (Keras 0.99) and `drop_path_rate=0.2`.
  The Keras stem (`Rescaling(1/255)` then the `Normalization` layer whose
  ImageNet state holds the standard deviations in the variance slot) is
  replicated in `_prepare_backbone_input`. `docs/model_repo_differences.md`
  records a mechanical audit of all 1 035 weight-bearing arrays.
* **Channel order.** Keras flattens NHWC feature maps into primary capsules;
  the port permutes to NHWC before the reshape
  (`models/ht_capsnet/model.py:_build_primary_caps`).

### 4.2 Capsule stack — 1:1 port, no known differences

Verified equal against the upstream layer (`src/model_arch/HTRCapsNet.py`) and
covered by `tests/test_ht_capsnet_fidelity.py`:

| Component | Value / behaviour |
| --- | --- |
| `squash`, `safe_norm` | `sqrt(‖s‖² + 1e-7)`, matching `K.epsilon()` (`:94`, `:110`) |
| Primary capsules | one squashed 8-D tensor, reshaped to 64/32-D at later levels — the source's behaviour, not Eq. 4/Fig. 2's independent primary layers (`:441`) |
| Routing | 3 iterations, agreement update by dot product, `squash` over the capsule dim (`:396`) |
| Taxonomy mask | softmax over parent capsule *lengths* first (the source, not Eq. 13's direct norm), then `range·sigmoid(T·(m − center)) + low`, tiled over primary capsules with the first-row remainder pattern (`:296`) |
| Hierarchical agreement | `votes · sigmoid(Σ gate ⊙ (votes · Wprev))`, gate init 0.5, transform init N(0, 0.1) (`:364`) |
| Per-level `LayerNorm` | eps 1e-6, applied to `caps + attention(caps)` (`:439`) |
| `W` init | N(0, 0.1) (`:247`) |
| Hyperparameters | `PCaps_dim 8`, `SCaps [64, 32, 16]`, `routings 3`, mask `0.99/0.1`, taxonomy/mask temperature 0.5, mask centre 0.5, attention 16×32 — all matching the executed `args_dict` and the saved `args.json` |

### 4.3 Capsule readout — `squash` instead of `LayerNorm` **[protocol]**

* Local: `model.attn_postprocess: squash` in all three presets.
* Upstream: `route_level` returns `layer_norms[level](output + attended)`
  (`src/model_arch/HTRCapsNet.py:439`), and the paper specifies the same in
  Eq. (18) and Algorithm 1 line 20 ("Normalization process [47] with default
  parameters [48]", [47] = Ba et al., *Layer Normalization*).

This is the one deviation that changes whether the model trains at all, so the
evidence is recorded in full.

**Why the released readout cannot work.** For a capsule `x ∈ ℝ^d`, layer
normalization divides by exactly the quantity that would otherwise set the
vector's length: with `γ = 1, β = 0`,
`‖LN(x)‖² = Σⱼ(xⱼ − μ)² / σ² = d·σ²/σ² = d`. So `‖s‖ = √d` identically, for
every class, every image and every input scale. The directions still differ —
only the magnitude is erased — but magnitude is the only thing the model reads
out: Eq. (10) is `softmax(‖s‖)` and the margin loss of Eq. (19) acts on `‖s‖`
against `m⁺ = 0.9`, `m⁻ = 0.1`. Table 1's `d_s = 64·2^-(l-1)` then fixes that
constant at 8 / 5.66 / 4. The contradiction is robust to interpretation: read
`‖·‖ₙ` as layer normalization or as L2 normalization to unit length, the class
score is constant either way, and Eq. (6)/(16)'s squash semantics
("the length of the output vector encodes the probability that the entity
exists", bounded in [0, 1]) are what Eq. (18) breaks.

**Measured in the released TensorFlow code** (pinned commit, TF 2.8.4 /
Keras 2.8.0, launcher settings, `custom` backbone; reproduction script in
`.tmp_htcapsnet_upstream/verify_defects.py`):

| Level | C | d | initial length | √d | class spread | initial margin loss |
| --- | --- | --- | --- | --- | --- | --- |
| coarse 1 | 8 | 64 | 7.99957 | 8.00000 | 1.008e-04 | 218.41 |
| coarse 2 | 20 | 32 | 5.65645 | 5.65685 | 3.763e-04 | 293.30 |
| fine | 100 | 16 | 3.99997 | 4.00000 | **1.281e-06** | **752.88** |

**Control experiment** — same 16 CIFAR-100 images, margin loss, backbone,
optimizer and dynamic level weights, 60 steps, all within the authors' own code:

| Model | loss @1 | loss @60 | accuracy @60 |
| --- | --- | --- | --- |
| `HD_CapsNet` (their earlier published model) | 1.083 | 0.000 | 1.00 / 1.00 / 1.00 |
| `HTRCapsNet copy.py` (their unused sibling, squash readout) | 0.674 | 0.634 | 1.00 / 1.00 / 0.56 |
| shipped `HTR_CapsNet` (attention + LayerNorm) | 308.467 | 362.15 | 1.00 / 0.06 / **0.00** |
| shipped `HTR_CapsNet`, only the LayerNorm swapped for squash | 0.683 | 0.082 | 1.00 / 1.00 / 1.00 |

The last row subclasses `HTR_Capsule` and replaces `self.layer_norms` in
`build()`; attention, taxonomy routing, hierarchical agreement and every
hyperparameter are untouched. Pipeline, loss, backbone and optimizer are
therefore sound, and the readout alone decides whether the model learns.

**Port fidelity** is not a factor: initial total loss is `308.467` in the
original Keras and `308.463` in this port, agreeing to four significant figures
across two frameworks with the same capsule lengths and the same failure.

The `layernorm` value remains available and reproduces the released default; the
collapsed runs it produces are reported as a reproduction result rather than
deleted.

---

## 5. Loss and level weights — no known differences

* Margin loss: `Σ_c [y·max(0, m⁺ − s)² + λ(1 − y)·max(0, s − m⁻)²]`, mean over
  the batch, with `m⁺ 0.9`, `m⁻ 0.1`, `λ 0.5`
  (`src/model_arch/HTRCapsNet.py:163`).
* During training the loss consumes raw capsule lengths; during validation and
  test it consumes the softmax output, because the source `LengthLayer` softmaxes
  when `training` is falsy (`src/model_arch/HTRCapsNet.py:140`). Argmax
  predictions are unaffected.
* Dynamic level weights reproduce the released callback exactly
  (`src/models.py:84`): `τ_i = 1 − acc_i · initial_i` (the source's
  parenthesisation, not Eq. 21's `(1 − acc_i)·initial_i`), normalized by `Στ`,
  computed from **epoch-to-date** accuracy (what Keras puts in the `logs` dict),
  and applied to the *following* batch. `initial_i` is the inverse class-share
  weighting of `src/models.py:10`.

---

## 6. Evaluation and reporting

### 6.1 Decoding modes **[extension]**

* Upstream decodes each level independently: `argmax` per output, then metrics
  (`src/metrics.py:82`, `:109`). There is no top-down decoding anywhere in the
  released code.
* Local reports **both** independent and top-down decoding, with a separate
  checkpoint per mode.

### 6.2 Metric suite **[protocol]**

| Upstream (`src/metrics.py`) | Local equivalent |
| --- | --- |
| Exact match | **FPA** — same definition (all levels correct) |
| Consistency | **1 − TICE** under independent decoding — same definition (whole predicted path valid in the tree) |
| Per-level top-k accuracy | per-level top-1 accuracy |
| Hierarchical precision / recall / F1 | not computed |
| Harmonic / arithmetic mean accuracy (k = 1, 2, 5) | not computed |
| `mAP`, hierarchical `mAP` | not computed; the local **weighted AP** is a different quantity |
| — | **AHD** (average hierarchical distance), added |

Upstream metrics are computed once, after `fit`, on the reloaded finest-best
weights. Local metrics are computed every epoch on validation and once on test
per selected checkpoint.

---

## 7. Additions absent upstream **[extension]**

These belong to the thesis framework, not to HT-CapsNet, and are **off** in the
three baseline presets:

* Hierarchical Constraint Cascade (HCC) output-space projection.
* Lexicographic projected-gradient training (`train.lexicographic`).
* Direct subspace supervision (`train.subspace_supervision`).
* Gradient / parameter / lexicographic diagnostics (`train/trunk_metrics.py`).
* Resume with strict config checking, `run_log.jsonl`, `config_resolved.yaml`,
  `test_metrics.yaml`, and config validation that rejects unknown keys.

When a lex preset enables gradient projection, HT-CapsNet's level weights scale
the per-level gradients before projection, so `weight_mode` is honoured on the
projected step exactly as it is on a plain backward pass (see
`docs/LEX_MODEL_ADAPTATION.md`). The shipped lex launcher nonetheless overrides
`weight_mode=none`, so its runs are unit-weighted by choice, not by omission.

---

## 8. Upstream defects deliberately preserved

Kept visible rather than silently fixed, because they are the released
behaviour; each is a candidate confounder when reading HT-CapsNet rows.

1. **Double normalization.** The pipeline z-scores images, then the Keras
   EfficientNet stem applies `Rescaling(1/255)` and its own ImageNet
   normalization. On CIFAR this compresses image variation by roughly 60×.
2. **Unbounded capsule lengths — corrected, not preserved.** The `LayerNorm`
   readout pins lengths at `sqrt(D)`, outside the margin loss's `[0, 1]` design
   range, giving initial losses of 600–1000 and gradient norms up to 5e4 with no
   clipping, and it is the mechanism behind the margin-collapse runs (see
   `docs/hcc_hcast_research_report.md` and the Aircraft/CUB collapse notes).
   This is the one defect the presets do **not** keep: it makes the model
   untrainable rather than merely handicapped, so they set
   `attn_postprocess: squash` (section 4.3). Runs made with `layernorm` are
   retained and reported as a reproduction of the released default.
3. **Gate/transform initialization.** The paper says the hierarchical-agreement
   gate and transform are taxonomy-biased at initialization but publishes no
   values; the code initializes every gate to 0.5 and the transform from a
   normal distribution. The port keeps the executable values.
4. **Taxonomy tiling.** The source cyclically tiles taxonomy rows over primary
   as well as previous-level capsules; the paper does not specify this mapping.
   Kept source-aligned and explicitly unresolved.

---

## 9. Summary table

| # | Item | Local | Upstream | Tag |
| --- | --- | --- | --- | --- |
| 1 | Epochs | 100 | 200 (launcher) / 100 (parser default) | protocol |
| 2 | LR decay rate | 0.95 | 0.9 | open |
| 3 | Optimizer | PyTorch Adam, eps 1e-8 | Keras 2.8 Adam, eps 1e-7 | protocol |
| 4 | Early stopping | none | patience 20, restore best | protocol |
| 5 | Checkpoint selection | `(FPA, −TICE, wAP)`, per decoder | best finest-level val accuracy | protocol |
| 6 | Batch size | 32 (CUB, Aircraft); 256 (CIFAR, working tree) | 32 | protocol |
| 7 | Seeds | 42 in presets; 0/1/2 in sweeps | 42; 101/202/303 in README | protocol |
| 8 | Determinism | cuDNN deterministic, seeded workers | seeds only | protocol |
| 9 | CIFAR validation | 10% of train | the test set | protocol |
| 10 | CUB validation | 15% of train | 49.9% of test | protocol |
| 11 | CUB labels/taxonomy | official release, 13/38/200 | `v0.2` tarball + external CSVs | protocol |
| 12 | FGVC-Aircraft | full preset | absent | extrapolation |
| 13 | Aircraft banner crop | 20 px removed | undefined | extrapolation |
| 14 | MixUp partner | permutation within batch | second shuffled dataset | port |
| 15 | Shuffle | full permutation | 3 200-element buffer | port |
| 16 | **Capsule readout** | **`squash`** | **`LayerNorm` (Eq. 18)** | **protocol** |
| 17 | Attention | SDPA, Keras projection shapes and fans | Keras `MultiHeadAttention` | port |
| 18 | Backbone | timm `tf_efficientnet_b7.aa_in1k` | Keras `EfficientNetB7` | port |
| 19 | Decoding | top-down and independent | independent only | extension |
| 20 | Metrics | FPA, AHD, TICE, weighted AP, per-level | exact match, consistency, hP/hR/hF1, mAP, mean accuracies | protocol |
| 21 | HCC / lex / subspace | available, off in baselines | absent | extension |

Everything not listed here — the capsule stack, the routing algorithm, the
taxonomy mask, the hierarchical agreement, the margin loss, the dynamic level
weights, the CIFAR-100 hierarchy, the standardization scopes and the resize
chain — was checked against the pinned source and matches.
