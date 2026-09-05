# Presets and mechanism configuration

Reference for the runnable presets under `configs/` and for the configuration
surface of each mechanism. It answers "what does this preset actually set, and
why", which the README deliberately leaves out.

The scientific description of the methods is in the thesis
(`docs/04-methodology.tex`); the fidelity discussion of each family against its
published source is in `docs/05-experiments.tex`, section *Baseline
comparability*. This file is the implementation-level companion to both.

## Contents

- [Configuration rules](#configuration-rules)
- [H-CAST](#h-cast)
- [LH-DNN](#lh-dnn)
- [HT-CapsNet](#ht-capsnet)
- [HRN](#hrn)
- [Hier-COS](#hier-cos)

## Configuration rules

Runnable configs use these required top-level sections:

```text
model, dataset, dataloader, train, optim, scheduler, runtime
```

The optional top-level section is `hcc`. OmegaConf environment interpolation and
command-line dotlist overrides are resolved before validation.

Configuration is fail-fast:

- unknown keys are rejected;
- level-name count must equal `dataset.hierarchy_depth`;
- model/dataset/HCC/lexicographic incompatibilities are rejected;
- `dataloader.drop_last_eval: true` is rejected by the corrected protocol;
- explicit annotation paths are mandatory and missing files fail at dataset
  construction;
- HT-CapsNet, LH-DNN, and Hier-COS require a complete taxonomy.

The commented fragments under `configs/templates/` document the accepted fields.
They are not standalone runnable experiments.

## H-CAST

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
`[p123, p12, p1]` behavior. Baseline configs list every nonempty block so their
diagnostics cover the complete support partition; lex launchers override that
list with only the multi-objective blocks on which projection is defined. Thus
H-CAST uses `[p123,p12,p1,p2,p3]` versus `[p123,p12]`, HT-CapsNet and HRN use
`[p123,p23,p3]` versus `[p123,p23]`, and Hier-COS uses `[p123]` in both cases.
LH-DNN's baseline-only partition is `[p123,p1,p2,p3]`.

## LH-DNN

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

## HT-CapsNet

- `configs/capsnet/capsnet_cifar100.yaml`
- `configs/capsnet/capsnet_cub200.yaml`
- `configs/capsnet/capsnet_aircraft.yaml`

The presets port the released TensorFlow architecture and callback behavior:
one squashed primary-capsule tensor is reshaped at later levels, parent capsule
lengths are softmaxed before taxonomy masking, and dynamic loss weights use the
released callback's parentheses. The selected paper hyperparameters include
taxonomy temperature `0.5`, mask bounds `0.99/0.1`, and learning-rate decay
`0.95` after epoch 10; the released launcher instead inherits parser defaults
of `0.9` for both the upper mask bound and decay factor. The port retains 16×32
Keras-shaped attention with rank-three Keras Glorot initialization, per-example
MixUp, capsule margin loss, and next-batch dynamic level weights. Optimisation
uses stock PyTorch Adam with `eps=1e-8`, which differs from the released
Keras 2.8 Adam update and its default `eps=1e-7`. Local execution is
deterministic. The presets run 100 epochs rather than the paper's 200, under the repository-wide
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
file contain material contradictions; the resulting deviations, the LayerNorm
readout defect and the squash correction are derived in `docs/05-experiments.tex`,
section *Baseline comparability*.
Native lexicographic HT-CapsNet uses the same three capsule margin losses. The
lex launcher selects `model.loss.weight_mode: none`, which is the existing
unit-weight mode; the paper-baseline configs retain their dynamic weighting.

## HRN

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

The optional `model.projection.enabled: true` path applies the LH-DNN
projection to a shared vector derived from HRN's trunk feature map. The native
map from that spatial tensor to a level's logits is not of the LH form, so
enabling the projection also replaces the output branches, with no separate
switches:

- HRN's existing global average pool is relocated from the three individual
  branches to one position immediately after the trunk. It changes
  `[B,D,H,W]` into `[B,D]`: pooling removes the spatial dimensions and does not
  reduce `D` to a class count. Pooling is a tractable implementation choice, not
  a requirement of the projection algebra; flattening the spatial map would
  instead require heads and projector systems of width `D*H*W`.
- A shared `shared_linear` + `shared_relu` pair is inserted at the branching
  point, named and shaped after LH-DNN's own. This is the layer the guarantee
  bites on: without it every shared parameter is convolutional trunk, and
  orthogonality of the fine gradient at the branching point does not survive the
  trunk's Gram factor. Stepping the convolutional trunk therefore remains
  outside the branch-point guarantee. LH-DNN's `shared_linear` and Hier-COS's
  `f_theta` play the corresponding role, and their convolutional stacks are
  likewise uncovered. `rho' = 1[pre_activation > 0]` is read at the ReLU's
  pre-activation, so it is idempotent in the projector construction.
- Each complete RFM--FC--classifier branch is replaced by one direct
  `Linear(D, C_l)` head. The code does not retain a chain of linear stand-ins:
  without intervening nonlinearities that chain would be exactly one affine map
  but would introduce redundant parameters and different deep-linear
  optimisation. This substitution removes the native branch convolutions,
  BatchNorm, ReLU, ELU, dropout, and the `embedding_dim` bottleneck from the
  projected arm; those capacity changes remain confounded with the projection
  and must not be attributed to it alone.
- The coarse-to-fine residual moves from the embedding to the score and becomes
  LH-DNN's advantage: every level adds its detached parent's advantage score,
  gathered through the taxonomy. Enabling the projection therefore requires a
  taxonomy with parent mappings.

`A = [W_1; ...; W_(l-1)] * rho'` is then built directly from the preceding
heads' weights. The detached advantage remains present in the forward score but
has zero derivative through the current level, so each branch-local gradient
returns through its own `W_l`. The backward-only `z - c + sg(c)` construction
leaves forward values untouched, the shared width must exceed
`sum(num_classes_per_level[:-1])`, `model.dropout` must be 0, and the path is
rejected together with `hcc.enabled: true`.

The fine level retains two direct `Linear(D, C_3)` heads, one for the tree term
and one for leaf CE/reporting. Nothing follows the fine level, so neither enters
a later projector. Both read the same projected input and receive the same
parent advantage.

Replacing every native branch by a direct head is a deliberate departure from
paper-HRN. The result is HRN's trunk with LH-DNN's minimal head structure; it is
a study of the projection adaptation package, not an HRN reproduction.
Checkpoints produced by the former factorised-linear projected branch are not
state-dict compatible with this direct-head implementation and must be retrained.

The launcher is `scripts/hrn/run_hrn_lhdnn_projection.sh`, which names these
runs `projection`. It keeps `model.loss: level_conditional` and
`train.gradient_blocks: [p123, p23, p3]`, matching the baseline and HCC arms:
the conditional terms stay triangular under the projection, so the three
gradient-support blocks are unchanged and the diagnostics remain comparable
across arms.

## Hier-COS

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
Set `model.projection.advantage_enabled: true` to build a local Hier-COS
score-space advantage. After the per-level fixed frame, the model takes the
absolute node coordinates used by the level loss and recursively adds the
detached parent score to every child. Cross-entropy and both evaluation
decoders consume these advantage scores; the Hier-COS regularizer remains on
the native absolute coordinates. The scores are not fed back through the
taxonomy-subspace norms, which would count parent evidence again. This path
requires `model.loss: level_softmax_ce_reg`.

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
