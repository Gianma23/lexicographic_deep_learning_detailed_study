# Lexicographic Mode: Per-Model Adaptation, Constraints, and Quirks

How explicit gradient-space lexicographic training (`train.lexicographic.enabled`)
is adapted to each model family, what each adaptation requires in the config, and
which behaviours change silently.

This file complements two others and does not repeat them:

- [`GRADIENT_PARAM_DIAGNOSTIC_LOGS.md`](GRADIENT_PARAM_DIAGNOSTIC_LOGS.md): the
  `run_log.jsonl` key glossary (canonical `p...` blocks and deprecated `t...`
  compatibility aliases).
- `README.md`: user-facing preset descriptions.

Scope note: lexicographic mode here means **explicit parameter-gradient
projection**. It is not HCC (an output-space affine constraint) and not the
LH-DNN-style branch projection (`model.projection.*`), which is a forward-path
reparameterization with a backward-only effect. The three are distinct
mechanisms and are not interchangeable.

## 1. The engine contract

Everything below follows from one fact in `train/engine.py:120-169`.

When `train.lexicographic.enabled` is true:

1. `loss.backward()` is **never called**.
2. Three separate `torch.autograd.grad` passes are taken, one per entry of
   `loss_aux["level_losses"]`, over all trainable parameters
   (`train/lexicographic/gradients.py:257-276`).
3. Those three gradients are projected against each other, summed, and written
   directly onto `param.grad` via `assign_grads_to_params`.
4. `optimizer.step()` runs on that sum.

**Consequence: `loss_aux["level_losses"]` is the only channel into the update.**
Any term contributing to the reported `total` loss but not present inside those
three tensors is logged and then discarded from the optimizer step. Every
per-model constraint below exists to make that channel well-defined.

Projection has no onset control: when `enabled` is true it is active from epoch 0
to the end of the run, so optimization semantics never change partway through.

Validation runs in two places, both of which must pass:

- statically, from `validate_config` (`train/config_validation.py:881-941`);
- at the first training batch, from `validate_lexicographic_requirements`
  (`train/lexicographic/config.py:126-181`).

## 2. Universal constraints

| Constraint | Where enforced |
| --- | --- |
| Hierarchy depth must be exactly 3 | `config_validation.py:909`, `config.py:177` |
| Exactly 3 differentiable, scalar, `requires_grad` level losses | `gradients.py:35-50` |
| Model must be `hcast`, `hiercos`, `ht_capsnet`, or `hrn` | `config.py` |
| `model.name: lhdnn` is rejected unconditionally | `config.py:132-133` |
| Hier-COS `model.projection.enabled: true` is mutually exclusive with lex | `config_validation.py:903-907` |
| `projection_mode` in `coarse_first`, `fine_first` | `config.py:9` |
| `eps > 0` | `config_validation.py:925` |

## 3. Per-model adaptation

### 3.1 H-CAST (`model.name: hcast`)

**Level objectives.** The three raw per-level cross-entropies from
`_level_losses_from_scores` (`models/hcast/losses.py:144-166`), each on its own
per-level softmax. When HCC is active they are computed on the HCC-projected
logits, because `select_effective_logits` chooses the same score source for both.

**Required config.**

```yaml
model:
  loss:
    globalkl: false     # required
train:
  lexicographic:
    enabled: true
```

Presets: none. `configs/hcast/` keeps only the three base presets; the
lexicographic block is applied as CLI overrides by
`scripts/hcast/run_hcast_lex.sh`, which also passes `model.loss.globalkl=false`.

**Why `globalkl: false` is required.** `_global_kl_loss`
(`models/hcast/losses.py:114-141`) contributes to `total` but is not one of the
three `level_losses`, so under lex it would be silently dropped from the update.

**Why it is not simply decomposed.** The global KL *can* be split exactly into
three level terms: its target is `t = Σ_l w_l · e_{n_l}` with uniform
`w_l = 1/L`, so `KL(t‖p) = H(t) − Σ_l w_l log p_{n_l}`, where `H(t) = −log L` is
parameter-independent. This is the same algebra that turns Hier-COS's `kl_reg`
into `global_softmax_ce_reg`. The decomposition is exact, but the resulting
terms share one softmax over the concatenated hierarchy, so every level's
gradient reaches every head through the log-partition. That would collapse
H-CAST's competing support partition to `p123` only (Section 4), which is the
specific property
that makes H-CAST informative for this study. `globalkl: false` is therefore a
deliberate design choice, not only a plumbing convenience.

**Level weights reach the step.** `aux["level_losses"]` holds the *weighted*
tensors, so `sum(level_losses) == total` exactly whenever `globalkl: false`
(`models/hcast/losses.py:243-278`). `model.loss.level_weighting.mode: dynamic`
therefore changes the projected update as well as the logged scalar, and
projection is a plugin over the configured objective rather than a substitute
for it. The `loss_level_*` metrics stay **unweighted** and are reported next to
`loss_weight_level_*`, so the per-level scalars remain comparable across weight
modes. Before 2026-08-30 this channel carried the unweighted tensors and level
weights were a silent no-op under lex; runs from before that date are
effectively unit-weighted regardless of what `config_resolved.yaml` records.

### 3.2 Hier-COS (`model.name: hiercos`)

**Level objectives.** `level_losses[l] = ce_level_l + alpha * reg_level_l`
(`models/hiercos/losses.py`). Level weights are folded **inside**
`ce_level_losses` by `_weighted_target_ce_level_losses`, so they survive into
the update; H-CAST and HT-CapsNet now follow the same convention.

**Required config.**

```yaml
model:
  loss: global_softmax_ce_reg   # or level_softmax_ce_reg
  projection:
    enabled: false              # mutually exclusive with lex
```

Launcher: `scripts/hiercos/run_hiercos_lex.sh`.

**Why the loss mode is coerced.** The upstream-aligned default `kl_reg` produces
a single KL over the whole node vector and exposes no per-level tensors
(`models/hiercos/losses.py`). `global_softmax_ce_reg` is its exact
per-level regrouping; `level_softmax_ce_reg` additionally replaces the global
softmax with per-level softmaxes and is therefore a genuinely different
objective, not a regrouping.

**Comparability warning.** Baseline Hier-COS presets default to
`model.loss: kl_reg` with `model.weight_mode: kl_leaf`. A lex run is neither, so
a lex-vs-baseline table built from default presets confounds the lexicographic
effect with both a loss-mode change and a weight change. See
`docs/hiercos_level_weights.md` Section 10b for the measured decomposition; the
weight effect alone is worth roughly 1-2 pp of fine accuracy.

### 3.3 HT-CapsNet (`model.name: ht_capsnet`)

**Level objectives.** The three raw capsule margin losses
(`models/ht_capsnet/losses.py:207-227`), computed on capsule norms.

**Required config.** No loss-mode coercion is enforced, but the launcher
`scripts/capsnet/run_ht_capsnet_lex.sh` deliberately sets
`model.loss.weight_mode=none`, while the baseline preset ships
`weight_mode: dynamic` (`configs/capsnet/capsnet_cifar100.yaml:46`). It also
passes `train.epochs=100`, which is now the baseline preset's value as well, so
the arms are training-budget matched; the override is kept as an explicit guard.
Comparisons with the native baseline therefore remain not level-weight matched,
but are no longer budget-mismatched. Since the weights now reach the step (see
below), dropping the `weight_mode=none` override is what makes the two arms
weight-matched — at the cost of superseding every earlier HT-CapsNet lex run.

**Level weights reach the step.** `aux["level_losses"]` holds the *weighted*
margins, so `sum(level_losses) == total` exactly
(`models/ht_capsnet/losses.py:242-266`), and the `level_loss_weights` that
`post_optimizer_step` refreshes every batch under `weight_mode: dynamic` now
scale the projected per-level gradients. The `loss_level_*` metrics stay
**unweighted** next to `loss_weight_level_*`. Before 2026-08-30 the channel
carried the unweighted margins, so a lex run at `dynamic` silently produced a
unit-weight update; runs from before that date are effectively unit-weighted.

### 3.4 HRN (`model.name: hrn`)

**Level objectives.** `level_losses = [tree_0, tree_1, tree_2 + ce_loss]`
(`models/hrn/losses.py`) — the three *conditional* tree NLLs produced by
`_level_conditional_tree_losses`, with the auxiliary leaf CE head folded into the
fine level.

`_hierarchical_loss` returns `-log P(state ∈ subtree(u))` for an observed node
`u`. Within one sample the observed nodes are nested, so the chain rule splits
the leaf term into three conditionals — the coarse subtree, the middle subtree
given the coarse one, the leaf given the middle one — computed as the successive
differences of the three marginal terms. They telescope: the three tensors sum
exactly to the `native` total, and `autograd.grad` of that sum matches
`native`'s gradient to `0.0`. With projection disabled, `level_conditional` and
`native` are the same run.

**Required config.**

```yaml
model:
  loss: level_conditional # required; `native` exposes no per-level tensors
```

Launcher: `scripts/hrn/run_hrn_lex.sh`.

**Quirk — the auxiliary leaf head is projected as a fine-level term.**
`classifier_3_1` is trained through `ce_loss`, which sits inside the fine level
objective. Lexicographic projection therefore treats that auxiliary head as part
of the fine hierarchical objective and orthogonalizes it against the coarse and
mid gradients on the selected `p123` and `p23` competing blocks. This is a
modelling choice, not a neutral wiring detail, and should be stated when
reporting HRN lex results.

**Comparability.** Unlike the previous `level_marginal` mode (removed), which
summed the three *cumulative* marginals and so optimised a `(3, 2, 1)`-weighted
objective, `level_conditional` leaves the objective untouched. The matched
baseline for an HRN lex run is therefore a plain `native` run, and the projection
is the only difference between the two.

### 3.5 LH-DNN — not supported

Rejected unconditionally (`config.py`, `config_validation.py`). LH-DNN's own branch
projection (`z - c + sg(c)`, `models/lhdnn/model.py:285`) is an always-on,
per-sample, branch-point gradient intervention. Stacking a second, global
parameter-space projection on top would make the resulting update attributable to
neither mechanism.

## 4. Gradient-support partition per model

Exact blocks are derived at runtime from which level gradients are non-`None`
per parameter. Their canonical name is `p` followed by the active level indices:
`p123`, `p12`, `p23`, `p1`, and so on. The deprecated log aliases are
`t1 = p123`, `t2 = p12`, and `t3 = p1`.

The partition is a property of the **architecture**, not of the config, and it
differs sharply across models.

**H-CAST's competing blocks are `p123` and `p12`.** Its heads read different
trunk depths — `head(out2)` fine, `family_head(out3)` mid, and
`manufacturer_head(out4)` coarse
(`models/hcast/internal/cast_deit_hier.py:221-223`) — so blocks below each head
are exclusive to the coarser levels. The complete support partition also
contains the singleton blocks `p1`, `p2`, and `p3`; no inter-objective
projection is defined on them. Baseline configs diagnose all five nonempty
blocks, whereas the lex launcher selects only `p123` and `p12`. Measured on
`/scratch/g.saggini1/outputs/hcast_cifar100_lex/seed_0`, epoch 2, seed 0:

| | `p123` (`t1`) | `p12` (`t2`) | `p1` (`t3`) |
| --- | --- | --- | --- |
| `param_norm` | 321.2 | 298.6 | 198.2 |

The projection also behaves differently per block in that run:
`cos_t1_mid_coarse = 0.631` versus `cos_t2_mid_coarse = 0.152`, i.e. mid and
coarse gradients are strongly aligned on the fully shared trunk but nearly
orthogonal on the coarse+mid-only trunk.

**Hier-COS reduces to `p123` as its only competing block.** It has a single
shared backbone with no branches and a frozen fixed frame, so all trainable
parameters receive all three level gradients. Measured on the Aircraft lex runs,
epoch 2, seed 0: `param_norm_t1 = param_norm_t2t1 = param_norm_t3t2t1 = 98.6`
(`global_softmax_ce_reg`) and `98.7` (`level_softmax_ce_reg`) — identical under
both loss modes, confirming the degeneracy is architectural rather than
loss-induced. This is expected for Hier-COS, not a defect.

**HRN uses `p123`, `p23`, and `p3`.** The conditional loss construction detaches
scores finer than the level being defined, so its support is triangular. The
shared trunk and coarse branch are reached by all three level objectives
(`p123`), the middle branch by the middle and fine objectives (`p23`), and the
fine branch by the fine objective alone (`p3`). The matched baseline and lex
settings use different block lists by design: the baseline diagnoses
`[p123,p23,p3]`, while the lex launcher selects the projection-relevant
`[p123,p23]`. The singleton `p3` needs no inter-objective projection.

**HT-CapsNet instead uses `p123`, `p23`, and `p3`.** Its capsule stages are
chained coarse-to-fine: the coarse stage is reached by all three losses, the
middle stage by middle and fine, and the final stage by the fine loss only.
The baseline configs therefore diagnose `[p123,p23,p3]`, while the lex launcher
selects `[p123,p23]`; coarse-first mode additionally projects the fine gradient
against the middle gradient on `p23`.

**Parameters reached by no level objective are outside the `p...` partition.**
The executable support audit also found two such cases: HT-CapsNet's three
`post_attn_norms` modules are inactive when `attn_postprocess: squash`, and the
CIFAR WideResNet used by HRN and Hier-COS does not use the `bn1` affine
parameters in the first unequal-width block of stages 2 and 3. Their gradients
are `None` for all three objectives, so they require no projection and produce
no block diagnostics. They nevertheless remain registered as trainable
parameters and are therefore relevant to raw parameter-count reporting.

Practical consequences:

- Mask-dependent canonical diagnostic keys are emitted only when the selected
  mask is non-empty, so the absence of (for example) `grad_norm_p23_*` is not an
  error when that support does not occur.
- `projection_mode: coarse_first` means structurally different things across
  models. On H-CAST it acts on `p123` and `p12`; on HT-CapsNet and HRN it acts
  on `p123` and `p23`; on Hier-COS it reduces to `p123`.
- These numbers are single-seed, single-epoch snapshots. Confirm stability across
  epochs and seeds before treating them as load-bearing thesis claims.

## 5. Cross-cutting quirks

**Cost.** Three full `autograd.grad` passes per batch with `retain_graph`
(`gradients.py:257-276`), so roughly 3x backward cost. Note that
`compute_trunk_grad_metrics` runs those passes on **every** batch of **every**
run whose loss exposes `level_losses`, including non-lex baselines, where
`loss.backward()` then adds a fourth traversal (`engine.py:104-110`). Hier-COS
`kl_reg` and HRN `native` are the only presets that avoid this, because their
loss modules emit no `level_losses`.

`_project_onto_reference` (`gradients.py:339`) is synchronization-free: the
projection coefficient and applied flag stay on-device as 0-dim tensors, and the
flags are reduced to floats inside the single batched metric transfer.

**AMP.** Supported but non-standard: a dummy `scaler.scale(torch.ones(...))`
keeps `GradScaler` bookkeeping consistent while gradients are assigned manually
(`engine.py:123-127`). Projection coefficients and metrics stay in unscaled
units; only the returned gradients are rescaled.

**Gradient accumulation and DDP.** Incompatible in principle —
`assign_grads_to_params` overwrites `param.grad` outright rather than
accumulating, and DDP's allreduce hooks fire on `.backward()`, which never runs.
Neither is currently used in this repository, so neither is a live concern.

**Logged versus optimized loss.** `loss_dict["total"]` is the conventional loss,
which under lex is not what the optimizer stepped on. Compare `loss_level_*`
against the projected gradient diagnostics rather than reading `total` as the
training objective.

## 6. Summary table

| Model | Lex supported | Required lex loss config | Baseline diagnostic blocks | Lex projection blocks | Level weights reach the step |
| --- | --- | --- | --- | --- | --- |
| H-CAST | yes | `loss.globalkl: false` | `p123`+`p12`+`p1`+`p2`+`p3` | `p123`+`p12` | yes (since 2026-08-30) |
| Hier-COS | yes | `loss: global_softmax_ce_reg` or `level_softmax_ce_reg`; `projection.enabled: false` | `p123` | `p123` | yes |
| HT-CapsNet | yes | none enforced; launcher sets `weight_mode: none` | `p123`+`p23`+`p3` | `p123`+`p23` | yes (since 2026-08-30) |
| HRN | yes | `loss: level_conditional` | `p123`+`p23`+`p3` | `p123`+`p23` | n/a (unweighted objective) |
| LH-DNN | **no** | — | `p123`+`p1`+`p2`+`p3` | — | — |
