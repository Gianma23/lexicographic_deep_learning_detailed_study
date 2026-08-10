# Lexicographic Mode: Per-Model Adaptation, Constraints, and Quirks

How explicit gradient-space lexicographic training (`train.lexicographic.enabled`)
is adapted to each model family, what each adaptation requires in the config, and
which behaviours change silently.

This file complements two others and does not repeat them:

- [`GRADIENT_PARAM_DIAGNOSTIC_LOGS.md`](GRADIENT_PARAM_DIAGNOSTIC_LOGS.md): the
  `run_log.jsonl` key glossary (`grad_norm_t*`, `cos_t*`, `post_*`).
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
| `projection_rule` in `orthogonalize_all`, `conflict_only` | `config.py:10` |
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
H-CAST's trunk partition to `t1` only (Section 4), which is the specific property
that makes H-CAST informative for this study. `globalkl: false` is therefore a
deliberate design choice, not only a plumbing convenience.

**Quirk — level weights are dropped.** `total` uses the *weighted* level losses
but `aux["level_losses"]` holds the **unweighted** tensors
(`models/hcast/losses.py:243-274`). Under lex, `model.loss.level_weighting.mode:
dynamic` therefore has no effect on the update, only on the logged scalar. The
shipped lex presets set `mode: static` (all weights 1.0), so they are unaffected;
deviating from that is the trap.

### 3.2 Hier-COS (`model.name: hiercos`)

**Level objectives.** `level_losses[l] = ce_level_l + alpha * reg_level_l`
(`models/hiercos/losses.py`). Level weights are folded **inside**
`ce_level_losses` by `_weighted_target_ce_level_losses`, so unlike H-CAST and
HT-CapsNet they do survive into the update.

**Required config.**

```yaml
model:
  loss: global_softmax_ce_reg   # or level_softmax_ce_reg
  projection:
    enabled: false              # mutually exclusive with lex
```

Launchers: `scripts/hiercos/run_hiercos_lex_orthogonalize_all.sh`,
`scripts/hiercos/run_hiercos_lex_conflict_only.sh`.

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
`scripts/capsnet/run_ht_capsnet_lex.sh:153` sets `model.loss.weight_mode=none`
deliberately.

**Quirk — level weights are dropped, and `dynamic` is misleading.** As with
H-CAST, `total` is the weighted sum while `aux["level_losses"]` holds the
unweighted tensors. The baseline preset ships `weight_mode: dynamic`
(`configs/capsnet/capsnet_cifar100.yaml:29`), and under lex the
`post_optimizer_step` callback keeps updating `level_loss_weights` every batch
with **no effect on the optimizer step**. Static validation accepts this
combination without warning (`tests/test_lexicographic_model_support.py:44-46`).
Using the launcher's `weight_mode=none` makes the actual behaviour explicit;
running lex at `dynamic` silently produces a unit-weight run.

### 3.4 HRN (`model.name: hrn`)

**Level objectives.** `level_losses = [tree_0, tree_1, tree_2 + ce_loss]`
(`models/hrn/losses.py:257-261`) — the three tree-marginal NLLs, with the
auxiliary leaf CE head folded into the fine level.

**Required config.**

```yaml
model:
  loss: level_marginal    # required; `native` exposes no per-level tensors
```

Launcher: `scripts/hrn/run_hrn_lex.sh:163`.

**Quirk — the auxiliary leaf head is projected as a fine-level term.**
`classifier_3_1` is trained through `ce_loss`, which sits inside the fine level
objective. Lexicographic projection therefore treats that auxiliary head as part
of the fine hierarchical objective and will orthogonalize it against the coarse
and mid gradients. This is a modelling choice, not a neutral wiring detail, and
should be stated when reporting HRN lex results.

**Comparability warning.** `level_marginal` is not HRN's paper-aligned objective
(`native`: leaf-observed joint tree marginal plus leaf CE). An HRN lex-vs-baseline
comparison is only clean if the baseline also runs at `level_marginal`.

### 3.5 LH-DNN — not supported

Rejected unconditionally (`config.py`, `config_validation.py`). LH-DNN's own branch
projection (`z - c + sg(c)`, `models/lhdnn/model.py:285`) is an always-on,
per-sample, branch-point gradient intervention. Stacking a second, global
parameter-space projection on top would make the resulting update attributable to
neither mechanism.

## 4. Trunk partition per model

`t1`/`t2`/`t3` are derived at runtime from which level gradients are non-`None`
per parameter (`gradients.py:239-254`):

- `t1`: receives coarse + mid + fine gradients
- `t2`: coarse + mid only
- `t3`: coarse only

The partition is a property of the **architecture**, not of the config, and it
differs sharply across models.

**H-CAST is the only model with a non-degenerate partition.** Its heads read
different trunk depths — `head(out2)` fine, `family_head(out3)` mid,
`manufacturer_head(out4)` coarse
(`models/hcast/internal/cast_deit_hier.py:221-223`) — so blocks below each head
are exclusive to the coarser levels. Measured on
`/scratch/g.saggini1/outputs/hcast_cifar100_lex/seed_0`, epoch 2, seed 0:

| | `t1` | `t2` | `t3` |
| --- | --- | --- | --- |
| `param_norm` | 321.2 | 298.6 | 198.2 |

The projection also behaves differently per block in that run:
`cos_t1_mid_coarse = 0.631` versus `cos_t2_mid_coarse = 0.152`, i.e. mid and
coarse gradients are strongly aligned on the fully shared trunk but nearly
orthogonal on the coarse+mid-only trunk.

**Every other model collapses to `t1` only.** Hier-COS has a single shared
backbone with no branches and a frozen fixed frame, so all trainable parameters
receive all three level gradients. Measured on the Aircraft lex runs, epoch 2,
seed 0: `param_norm_t1 = param_norm_t2t1 = param_norm_t3t2t1 = 98.6`
(`global_softmax_ce_reg`) and `98.7` (`level_softmax_ce_reg`) — identical under
both loss modes, confirming the degeneracy is architectural rather than
loss-induced. This is expected for Hier-COS, not a defect.

Practical consequences:

- Mask-dependent diagnostic keys are emitted only when the mask is non-empty
  (`gradients.py:475-519`), so `grad_norm_t2_*` and `grad_norm_t3_*` simply do
  not appear in single-trunk runs. Their absence is not an error.
- `projection_mode: coarse_first` means structurally different things across
  models. On H-CAST it exploits a real nested parameter partition; elsewhere it
  reduces to one global projection over an undifferentiated parameter block.
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

| Model | Supported | Required loss config | Trunk partition | Level weights reach the step |
| --- | --- | --- | --- | --- |
| H-CAST | yes | `loss.globalkl: false` | `t1`+`t2`+`t3` | no (unweighted tensors) |
| Hier-COS | yes | `loss: global_softmax_ce_reg` or `level_softmax_ce_reg`; `projection.enabled: false` | `t1` only | yes |
| HT-CapsNet | yes | none enforced; launcher sets `weight_mode: none` | `t1` only | no (unweighted tensors) |
| HRN | yes | `loss: level_marginal` | `t1` only | n/a (unweighted objective) |
| LH-DNN | **no** | — | — | — |
