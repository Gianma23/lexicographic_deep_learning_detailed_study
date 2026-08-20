# Gradient and Parameter Diagnostic Logs (Current Code + Recent Run Audit)

This file documents the `train_metrics` keys written to `run_log.jsonl` by the **current** training code (`train/lexicographic/gradients.py`, `train/engine.py`).

## Scope

- Levels:
  - `coarse` = level 0
  - `mid` = level 1
  - `fine` = level 2
- Exact gradient-support blocks use `p` followed by the levels whose losses
  reach the parameter. The complete three-level vocabulary is `p1`, `p2`,
  `p3`, `p12`, `p13`, `p23`, and `p123`.
- Examples:
  - `p123`: params that receive `coarse + mid + fine` gradients
  - `p12`: params that receive `coarse + mid` gradients only
  - `p23`: params that receive `mid + fine` gradients only
  - `p3`: params that receive the fine gradient only
- `train.gradient_blocks` selects which exact blocks are diagnosed and,
  when lexicographic mode is active, projected. Its omitted-config default is
  `[p123, p12, p1]`, which reproduces the historical H-CAST partition.

The three old exact-block names remain emitted as deprecated aliases:
`t1 = p123`, `t2 = p12`, and `t3 = p1`. Historical union aliases
`t2t1 = p12 ∪ p123` and `t3t2t1 = p1 ∪ p12 ∪ p123` are also retained when
their component blocks are selected. Old `run_log.jsonl` files therefore remain
usable without conversion.

Important: mask-dependent keys are emitted only when the corresponding mask is active (`any(mask)` in code).
Pairwise cosine keys listed below are emitted for every epoch summary.
Per-model adaptation details, config constraints, and silent quirks are in
[`LEX_MODEL_ADAPTATION.md`](LEX_MODEL_ADAPTATION.md). Summarized, the native
model requirements are:

- H-CAST exposes its three raw per-level objectives and requires
  `model.loss.globalkl: false` in lex mode.
- HT-CapsNet exposes its three raw capsule margin losses. Its lex launcher uses
  `model.loss.weight_mode: none` for unit weights; other configured scalar
  weights do not change the raw tensors consumed by lex projection.
- HRN requires `model.loss: level_marginal`, which exposes coarse and middle
  tree-marginal NLLs plus a fine tree-marginal NLL with the original leaf CE
  term.
- Hier-COS requires `model.loss: global_softmax_ce_reg` or
  `level_softmax_ce_reg`; plain `kl_reg` does not expose per-level loss tensors.
  Both decomposed modes use weighted target CE plus the same unweighted level
  regularizer and differ only by global-taxonomy versus per-level softmax
  normalization.
- LH-DNN is not supported.

Logged `loss_level_*` values match the tensors used by lexicographic
optimization.

## Canonical `p...` metric keys

For every selected, non-empty block `pA`, the current code emits:

- `grad_norm_pA_<level>` for every level in `A`;
- `cos_pA_<lower>_<higher>` for every pair of gradients present on the block;
- `param_norm_pA` and `delta_param_norm_pA`.

For example, selecting `p23` adds `grad_norm_p23_mid`,
`grad_norm_p23_fine`, `cos_p23_fine_mid`, `param_norm_p23`, and
`delta_param_norm_p23`. Selecting singleton `p3` adds its norm metrics but no
cosine and no projection, because no objectives compete there.

With lexicographic mode and projection logging enabled, the corresponding
post-update diagnostics use `post_`:

- `post_projection_applied_pA_<target>_<reference>`;
- `post_grad_norm_pA_<level>`;
- `post_cos_pA_<target>_<reference>`.

On a three-gradient block the final target is projected against the resultant
of the already processed higher-priority gradients, denoted by `higher`; for
example `post_cos_p123_fine_higher` in coarse-first mode. On `p23`,
coarse-first mode produces `post_projection_applied_p23_fine_mid` and
`post_cos_p23_fine_mid`.

## A) Deprecated compatibility metrics (non-lex and lex)

The keys below are still emitted alongside the canonical `p...` keys so old
analysis notebooks can read new runs. They are not the naming convention for
new analysis code.

### Pre-gradient norms

- `grad_norm_t3t2t1_coarse`
- `grad_norm_t3_coarse`
- `grad_norm_t2t1_coarse`
- `grad_norm_t2t1_mid`
- `grad_norm_t2_coarse`
- `grad_norm_t2_mid`
- `grad_norm_t1_coarse`
- `grad_norm_t1_mid`
- `grad_norm_t1_fine`

### Pre-projection gradient cosines

- `cos_t2_mid_coarse`
- `cos_t1_mid_coarse`
- `cos_t2t1_mid_coarse`
- `cos_t1_fine_higher`
- `cos_t1_fine_coarse`
- `cos_t1_fine_mid`

### Parameter norms and per-epoch deltas

- `param_norm_t3t2t1`, `param_norm_t3`, `param_norm_t2t1`, `param_norm_t2`, `param_norm_t1`
- `delta_param_norm_t3t2t1`, `delta_param_norm_t3`, `delta_param_norm_t2t1`, `delta_param_norm_t2`, `delta_param_norm_t1`

Notes:
- `delta_param_norm_*` is the norm of `(end_of_epoch - start_of_epoch)`.
- There are no `post_param_*` metrics.

## B) Lex-Only Additional Metrics (`train.lexicographic.enabled=true` and `log_metrics=true`)

The expectations below describe the default
`train.lexicographic.projection_mode: coarse_first` behavior. Applicable
gradient pairs are orthogonalized unconditionally.

Post-projection applied flags (`post_` prefix):

- `post_projection_applied_t2_mid_coarse`
- `post_projection_applied_t1_mid_coarse`
- `post_projection_applied_t1_fine_higher`
- `post_projection_applied_t1_mid_fine`
- `post_projection_applied_t2_coarse_mid`
- `post_projection_applied_t1_coarse_higher`

Lex projection mode indicators:

- `lex_projection_mode_coarse_first`
- `lex_projection_mode_fine_first`

Lex cosine diagnostics (post naming):

- `post_cos_t2_mid_proj_coarse`
- `post_cos_t1_mid_proj_coarse`
- `post_cos_t2t1_mid_proj_coarse`
- `post_cos_t1_fine_proj_higher`
- `post_cos_t1_fine_proj_coarse`
- `post_cos_t1_fine_proj_mid_proj`

Post-lex gradient norms:

- `post_grad_norm_t3t2t1_coarse`
- `post_grad_norm_t3_coarse`
- `post_grad_norm_t2t1_coarse`
- `post_grad_norm_t2t1_mid`
- `post_grad_norm_t2_coarse`
- `post_grad_norm_t2_mid`
- `post_grad_norm_t1_coarse`
- `post_grad_norm_t1_mid`
- `post_grad_norm_t1_fine`

Semantics:

- `post_*_coarse`: coarse component after lex composition (equal to coarse pre-component).
- `post_*_mid`: projected mid component.
- `post_*_fine`: projected fine component.
- `post_projection_applied_*`: whether the corresponding projection step was applied (1) or
  skipped (0). Skips happen only for denominator safety (`denom <= eps`).
- `cos_t2_mid_coarse`, `cos_t1_mid_coarse`, and `cos_t2t1_mid_coarse`
  measure the raw mid/coarse alignment before lex projection, block-wise and
  on the combined `t2t1` view.
- `cos_t1_fine_higher` measures raw fine alignment against the current
  raw higher-priority `t1` direction: `coarse + mid`.
- `cos_t1_fine_coarse` and `cos_t1_fine_mid` measure raw fine alignment
  against each raw higher-priority component on `t1`.
- `post_cos_t2_mid_proj_coarse`, `post_cos_t1_mid_proj_coarse`, and
  `post_cos_t2t1_mid_proj_coarse` are expected to move near 0 where projection
  is applied.
- `post_cos_t1_fine_proj_higher` is expected to move near 0 where projection is
  applied.
- `post_cos_t1_fine_proj_coarse` and `post_cos_t1_fine_proj_mid_proj` are component-wise
  diagnostics only. They are not expected to be near 0 under the current
  single-vector fine projection.

## C) AMP Semantics

- Lex projection coefficients and diagnostics are computed on unscaled gradients, even when AMP is enabled.
- When AMP is enabled, the final projected gradients assigned to parameters are multiplied by the current
  `grad_scale` before `GradScaler.step`, so PyTorch can unscale and check them normally.
- Logged gradient norms (`grad_norm_*` and `post_grad_norm_*`) are reported in unscaled-equivalent units.
- Cosines/flags are scale-invariant.

## D) Recent Log Audit (Concrete, from `/scratch/g.saggini1/outputs`)

Audit date: **April 21, 2026**.
Scanned runs: `hcast*` with `run_log.jsonl` present.
### Observed key-set patterns

1. **49 keys** (full standard + full lex in older logs):
- Seen in: `hcast_lex_cifar100`, `hcast_lex_cub200`, `hcast_lex_aircraft`.
- These runs were produced before cosine/flag renaming and before dropping `lex_*_norm_*`
  raw/projected diagnostics and projection coefficients.
- Current runs add canonical `p...` keys to the compatibility keys, so their
  total depends on `train.gradient_blocks` and on which selected supports
  are non-empty; there is no longer one model-independent expected key count.

2. **25 keys** (full standard only):
- Seen in: `hcast_cifar100`, `hcast_cub200`, `hcast_aircraft`.
- Breakdown: `9 grad_norm + 6 cos + 5 param_norm + 5 delta_param_norm`.

3. **15 keys** (no union trunks logged):
- Seen in: `hcast_hcc_cifar100_step_0epoch`, `hcast_hcc_cub200_step_0epoch`, `hcast_hcc_cub200_step_0epoch_nokl`, `hcast_hcc_cub200_step_100epochs_cond_nokl`, `hcast_hcc_cub200_step_100epochs_cond2_nokl`.
- Breakdown: `6 grad_norm (t1/t2/t3 only) + 3 cos + 3 param_norm (t1/t2/t3) + 3 delta_param_norm (t1/t2/t3)`.

4. **7 keys** (t1-only logging):
- Seen in: `hcast_hcc_cub200_step_0epoch_cond_nokl`, `hcast_hcc_cub200_step_0epoch_cond2_nokl`.
- Keys:
  - `grad_norm_t1_coarse`, `grad_norm_t1_mid`, `grad_norm_t1_fine`
  - `cos_t1_mid_coarse`, `cos_t1_fine_mid`
  - `param_norm_t1`, `delta_param_norm_t1`
- These t1 cosine names are still supported by the current analysis utilities.

### Why counts differ across runs

Historical counts differ because metric emission was mask-dependent. Current
canonical counts additionally depend on the selected `p...` list and the
non-empty exact supports. The compatibility keys retain their historical
mask-dependent behavior.

## E) Mode Expectations

### Non-lex (`train.lexicographic.enabled=false`)

- Includes: canonical metrics for the selected blocks plus compatibility trunk metrics, including
  `cos_t2_mid_coarse`, `cos_t1_mid_coarse`, `cos_t2t1_mid_coarse`,
  `cos_t1_fine_higher`, `cos_t1_fine_coarse`, `cos_t1_fine_mid`.
- Excludes:
  - post-projection applied flags (`post_projection_applied_*`)
  - post-projection gradient norms (`post_grad_norm_*`)
  - post lexicographic cosine diagnostics:
    `post_cos_t2_mid_proj_coarse`, `post_cos_t1_mid_proj_coarse`,
    `post_cos_t2t1_mid_proj_coarse`, `post_cos_t1_fine_proj_higher`,
    `post_cos_t1_fine_proj_coarse`, `post_cos_t1_fine_proj_mid_proj`.

### Lex (`enabled=true`, `log_metrics=true`)

- Includes: canonical and compatibility standard metrics + post-projection applied flags (`post_projection_applied_*`)
  + post-grad norms (`post_grad_norm_*`) + post cosine diagnostics listed above.

### Lex (`enabled=true`, `log_metrics=false`)

- Includes: canonical selected-block and compatibility metrics (including raw pairwise cosines).
- Excludes: post-projection applied flags, post-grad norms, and post lexicographic cosine diagnostics.
