# Gradient and Parameter Diagnostic Logs (Current Code + Recent Run Audit)

This file documents the `train_metrics` keys written to `run_log.jsonl` by the **current** training code (`train/lexicographic/gradients.py`, `train/engine.py`).

## Scope

- Levels:
  - `coarse` = level 0
  - `mid` = level 1
  - `fine` = level 2
- Trunks:
  - `t1`: params that receive `coarse + mid + fine` grads
  - `t2`: params that receive `coarse + mid` grads only
  - `t3`: params that receive `coarse` grads only
- Union trunks:
  - `t2t1 = t2 ∪ t1`
  - `t3t2t1 = t3 ∪ t2 ∪ t1`

Important: mask-dependent keys are emitted only when the corresponding mask is active (`any(mask)` in code).
Pairwise cosine keys listed below are emitted for every epoch summary.
For Hier-COS, these diagnostics require a loss mode with differentiable per-level losses (`model.loss: per_level_kl_reg` or `per_level_ce`); plain `model.loss: kl_reg` does not expose per-level loss tensors. `model.weight_mode` is shared across Hier-COS KL target-path weighting and CE level-loss weighting; `per_level_kl_reg` still uses KL+reg per-level decomposition.

## A) Standard Trunk Metrics (non-lex and lex)

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

### Post-projection applied flags (`post_` prefix)

- `post_projection_applied_t2_mid_coarse`
- `post_projection_applied_t1_mid_coarse`
- `post_projection_applied_t1_fine_higher`

### Lex cosine diagnostics (post naming)

- `post_cos_t2_mid_proj_coarse`
- `post_cos_t1_mid_proj_coarse`
- `post_cos_t2t1_mid_proj_coarse`
- `post_cos_t1_fine_proj_higher`
- `post_cos_t1_fine_proj_coarse`
- `post_cos_t1_fine_proj_mid_proj`

### Post-lex gradient norms

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
  skipped for numerical safety (0).
- `cos_t2_mid_coarse`, `cos_t1_mid_coarse`, and `cos_t2t1_mid_coarse`
  measure the raw mid/coarse alignment before lex projection, block-wise and
  on the combined `t2t1` view.
- `cos_t1_fine_higher` measures raw fine alignment against the current
  raw higher-priority `t1` direction: `coarse + mid`.
- `cos_t1_fine_coarse` and `cos_t1_fine_mid` measure raw fine alignment
  against each raw higher-priority component on `t1`.
- `post_cos_t2_mid_proj_coarse`, `post_cos_t1_mid_proj_coarse`, and
  `post_cos_t2t1_mid_proj_coarse` are expected to move near 0 because mid is
  projected off coarse separately on `t2` and `t1`.
- `post_cos_t1_fine_proj_higher` is expected to move near 0 because fine is projected
  off the composed `coarse + mid_projected_t1` update on `t1`.
- `post_cos_t1_fine_proj_coarse` and `post_cos_t1_fine_proj_mid_proj` are component-wise
  diagnostics only. They are not expected to be near 0 under the current
  single-vector fine projection.

## C) AMP Semantics

- Lex projection coefficients and diagnostics are computed on unscaled gradients, even when AMP is enabled.
- When AMP is enabled, the final projected gradients assigned to parameters are multiplied by the current
  `grad_scale` before `GradScaler.step`, so PyTorch can unscale and check them normally.
- Logged gradient norms (`grad_norm_*`, `post_grad_norm_*`) are reported in unscaled-equivalent units.
- Cosines/flags are scale-invariant.

## D) Recent Log Audit (Concrete, from `/scratch/g.saggini1/outputs`)

Audit date: **April 21, 2026**.
Scanned runs: `hcast*` with `run_log.jsonl` present.

### Observed key-set patterns

1. **49 keys** (full standard + full lex in older logs):
- Seen in: `hcast_lex_cifar100`, `hcast_lex_cub200`, `hcast_lex_aircraft`.
- These runs were produced before cosine/flag renaming and before dropping `lex_*_norm_*`
  raw/projected diagnostics and projection coefficients.
- With the current code, full lex runs are expected to expose **43** keys:
  standard (25) + post-grad norms (9) + post cosines (6) + post-projection flags (3).

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

5. **0 keys** (no trunk diagnostics logged):
- Seen in: `hcast_inat21mini`.

### Why counts differ across runs

Counts differ because metric emission is mask-dependent. If a run has no active `t2`, `t3`, `t2t1`, or `t3t2t1` masks, those keys are not logged.

## E) Mode Expectations

### Non-lex (`train.lexicographic.enabled=false`)

- Includes: standard trunk metrics, including
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

- Includes: standard metrics + post-projection applied flags (`post_projection_applied_*`) + post-grad norms
  (`post_grad_norm_*`) + post cosine diagnostics listed above.

### Lex (`enabled=true`, `log_metrics=false`)

- Includes: standard trunk metrics (including the raw pairwise and trunk-specific cosine keys).
- Excludes: post-projection applied flags, post-grad norms, and post lexicographic cosine diagnostics.
