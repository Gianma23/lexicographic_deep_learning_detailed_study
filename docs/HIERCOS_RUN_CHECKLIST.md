# Hier-COS Ablation Run Checklist

Run status re-audited on 2026-08-25. The full structural and mechanism campaign
uses CIFAR-100, CUB-200, and Aircraft for every required condition.

Use three training seeds (`0`, `1`, and `2`) and dataset split seed `0`. Keep
`kl_leaf`, the `full` transform, and each dataset's standard backbone,
pretraining, optimiser, augmentation, alpha, and input resolution fixed unless
the research question explicitly varies one of them.

## Contrast map: this is not a factorial

The required settings form a connected sequence. They do **not** require every
frame, softmax scope, mechanism, weight, transform, and backbone combination.

| Question | Matched conditions |
|---|---|
| Frame ladder | dense/global/none; block/global/none; identity/global/none |
| Softmax adaptation cost | identity/global/none; identity/level/none |
| Main explicit-lex effect | dense/global/none; dense/global/coarse-first lex |
| LH versus explicit lex | identity/level/none; identity/level/coarse-first lex; identity/level/LH |

Here `dense` means `fixed_frame_mode=orthonormal_random` with
`fixed_frame_per_level=false`; `block` means the canonical
`fixed_frame_mode=orthonormal_random` with `fixed_frame_per_level=true`;
`identity` means the identity frame; `global` means
`global_softmax_ce_reg`; and `level` means `level_softmax_ce_reg`. The legacy
`orthonormal_block_random` mode is only an alias for the canonical block pair
and should not be used for new runs.

All required frame, softmax, and mechanism comparisons cover CIFAR-100,
CUB-200, and Aircraft. Only separately labelled optional diagnostics, such as a
width or backbone sensitivity study, may use a prespecified subset.

## Completed anchors

- [x] Dense/global/`kl_leaf`/none: CIFAR-100, CUB-200, and Aircraft, seeds
  0--2.
- [x] Dense/global/`kl_leaf`/`coarse_first` lex: all three datasets, seeds
  0--2.
- [x] Identity/level/`kl_leaf`/LH with automatic taxonomy width: all three
  datasets, seeds 0--2.
- [x] Dense/global/`kl_leaf`/`fine_first` lex: all three datasets, seeds 0--2.
  The configs select `fine_first`, and the logs contain active post-projection
  diagnostics. Do not schedule these again.
- [x] Dense/global/`kl_leaf`/HCC: all three datasets, seeds 0--2.
  `proj_constraint_alpha=1` is present in the logs. Do not schedule these
  again.
- [x] Identity/level/`equal`/full-transform/`coarse_first` lex: all three
  datasets, seeds 0--2. This is the transform-study lex reference, not the
  `kl_leaf` row required by the LH-versus-lex comparison.
- [x] Identity/level/`equal`/`bn_linear`/`coarse_first` lex: Aircraft and
  CUB-200, seeds 0--2.

## Interpretation of the current LH comparison

Enabling the current LH path simultaneously creates learnable per-level heads,
adds a terminal PReLU, and applies the branch-point backward projection. The
vanilla identity/level baseline and explicit-lex arm do not use those heads or
that PReLU.

Therefore the currently runnable three-arm comparison identifies the effect of
the complete **LH adaptation package**, not the projection operator in
isolation. It is acceptable for a complete-method comparison if this limitation
is stated explicitly.

For a stronger projection-only claim, first implement an adapter-only mode with
the same heads and PReLU but with the LH backward projection bypassed. The
strictly matched study would then require adapter-only none, adapter-only
explicit lex, and adapter + LH. Do not build extra frame/softmax factorial cells
as a substitute for this missing control.

`model.projection.feature_dim` is ignored when `model.projection.enabled=false`:
a no-projection baseline always uses the taxonomy-node width. Existing baseline
directories containing `_d512` therefore have a misleading name but are valid
automatic-width baselines. They must not be reported as width-512 controls.

## Required full three-dataset campaign

This is the selected campaign. It completes the frame ladder, softmax
adaptation cost, and matched identity/level mechanism comparison independently
on CIFAR-100, CUB-200, and Aircraft. It requires **28 remaining seed-runs**.

The sections are ordered by interpretation: establish the frame first, then
measure the cost of changing the softmax, and only then compare mechanisms on
the adapted identity/level substrate. The training jobs themselves are
independent and may run in parallel.

### A. Frame ablation: dense versus block versus identity

- [x] Reuse the restored Aircraft identity/global/`kl_leaf`/none seeds 0--2
  under
  `hiercos_aircraft_global_softmax_ce_reg_baseline_kl_leaf_identity`. They
  retain every validation/test sample, match the current selection rule, and
  their stored checkpoint paths resolve after restoration. No retraining is
  needed.
- [x] Run CIFAR-100 identity/global/`kl_leaf`/none, seeds 0--2. Do not resume
  the archived run; all three archived seeds used `drop_last_eval=true`.
- [x] Run CUB-200 identity/global/`kl_leaf`/none, seeds 0--2. Do not resume the
  archived run; all three archived seeds used `drop_last_eval=true`.
- [x] Run Aircraft block/global/`kl_leaf`/none, seeds 0--2.
- [x] Run CIFAR-100 block/global/`kl_leaf`/none, seeds 0--2.
- [x] Run CUB-200 block/global/`kl_leaf`/none, seeds 0--2.

Compare the completed dense/global baseline with the new block/global and
identity/global baselines. Loss, weighting, transform, mechanism, and backbone
remain fixed, so only the frame changes.

Outstanding cost for section A: **15 seed-runs**.

### B. Softmax ablation: global versus per-level

- [ ] Complete identity/level/`kl_leaf`/full-transform/none:
  - [x] Aircraft seeds 0--2: reuse
    `hiercos_aircraft_level_softmax_ce_reg_baseline_kl_leaf_identity`.
    These runs used automatic taxonomy width.
  - [ ] CIFAR-100 seed 0: resume the current local `latest.pt` after epoch 4.
  - [ ] CIFAR-100 seeds 1 and 2: start fresh.
  - [x] CUB-200 seeds 0 and 1: reuse the completed current runs.
  - [ ] CUB-200 seed 2: resume the current local `latest.pt` after epoch 23.

Compare each identity/global baseline from section A with the corresponding
identity/level baseline here. Frame, weighting, transform, mechanism, and
backbone remain fixed, so only the softmax scope changes.

Outstanding cost for section B: **4 seed-runs**.

### C. Mechanism comparison on the adapted identity/level substrate

- [ ] Run identity/level/`kl_leaf`/full-transform/`coarse_first` lex on all
  three datasets, seeds 0--2. The completed directories without `_kl_leaf` use
  `equal` weighting and do **not** fill this row.

Compare the identity/level none baseline from section B, the new explicit-lex
row, and the completed identity/level LH row. This is the complete-method
comparison subject to the LH head/PReLU limitation stated above.

Outstanding cost for section C: **9 seed-runs**.

## Conditional transform study

Do not complete the transform ladder until the matched full-transform lex
effect has been evaluated.

- [ ] Run identity/level/`equal`/full-transform/none on all three datasets,
  seeds 0--2.
- [ ] Compare it with the completed matching identity/level/`equal`/full lex
  rows.
- [ ] If the paired lex effect is meaningful, complete the paired ladder on the
  same datasets:
  - [ ] `bn_linear`/none.
  - [ ] Complete `bn_linear`/lex. Aircraft and CUB-200 seeds 0--2 are complete;
    the CIFAR-100 directory contains only a seed-0 config and no checkpoint, so
    CIFAR-100 seeds 0--2 must start fresh if selected.
  - [ ] `final_only`/none.
  - [ ] `final_only`/lex.

For all three datasets this requires **9 seed-runs initially**, followed by
**30 more** only if the first comparison justifies continuing. Archived
`bn_linear` and `final_only` runs with `drop_last_eval=true` do not count and
must not be resumed for headline results.

## Optional mechanistic controls

- [x] `fine_first` lex ordering control: complete on all three datasets.
- [ ] No-lex `kl_coarse` baseline: run only if the thesis will test whether
  simple coarse-heavy scalar weighting reproduces the consistency effect of
  projection. Keep dense/global/full fixed and use all three datasets so it
  matches the associated lex comparison.
- [ ] Adapter-only head control: recommended before claiming that an observed
  LH difference is caused specifically by its backward projection.
- [ ] LH advantage parameterisation: separate model-variant question; do not
  fold it into the main LH-versus-lex contrast.
- [ ] LH width-512 diagnostic: separate representation-width question. The
  current CUB-200 width-512 directory has seed 0 complete and seed 1 stopped
  after epoch 8. Leave it on hold. If this CUB-only diagnostic is selected,
  resume seed 1 and start seed 2; do not rerun seed 0.

## Optional CIFAR-100 backbone/pretraining study

Do not include this in the core campaign. The current script changes the
standard CIFAR WRN-28-8/32-pixel setting to ResNet-50/224 pixels, so a comparison
against the standard CIFAR runs is an **architecture, pretraining, and input
pipeline sensitivity study**, not a clean backbone-only ablation.

Choose exactly one question before resuming any partial ResNet run:

1. **Pretraining-by-frame baseline study.** Use global softmax, `kl_leaf`, full
   transform, and no lex. Run
   `{pretrained, scratch} x {dense random, identity}`. Neither lex nor LH belongs
   in this matrix.
2. **Explicit-lex robustness.** Choose one ResNet initialisation and compare
   none versus explicit lex at dense/global. Do not add identity or LH.
3. **LH-versus-lex robustness.** Choose one ResNet initialisation and compare
   none, explicit lex, and LH at identity/level. Use identity only; a dense
   global frame is incompatible with LH.

Default decision: **hold all ResNet partial runs** until one of these questions
is selected. If question 1 is selected, the outstanding work is:

- [ ] Pretrained dense: resume seed 0 after epoch 34; start seeds 1 and 2.
- [x] Pretrained identity seed 0: complete.
- [ ] Pretrained identity seed 1: resume from its local checkpoint at epoch 4.
- [ ] Pretrained identity seed 2: start fresh.
- [x] Scratch identity seed 0: complete.
- [ ] Scratch identity seeds 1 and 2: start fresh.
- [ ] Scratch dense seeds 0--2: start fresh.

The completed Aircraft from-scratch identity bundle is a negative pilot: the
small training set collapsed near chance. Do not extend it into a larger
Aircraft pretraining matrix or use it as a headline backbone result.

## Stale and archived run disposition

### Reuse

- [x] Aircraft identity/global/`kl_leaf`/none, seeds 0--2: restored to its
  original output path; all stored checkpoint references resolve.
- [x] Current Aircraft identity/level/`kl_leaf`/none directory: renamed without
  the misleading `_d512` suffix; it is an automatic-width baseline.
- [x] Current CUB-200 identity/level/`kl_leaf`/none seeds 0 and 1.
- [x] Current dense/global `fine_first` and HCC bundles: complete and active.

### Resume now

- [ ] Current CIFAR-100 identity/level/`kl_leaf`/none seed 0 from its exact
  local checkpoint after epoch 4:
  `/scratch/g.saggini1/outputs/hiercos_cifar100_level_softmax_ce_reg_baseline_kl_leaf_d512_identity/seed_0/latest.pt`.
- [ ] Current CUB-200 identity/level/`kl_leaf`/none seed 2 from its exact local
  checkpoint after epoch 23:
  `/scratch/g.saggini1/outputs/hiercos_cub200_level_softmax_ce_reg_baseline_kl_leaf_d512_identity/seed_2/latest.pt`.

### Rerun cleanly now

- [ ] Archived CIFAR-100 identity/global/`kl_leaf`/none seeds 0--2.
- [ ] Archived CUB-200 identity/global/`kl_leaf`/none seeds 0--2.

Their old checkpoints were selected with truncated evaluation batches. Do not
resume them into the new namespace or combine old and new seeds in one mean.

### Hold unless the optional question is selected

- [ ] CUB-200 LH width-512 seed 1 at epoch 8 and missing seed 2.
- [ ] All partial CIFAR-100 ResNet-50 runs listed above.

### Retire from headline comparisons

- [x] Archived CIFAR-100 and CUB-200 runs with `drop_last_eval=true`.
- [x] Archived one-seed `bn_linear`, `final_only`, old `fine_first`, and old
  `kl_leaf` transform variants with truncated evaluation.
- [x] The failed hand-launched CIFAR ResNet directory with no checkpoint.
- [x] Superseded legacy conflict-gated lex runs.
- [x] The Aircraft from-scratch collapse: retain only as a documented pilot.

Do not delete retired artifacts; leave them under `recovery/` or their current
directory and exclude them in analysis code.

## Launcher prerequisites

Resolve these before launching new frame or transform conditions:

- [ ] Expose and validate `FIXED_FRAME_PER_LEVEL` in
  `run_hiercos_baselines.sh`; launch the block arm with
  `FIXED_FRAME_MODE=orthonormal_random` and
  `FIXED_FRAME_PER_LEVEL=true`.
- [ ] Give block-frame baseline and transform outputs an explicit `_block`
  suffix so they cannot collide with dense-frame output names.
- [ ] Make the baseline runner actually parse `DATASETS`; its comment says the
  variable is overridable, but the current script hard-codes all three
  datasets.
- [ ] Do not use `FEATURE_DIM` to name no-projection baseline runs; it does not
  change their model width.
- [ ] Verify every generated output directory with `DRY_RUN=1` before training.
- [ ] For partial seeds, pass the exact local `latest.pt` explicitly. Do not
  relaunch completed seeds or rely on an automatic resume guess.
- [ ] In `run_hiercos_cifar100_resnet50_pretrained.sh`, reconcile the prose
  saying the default is pretrained with the actual
  `PRETRAINED_MODE=false` default before using the launcher.

## Do not schedule as a factorial

- [ ] Do **not** cross every frame with every loss, weight, transform,
  mechanism, backbone, and dataset.
- [ ] Do **not** run the legacy `orthonormal_block_random` alias in addition to
  `orthonormal_random` + `fixed_frame_per_level=true`; they resolve to the same
  block-frame condition.
- [ ] Do **not** treat identity with and without per-level construction as
  separate frame conditions.
- [ ] Do **not** build a parallel `kl_reg` grid. With fixed targets, native KL
  and global CE have the same training gradient and differ by a
  target-entropy constant.
- [ ] Do **not** run every weight under every frame, loss, and transform.
- [ ] Do **not** stack LH projection, HCC, and explicit lex in the main
  attribution matrix.
- [ ] Treat branching weights, LH advantage, width 512,
  backbone/pretraining, HCC, and direct subspace supervision as separate
  research questions.

## Reporting checks

- [ ] Require `test_metrics.yaml` and a final `test` event before marking a
  seed complete.
- [ ] Report the actual seed count for every result.
- [ ] Use the top-down-selected checkpoint for top-down rows and the
  independent-selected checkpoint for independent rows.
- [ ] Report FPA, weighted AP, and accuracy as higher-is-better; report AHD and
  TICE as lower-is-better.
- [ ] Report percentage differences in percentage points.
- [ ] Confirm lex activity from post-projection diagnostics and HCC activity
  from `proj_constraint_alpha`, not only from directory names.
- [ ] Treat partial logs as training-dynamics/debug artifacts, not final
  results.
- [ ] Never mix archived `drop_last_eval=true` results with clean current runs
  in a headline aggregate.
