# Hier-COS Ablation Run Checklist

Status audited on 2026-08-10.

Unless a checklist item says otherwise, run three training seeds (`0`, `1`,
and `2`) with dataset split seed `0`. Keep the dataset's standard backbone,
pretraining, optimizer, augmentation, and alpha setting fixed. The default
transform for the core matrix is `full`.

## Completed anchors

- [x] Dense `orthonormal_random` frame, `global_softmax_ce_reg`, `kl_leaf`,
  no lex: CIFAR-100, CUB-200, and Aircraft, seeds 0--2.
- [x] Dense `orthonormal_random` frame, `global_softmax_ce_reg`, `kl_leaf`,
  `coarse_first` + `orthogonalize_all` lex: CIFAR-100, CUB-200, and Aircraft,
  seeds 0--2.
- [x] Identity frame, `level_softmax_ce_reg`, `kl_leaf`, LH projection with
  automatic feature width: CIFAR-100, CUB-200, and Aircraft, seeds 0--2.

## Minimum recommended campaign: Aircraft

Completing this section gives a compact, question-driven Aircraft ablation
covering the frame ladder, softmax scope, explicit lexicographic projection,
and LH projection.

- [ ] Validate and reuse the archived Aircraft run with identity frame,
  `global_softmax_ce_reg`, `kl_leaf`, full transform, and no lex.
  - It completed seeds 0--2 using all validation and test samples.
  - Its selected epochs match the current selection rule.
  - Its stored absolute checkpoint paths are stale after archival and must be
    resolved to the colocated checkpoints.
  - If archived artifacts are excluded from the thesis, rerun seeds 0--2
    instead.
- [ ] Run Aircraft with `orthonormal_block_random`,
  `global_softmax_ce_reg`, `kl_leaf`, full transform, and no lex, seeds 0--2.
- [ ] Run Aircraft with identity frame, `level_softmax_ce_reg`, `kl_leaf`,
  full transform, and `coarse_first` + `orthogonalize_all` lex, seeds 0--2.

New training required: **6 seed-runs**, or **9** if the archived identity
baseline is rerun.

## Optional: replicate the core on every dataset

Run this section only if the frame, softmax, and LH-versus-lex conclusions must
be supported independently on all three datasets.

- [ ] Run CIFAR-100 with identity frame, `global_softmax_ce_reg`, `kl_leaf`,
  full transform, and no lex, seeds 0--2.
- [ ] Run CUB-200 with identity frame, `global_softmax_ce_reg`, `kl_leaf`,
  full transform, and no lex, seeds 0--2.
- [ ] Run CIFAR-100 with `orthonormal_block_random`,
  `global_softmax_ce_reg`, `kl_leaf`, full transform, and no lex, seeds 0--2.
- [ ] Run CUB-200 with `orthonormal_block_random`,
  `global_softmax_ce_reg`, `kl_leaf`, full transform, and no lex, seeds 0--2.
- [ ] Complete the CIFAR-100 identity-frame, `level_softmax_ce_reg`,
  `kl_leaf`, full-transform, no-lex baseline:
  - [ ] Resume seed 0 from epoch 4.
  - [ ] Start seed 1.
  - [ ] Start seed 2.
- [ ] Complete the CUB-200 identity-frame, `level_softmax_ce_reg`, `kl_leaf`,
  full-transform, no-lex baseline:
  - [ ] Resume seed 2 from epoch 23.
- [ ] Run CIFAR-100 with identity frame, `level_softmax_ce_reg`, `kl_leaf`,
  full transform, and `coarse_first` + `orthogonalize_all` lex, seeds 0--2.
- [ ] Run CUB-200 with identity frame, `level_softmax_ce_reg`, `kl_leaf`,
  full transform, and `coarse_first` + `orthogonalize_all` lex, seeds 0--2.

Additional training required after the minimum Aircraft campaign:
**22 seed-runs**.

## Conditional transform study

Do not complete the transform ladder until a matched lex effect has first been
confirmed on Aircraft.

- [ ] Run Aircraft with identity frame, `level_softmax_ce_reg`, `equal`
  weighting, full transform, and no lex, seeds 0--2.
- [ ] Compare it with the existing matching full-transform lex runs.
- [ ] If the matched lex effect is meaningful, run the remaining paired ladder:
  - [ ] No lex with `bn_linear`, seeds 0--2.
  - [ ] No lex with `final_only`, seeds 0--2.
  - [ ] `coarse_first` + `orthogonalize_all` lex with `final_only`, seeds 0--2.

The matching Aircraft lex runs for `full` and `bn_linear` already exist. The
conditional study therefore requires **3 seed-runs initially**, followed by
**9 more** only if the first comparison justifies continuing.

## Optional mechanistic controls

Run these only if the thesis will make the corresponding stronger mechanistic
claims. Keep the dense frame, global softmax, `kl_leaf`, full transform, and
Aircraft dataset fixed unless the item explicitly changes one of them.

- [ ] Run `fine_first` + `orthogonalize_all` lex, seeds 0--2, to test whether
  the coarse-to-fine ordering itself matters.
- [ ] Run `coarse_first` + `conflict_only` lex, seeds 0--2, to test whether
  conditional conflict removal matches unconditional projection.
- [ ] Run a no-lex `kl_coarse` baseline, seeds 0--2, as the inexpensive
  reweighting falsification control.

Do not add the `fine_first` + `conflict_only` interaction unless that
interaction becomes an explicit research question.

## Do not schedule as a factorial

- [ ] Do **not** cross every frame with every loss, weight, transform,
  mechanism, backbone, and dataset.
- [ ] Do **not** run both `orthonormal_block_random` and
  `orthonormal_random` + `fixed_frame_per_level=true`; they represent the same
  conceptual block-frame condition.
- [ ] Do **not** treat identity with and without per-level construction as two
  separate frame conditions.
- [ ] Do **not** build a complete parallel `kl_reg` grid. With fixed targets,
  native KL and global CE have the same training gradient and differ by a
  target-entropy constant.
- [ ] Do **not** run every weight under every frame, loss, and transform.
- [ ] Do **not** run the transform ladder on every dataset.
- [ ] Do **not** stack LH projection, HCC, and lex in the main attribution
  matrix.
- [ ] Treat cumulative/marginal branching weights, LH advantage, width 512,
  backbone/pretraining, HCC, and direct subspace supervision as separate,
  optional research questions.

## Launcher prerequisites

Resolve these before scheduling block-frame or transform runs:

- [ ] Add `orthonormal_block_random` support to the Hier-COS baseline runner;
  it currently rejects that frame mode.
- [ ] Give block-frame transform runs an explicit output-directory suffix;
  the current transform runner can collide with dense-frame output names.
- [ ] Verify every generated output directory with `DRY_RUN=1` before training.
- [ ] For resumed seeds, pass the exact local `latest.pt` explicitly and avoid
  relaunching already completed seeds.

## Reporting checks

- [ ] Require `test_metrics.yaml` and a final `test` event before marking a
  seed complete.
- [ ] Report the actual seed count for every result.
- [ ] Use the top-down-selected checkpoint for top-down rows and the
  independent-selected checkpoint for independent rows.
- [ ] Report FPA, weighted AP, and accuracy as higher-is-better; report AHD and
  TICE as lower-is-better.
- [ ] Report percentage differences in percentage points.
- [ ] Confirm lex activity from the logged projection diagnostics, not only
  from the directory name.
- [ ] Treat partial logs as training-dynamics/debug artifacts, not final
  results.
- [ ] Do not use archived CIFAR-100 or CUB-200 runs with
  `drop_last_eval=true` for headline thesis comparisons.
