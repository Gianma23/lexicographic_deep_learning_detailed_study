# Hier-COS Ablation Run Checklist

Run status audited on 2026-08-10. Scheduling policy updated on 2026-08-14.

Every scheduled configuration is a three-dataset bundle: CIFAR-100, CUB-200,
and Aircraft are always launched together. Do not schedule a new condition for
only one or two datasets. Within each dataset, run three training seeds (`0`,
`1`, and `2`) with dataset split seed `0`, so a new configuration normally
means **9 seed-runs**. Existing completed dataset-seeds may be reused, but the
bundle is complete only when all nine dataset-seeds are complete. Keep each
dataset's standard backbone, pretraining, optimizer, augmentation, and alpha
setting fixed. The default transform for the core matrix is `full`.

## Completed anchors

- [x] Dense `orthonormal_random` frame, `global_softmax_ce_reg`, `kl_leaf`,
  no lex: CIFAR-100, CUB-200, and Aircraft, seeds 0--2.
- [x] Dense `orthonormal_random` frame, `global_softmax_ce_reg`, `kl_leaf`,
  `coarse_first` lex: CIFAR-100, CUB-200, and Aircraft, seeds 0--2.
- [x] Identity frame, `level_softmax_ce_reg`, `kl_leaf`, LH projection with
  automatic feature width: CIFAR-100, CUB-200, and Aircraft, seeds 0--2.

## Minimum recommended three-dataset campaign

Completing this section gives a compact, question-driven ablation on all three
datasets covering the frame ladder, softmax scope, explicit lexicographic
projection, and LH projection.

- [ ] Complete the identity-frame, `global_softmax_ce_reg`, `kl_leaf`,
  full-transform, no-lex bundle:
  - [ ] Run CIFAR-100, seeds 0--2.
  - [ ] Run CUB-200, seeds 0--2.
  - [ ] Validate and reuse the archived Aircraft seeds 0--2.
    - They used all validation and test samples.
    - Their selected epochs match the current selection rule.
    - Their stored absolute checkpoint paths are stale after archival and must
      be resolved to the colocated checkpoints.
    - If archived artifacts are excluded from the thesis, rerun all three
      Aircraft seeds instead.
- [ ] Run the `orthonormal_block_random`, `global_softmax_ce_reg`, `kl_leaf`,
  full-transform, no-lex bundle on all three datasets, seeds 0--2 each.
- [ ] Complete the identity-frame, `level_softmax_ce_reg`, `kl_leaf`,
  full-transform, no-lex bundle:
  - [x] Aircraft, seeds 0--2.
  - [ ] CIFAR-100: resume seed 0 from epoch 4; start seeds 1 and 2.
  - [ ] CUB-200: reuse completed seeds 0 and 1; resume seed 2 from epoch 23.
- [ ] Run the identity-frame, `level_softmax_ce_reg`, `kl_leaf`, full-transform,
  `coarse_first` lex bundle on all three datasets, seeds 0--2 each.

New training required: **28 seed-runs**, or **31** if the archived Aircraft
identity baseline is rerun.

## Conditional transform study

Do not complete the transform ladder until a matched lex effect has first been
evaluated on the full three-dataset bundle.

- [ ] Run identity frame, `level_softmax_ce_reg`, `equal` weighting, full
  transform, and no lex on all three datasets, seeds 0--2 each.
- [ ] Compare this bundle with the existing matching full-transform lex bundle
  for all three datasets.
- [ ] If the matched lex effect is meaningful, complete the remaining paired
  ladder on all three datasets:
  - [ ] No lex with `bn_linear`, seeds 0--2 each.
  - [ ] Complete `coarse_first` lex with `bn_linear`:
    reuse the completed Aircraft and CUB-200 seeds, and run CIFAR-100 seeds
    0--2.
  - [ ] No lex with `final_only`, seeds 0--2 each.
  - [ ] `coarse_first` lex with `final_only`, seeds 0--2 each.

The matching full-transform lex bundle already exists. The `bn_linear` lex
condition is complete for Aircraft and CUB-200 but not CIFAR-100. The
conditional study therefore requires **9 seed-runs initially**, followed by
**30 more** only if the first comparison justifies continuing.

## Optional mechanistic controls

Run these only if the thesis will make the corresponding stronger mechanistic
claims. Keep the dense frame, global softmax, `kl_leaf`, and full transform
fixed unless the item explicitly changes one of them. Run every selected
condition on all three datasets.

- [ ] Run `fine_first` lex, seeds 0--2 per dataset, to
  test whether the coarse-to-fine ordering itself matters.
- [ ] Run a no-lex `kl_coarse` baseline, seeds 0--2 per dataset, as the
  inexpensive reweighting falsification control.

## Do not schedule as a factorial

- [ ] Do **not** cross every frame with every loss, weight, transform,
  mechanism, and backbone.
- [ ] Do **not** run both `orthonormal_block_random` and
  `orthonormal_random` + `fixed_frame_per_level=true`; they represent the same
  conceptual block-frame condition.
- [ ] Do **not** treat identity with and without per-level construction as two
  separate frame conditions.
- [ ] Do **not** build a complete parallel `kl_reg` grid. With fixed targets,
  native KL and global CE have the same training gradient and differ by a
  target-entropy constant.
- [ ] Do **not** run every weight under every frame, loss, and transform.
- [ ] Do **not** split a selected configuration into dataset-specific
  campaigns; launch CIFAR-100, CUB-200, and Aircraft together.
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
- [ ] When a launcher exposes `DATASETS`, set it to
  `"cifar100 cub200 aircraft"`; do not rely on a subset default.
- [ ] Verify every generated output directory for all three datasets with
  `DRY_RUN=1` before training.
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
