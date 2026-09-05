# Hier-COS Ablation Run Checklist

Audited 2026-09-03 from `config_resolved.yaml`, `test_metrics.yaml`, and final
`test` events in `run_log.jsonl` under `/scratch/g.saggini1/outputs`, not from
directory names.

LH execution order amended 2026-09-03: the `kl_leaf` LH base is followed
immediately by the `feature_dim` ablation, and every later LH ablation uses
`feature_dim=512`.

3 seeds (`0`,`1`,`2`) + split seed `0` for every thesis-table row, including
the capacity sensitivity. Everything not named by the question stays fixed.

## Run queue

| P | Section | What | Seed-runs | Blocked by |
|---|---|---|---|---|
| 0 | C | `equal` x `{none, lex}` on all datasets | 9 | baseline dataset matrix |
| 3 | C | cumulative beta=1 x `{none, lex}` on all datasets | 18 | weight-mode/beta launcher support |
| 3 | C | Aircraft marginal x `{none, lex}` | 6 | weight-mode launcher support |
| 4 | G | CIFAR-100 WRN ladder on LH-projection, 3 seeds | 6 | — |

Sections A, B, D, E and F are complete. Section C can proceed independently of
the LH lane. The D-to-E sequencing gate has now been satisfied on every dataset;
all later LH weighting, advantage, capacity or other mechanism ablations must
still use the `d512` row rather than the automatic-width row.

The three LH phases differ as follows:

| Phase | Weighting | `feature_dim` | Purpose | Output-name marker |
|---|---|---:|---|---|
| D: pre-width base | `kl_leaf` | `0` (taxonomy width) | establish the pre-correction LH representation | no `_d...` suffix |
| E: width reference | `kl_leaf` | `512` | isolate and quantify the representation-width correction | `_d512` |
| F: later LH ablations | named axis | `512` fixed | vary weights or another LH-specific axis | `_d512` |

The launcher defaults to `FEATURE_DIM=512`. Therefore phase D must pass
`FEATURE_DIM=0` explicitly; an unqualified launcher call belongs to phase E or
F, never to the pre-width base.

Section G belongs to the Hier-COS ablation section of the thesis. It is a
non-blocking sensitivity of the corrected LH configuration and reuses E's
CIFAR-100 `d512` row as its WRN-28-8 anchor.

## Vocabulary

`dense` = `orthonormal_random` + `fixed_frame_per_level=false`; `block` = same +
`per_level=true`; `identity` = identity frame; `global` = `global_softmax_ce_reg`;
`level` = `level_softmax_ce_reg`. Do not use the legacy
`orthonormal_block_random` alias.

## Substrate rule

> Structural substrate = `level` + that dataset's best **LH-compatible** frame
> from section A. Weighting is the explicit axis in section C and is held
> matched between `none` and `coarse_first` within every cell.

LH-compatible means identity or per-level block; `models/hiercos/model.py:256`
rejects a dense frame under `projection.enabled=true`. Earlier versions of this
document assumed a single identity substrate for all three datasets, which
section A falsifies — that assumption is what made everything after C unusable.

| Dataset | Structural substrate | Section A evidence (FPA_ind / TICE_ind) |
|---|---|---|
| CIFAR-100 | **block**/level | block .7731/.0094 > identity .7687/.0097; dense .7752 is LH-incompatible |
| Aircraft | **identity**/level | identity .8274/.0035 > dense .8140/.0078 > block .8103/.0089 |
| CUB-200 | **identity**/level | identity .7656/.0021 > dense .7615/.0060 > block .7607/.0066 |

Hold the frame fixed **within** a dataset. Its variation **across** datasets is a
section A result; carry the substrate column on every mechanism table.

---

## A. Frame ladder — COMPLETE (0 outstanding)

Isolates: frame, at global/`kl_leaf`/none.

- [x] Dense/global/none: CIFAR-100, CUB-200, Aircraft, seeds 0--2.
- [x] Block/global/none: all three, seeds 0--2.
- [x] Identity/global/none: all three, seeds 0--2.

Result: best frame is dataset-dependent. Identity buys 1.3 pp FPA and halves
TICE on Aircraft, buys 0.4 pp on CUB-200, costs 0.7 pp on CIFAR-100. The earlier
"15 seed-runs outstanding" was stale.

## B. Softmax scope — COMPLETE (0 outstanding)

Isolates: global vs per-level softmax, at each dataset's substrate frame.

- [x] CIFAR-100 at block: `..._global_..._baseline_kl_leaf_block` vs
  `..._level_..._baseline_kl_leaf_block`, seeds 0--2 each.
- [x] Aircraft at identity: same pair, seeds 0--2 each.
- [x] CUB-200 at identity: same pair, seeds 0--2 each.

Result: per-level softmax is free. CIFAR-100 .7731 -> .7780 FPA and .0094 ->
.0087 TICE; Aircraft .8274 -> .8248 and .0035 -> .0032; CUB-200 .7656 -> .7649
and .0021 -> .0023. This establishes the structural substrate for section C.

## C. Level-weight ablation — 33 outstanding

Isolates: the numerical level weights, crossed with **both** `{none,
lex_coarse_first}` at the structural substrate on all three datasets. A
lex-only sweep cannot distinguish a generally better scalarisation from a
weight--lex interaction. Its result also cannot be transplanted to LH; the
separate LH weight contrast is phase F and uses `feature_dim=512` throughout.

### Why LH needs its own weight contrast

Earlier versions fixed the weighting on `{none, lex}` and carried the winning
rule into D. That assumes lex and LH respond to the level weights the same way.
They do not, and the asymmetry is structural:

- **lex** projects level *l*'s gradient off the summed higher-priority
  *gradients*. `train/lexicographic/gradients.py:364` divides by `<r,r>`, so the
  projection is scale-invariant in its reference and is rebuilt from scratch
  every step. Rescaling a level cannot change the projection at that step.
- **LH** projects level *l*'s input feature off `span{W_0..W_(l-1)}`, the
  previous heads' *weights* (`models/hiercos/model.py:409-428`, detached). That
  reference is an **accumulated** object: the weights change how `W_0` trains,
  hence the protected subspace at every later step.

Weights therefore have no memory in lex and full memory in LH, so a rule chosen
on lex is not evidence about LH. The LH contrast is still required, but it is
run only after the width correction: phase E supplies the three CIFAR-100
`kl_leaf`/`d512` cells, and phase F adds the three `equal`/`d512` cells. Those
six cells settle the weight--LH question without mixing representation widths.

Indicative support, measured 2026-08-28 — per-level trunk gradient norms
coarse:mid:fine at epoch 30 under `kl_leaf`: lex 1:1.59:6.50 (coarse 2.2% of
step energy) against LH 1:1.11:2.80 (10.0%). LH is already better balanced at
the same weights. The two rows sit on different frames (lex block, LH identity),
so this is indicative only — the six new cells remove that confound.

The selected grid is deliberately small:

| Rule | CIFAR-100 | Aircraft | CUB-200 | Coverage |
|---|---|---|---|---|
| `equal` | .333/.333/.333 | same | same | all datasets |
| `kl_leaf` | .162/.225/.613 | same | same | all datasets |
| `cumulative_branching`, beta=1 | .062/.156/.781 | .150/.350/.500 | .052/.151/.797 | all datasets |
| `marginal_branching` | — | .210/.490/.300 | — | Aircraft only |

The cumulative beta=0.5 and marginal vectors are close to `kl_leaf` on
CIFAR-100 and CUB-200, so they are not scheduled there. Aircraft retains the
marginal rule because it is the only selected mid-heavy vector. `kl_coarse` and
cumulative beta=1.5 are outside this ablation.

### Existing diagnosis motivating the grid

Diagnosed 2026-08-27 from `run_log.jsonl`, block substrate, lex vs matched
baseline. Unweighted train CE (`ce_level_l / w_l`) at epoch 50:

| | L0 | L1 | L2 |
|---|---|---|---|
| baseline | 0.124 | 0.196 | 0.261 |
| lex `kl_leaf` | **0.859** | **1.072** | 0.252 |
| lex `equal` | 0.228 | 0.205 | 0.305 |

- Fine level fit identically; the two levels holding lexicographic *priority* are
  7x and 5.5x underfit.
- Baseline `cos_p123_fine_coarse` is .36--.42 over epochs 5--50 with
  `|g_fine|/|g_coarse| ~ 5`, so the coarse-aligned part of the fine gradient is
  ~2x the coarse gradient itself. `coarse_first` deletes exactly that component.
- `kl_leaf` = `[.162,.225,.613]`, folded inside `ce_level_losses`
  (`models/hiercos/losses.py:389`), so it survives into `level_losses` and sets
  step magnitude while the projection direction is scale-invariant. Lex gives the
  coarse level priority in direction and a 0.162x step. `equal` gives 0.333x and
  the regression disappears.
- Aircraft/CUB cannot show this: pretrained ResNet-50, coarse train acc 1.0 by
  epoch ~10, coarse grad norm 2.6 -> 0.32. Lex inflates their coarse CE the same
  way (Aircraft .099 vs .050 at ep50) but from a negligible base.
- TICE worsens because lex orders update *directions* only; it cannot make an
  underfit coarse head agree with the fine head.
- Secondary: `level_losses[l] = w_l*CE_l + alpha*R_l` leaves the regulariser
  unweighted, so the priority objective is 31% regulariser at L0 vs 8% at L2.

Consequence: the level weights become relative per-level gradient scales in lex
mode, not a nuisance parameter. Section C is therefore required before the
final matched-weight interpretation, but it does not delay the D-to-E width
sequence.

### Run matrix

All cells use seeds 0--2 and split seed 0.

**CIFAR-100 at block/level** — run in the order given:

- [x] **(1)** `equal` x none: **3 seed-runs**. Blocking: without it the `equal`
  column has no baseline, so the 2x2 cannot separate a generally better
  scalarisation from a weight--lex interaction. Everything else in this section
  rests on a half-filled square until it lands.
- [x] **(1)** `equal` x lex: seeds 0--2 complete.
- [x] `kl_leaf` x `{none, lex}`: seeds 0--2 complete. This is the failing cell:
  unweighted train CE 0.859/1.072/0.252 at epoch 50 against the baseline's
  0.124/0.196/0.261, FPA .7659 vs .7780+-.0018, TICE .0137 vs .0087. It is the
  predicted starvation result on the correct substrate, not a frame artifact.
- [ ] **(3)** cumulative beta=1 x `{none, lex}`: **6 seed-runs**. Refines a
  curve whose shape (1)--(2) already give; lowest priority in this section, and
  below the LH cells.

**Aircraft at identity/level:**

- [x] `equal` x none: **3 seed-runs**.
- [x] `equal` x lex: seeds 0--2 complete.
- [x] `kl_leaf` x `{none, lex}`: seeds 0--2 complete.
- [ ] cumulative beta=1 x `{none, lex}`: **6 seed-runs**.
- [ ] marginal x `{none, lex}`: **6 seed-runs**.

**CUB-200 at identity/level:**

- [x] `equal` x none: **3 seed-runs**.
- [x] `equal` x lex: seeds 0--2 complete.
- [x] `kl_leaf` x `{none, lex}`: seeds 0--2 complete.
- [ ] cumulative beta=1 x `{none, lex}`: **6 seed-runs**.

Prediction: under lex on CIFAR-100, coarse/mid CE and TICE degrade monotonically
in `w_2`; under the baseline they do not. Parallel curves would mean the weights
are a plain scalarisation choice and 4.1.4's framing needs weakening.

- [ ] Report `w_2` next to every rule name.
- [ ] Report unweighted per-level train CE, not only test metrics.

## D. LH-projection base (`kl_leaf`, `feature_dim=0`) — COMPLETE (0 outstanding)

Establishes the pre-width LH anchor at the structural substrate. Here, "base"
means `weight_mode=kl_leaf`, `projection.feature_dim=0`, no advantage and no
other optional mechanism. `feature_dim=0` resolves to the dataset's taxonomy
width: 128 for CIFAR-100, 251 for CUB-200 and 200 for Aircraft.

- [x] Aircraft identity/level/`kl_leaf`/LH, `feature_dim=0`, seeds 0--2.
- [x] CUB-200 identity/level/`kl_leaf`/LH, `feature_dim=0`, seeds 0--2.
- [x] CIFAR-100 block/level/`kl_leaf`/LH, `feature_dim=0`, seeds 0--2. The
  existing identity bundle is off-substrate and remains only an LH
  frame-sensitivity point.

The CIFAR-100 base row is reproduced explicitly as:

```bash
FEATURE_DIM=0 WEIGHT_MODE=kl_leaf DATASETS=cifar100 \
FIXED_FRAME_MODE=orthonormal_random FIXED_FRAME_PER_LEVEL=true \
  scripts/hiercos/run_hiercos_lhdnn_projection.sh
```

All three base rows now have their matched `d512` row in E.

The base row is the reference for the width ablation, not the final reference
configuration for later LH experiments.

## E. Immediate LH representation-width ablation — COMPLETE (0 outstanding)

Isolates the representation correction by changing only
`projection.feature_dim`: `0` (the D row) versus `512`. The weighting remains
`kl_leaf`; frame, loss, seeds, optimiser and every other LH option remain fixed.
`feature_dim` is ignored when the projection is off, so there is no corresponding
baseline or explicit-lex width run.

This is a sequencing gate, not a late sensitivity check. The `d512` setting is
the reference LH representation for every subsequent ablation even if the
magnitude of the width gain differs by dataset; E quantifies that correction
rather than deciding whether later rows use it.

Protected rows = `sum(C_l[:-1])`: the auto-width base leaves 28/128 dimensions
reserved on CIFAR-100, 51/251 on CUB-200 and 100/200 on Aircraft. The fixed
`512`-dimensional representation supplies substantially more complement space
on every dataset, especially Aircraft, where the automatic representation
leaves half of the width in the protected non-leaf span.

- [x] Aircraft identity/level/`kl_leaf`/LH, `feature_dim=512`, seeds 0--2.
- [x] CUB-200 identity/level/`kl_leaf`/LH, `feature_dim=512`, seeds 0--2.
- [x] CIFAR-100 block/level/`kl_leaf`/LH, `feature_dim=512`, seeds 0--2.

Independent-selected checkpoint means (FPA/TICE), automatic width -> `d512`:
CIFAR-100 .7646/.0108 -> .7714/.0055; Aircraft .7952/.0072 ->
.8058/.0043; CUB-200 .7586/.0070 -> .7639/.0051. Thus the width correction
improves mean FPA by 0.68, 1.06 and 0.53 percentage points, respectively, while
also lowering mean TICE in all three cells. This establishes the corrected LH
reference; it does not isolate the projection operator from the wider adapter.

The launcher already defaults to `FEATURE_DIM=512`; set it explicitly in run
notes and dry runs anyway. These output directories must contain `_d512`.

The `d512` LH arm is wider than the no-projection and explicit-lex arms, whose
backbones remain at `total_nodes`. Therefore the later headline comparison
identifies the complete corrected LH configuration, not a width-matched effect
of the projection operator alone. A width-matched control would require the
backbone output dimension to be decoupled from `projection.enabled`.

## F. Post-width LH ablations — `feature_dim=512` only — COMPLETE (0 outstanding)

Every LH row after E inherits the `d512` representation. Change only the named
axis; never use the automatic-width D row as the parent for these experiments.

- [x] CIFAR-100 block/level/`equal`/LH, `feature_dim=512`, seeds 0--2:
  **3 seed-runs**. Together with E's CIFAR-100 `kl_leaf`/`d512` row, this is the
  matched LH weight contrast motivated in C.
- [x] Aircraft identity/level/`equal`/LH, `feature_dim=512`, seeds 0--2.
- [x] CUB-200 identity/level/`equal`/LH, `feature_dim=512`, seeds 0--2.
- [x] Aircraft identity/level/`kl_leaf`/LH + advantage, `feature_dim=512`,
  seeds 0--2; its matched no-advantage comparator in E is complete.
- [x] The matched headline comparison uses `kl_leaf`. E already supplies its
  LH-projection/`feature_dim=512` row at seeds 0--2 on all three datasets, so
  this check schedules no additional run. The `equal` rows above are the
  separate within-LH weight contrast; they do not replace the `kl_leaf`
  headline rows.

Completion is not evidence of stability: CUB-200 `equal`/LH has independent
FPA .7537/.6907/.2931 across seeds, and the Aircraft advantage pilot collapses
at seed 2 (FPA .0132, TICE .7351). Keep those outcomes in the analysis rather
than silently rerunning or averaging them away.

The LH arm adds learnable per-level heads and a terminal PReLU as well as the
backward projection. Until an adapter-only control exists, F identifies the
complete LH adaptation package rather than the projection operator alone.

## G. CIFAR-100 backbone capacity ladder — 6 outstanding (*sensitivity*)

Isolates: backbone capacity within the corrected LH-projection configuration.
Use `NUM_RUNS=3 ./scripts/hiercos/run_hiercos_cifar100_backbone_ladder.sh`; it
shrinks the WRN while fixing 32 px inputs, the block frame, per-level softmax,
`kl_leaf`, `feature_dim=512`, no advantage, and LH-projection. The ResNet-50/224
script changes architecture, pretraining, and resolution at once and cannot
answer this — it stays permanently on hold.

The WRN-28-8 anchor is E's completed CIFAR-100 block/level/`kl_leaf`/LH/`d512`
row at seeds 0--2, so that rung is free.

- [x] WRN-28-8 LH-projection: reuse E's `d512` anchor.
- [ ] WRN-16-8 LH-projection, seeds 0--2. **3 seed-runs.**
- [ ] WRN-28-4 LH-projection, seeds 0--2. **3 seed-runs.**
- [ ] Do **not** add baseline or explicit-lex arms to this ladder. Without a
  no-projection row at every rung, this is a capacity sensitivity of the
  complete LH configuration, not a capacity--projection interaction.
- [ ] Do **not** combine with the 224 px override: the hard-coded
  `avg_pool2d(out, 8)` then yields 7x7 and the embedding becomes `out_dim * 49`.

Interpretation: systematic deterioration across both reduced rungs would show
that the complete LH configuration is capacity-sensitive. Stability across the
depth-reduced and width-reduced rungs would make a simple parameter-count
explanation less plausible; this three-point sensitivity is not a scaling law.

---

## Deferred

- [ ] **Adapter-only head control** — highest-value deferred item. Enabling LH
  adds learnable per-level heads, a terminal PReLU, *and* the backward
  projection, so the LH comparison identifies the whole **LH adaptation
  package**, not the projection operator. State that limitation wherever it is
  reported. A
  projection-only claim needs adapter-only none / adapter-only lex / adapter+LH.
- [ ] **Further LH advantage expansion** — the Aircraft `kl_leaf`/`d512` pilot
  is complete at seeds 0--2; CIFAR-100 and CUB-200 remain unscheduled. Treat it
  as a separate model variant.
- [ ] **Direct subspace supervision** — separate research question (tau=0.1).
- [ ] **CIFAR-100 ResNet-50 / 224 px** — superseded by G. The Aircraft
  from-scratch bundle stays a documented negative pilot; do not extend it.
- [ ] **HCC** — complete at dense/global, all three datasets; report separately.

## Do not schedule as a factorial

- [ ] Do not cross every frame with every loss, weight, mechanism, backbone,
  and dataset.
- [ ] Do not run `orthonormal_block_random` alongside the canonical block pair.
- [ ] Do not treat identity with and without per-level construction as separate
  frames.
- [ ] Do not build a parallel `kl_reg` grid: with fixed targets, native KL and
  global CE have the same training gradient up to a target-entropy constant.
- [ ] Do not run C, D, E, F, or G on more than one structural substrate per
  dataset.
- [ ] Do not stack LH, HCC, and explicit lex in the attribution matrix.
- [ ] Do not re-scope the loss to `w_l * (CE_l + alpha * R_l)` mid-grid.
  Decided 2026-08-28: under `equal` it is an exact reparametrisation
  (`alpha -> 3*alpha`, identical gradients), so it would change nothing in the
  lex presets; under `kl_leaf` it moves the regulariser level split to
  .025/.077/.898 and would invalidate the completed `kl_leaf x lex` cells on all
  three datasets. The defect it fixes is presentational and is handled by the
  regulariser-fraction reporting check. Revisit only bundled with a
  `sqrt(C_l - 1)` regulariser normalisation, as a separate declared study with
  its own ablation against the upstream flat form
  (`util/hiercos_construction.py:325`).

## Disposition

Reuse as-is:

- [x] All section A and B bundles; `kl_leaf` x `{none, lex}` at all three
  structural substrates; all three structural-substrate `kl_leaf` LH bundles
  at both automatic width and `d512`; dense/global `fine_first`, HCC, and
  `coarse_first` lex bundles.
- [x] `equal`/lex at Aircraft and CUB-200 identity and CIFAR-100 block, seeds
  0--2.
- [x] `equal`/LH/`d512` at Aircraft and CUB-200 identity, seeds 0--2; Aircraft
  `kl_leaf`/LH/`d512` advantage pilot, seeds 0--2. Preserve the collapsed seeds
  as observed outcomes.
- [x] `..._cifar100_..._projection_kl_leaf_identity` — off-substrate LH point.
- [x] `..._cifar100_..._lex_coarse_first` (dense, `equal`, 2 seeds) and
  `..._lex_coarse_first_identity` (identity, `equal`, 3 seeds) — the evidence
  that CIFAR-100's lex regression is weight-driven. Keep and cite.

Retire, do not resume (leave on disk, exclude in analysis):

- [x] `..._cifar100_level_softmax_ce_reg_baseline_kl_leaf_d512_identity`
  (seed 0, 4 epochs) — CIFAR-100's `none` arm is at block and complete.
- [x] `..._cub200_level_softmax_ce_reg_baseline_kl_leaf_d512_identity`
  (seed 2, 23 epochs) — superseded by the complete `..._baseline_kl_leaf_identity`
  bundle. Earlier versions of this document told you to resume it; that was wrong.
- [x] All partial CIFAR-100 ResNet-50 directories.
- [x] Everything archived with `drop_last_eval=true`; old `fine_first` and
  `kl_leaf` variants; legacy conflict-gated lex runs; the failed hand-launched
  CIFAR ResNet directory.

Naming: `feature_dim` is ignored when the projection is off, so baseline
directories containing `_d512` are automatic-width baselines with a misleading
name. Never report them as width-512 controls.

## Launcher prerequisites

Resolved:

- [x] `FIXED_FRAME_PER_LEVEL` exposed and validated in `run_hiercos_baselines.sh`.
- [x] Block-frame baseline outputs carry an explicit `_block` suffix.

Open:

- [ ] `run_hiercos_baselines.sh:113` hard-codes `DATASETS=(cifar100)` despite its
  usage comment. Blocks C's Aircraft and CUB-200 baseline cells.
- [ ] Accept `cumulative_branching` and `marginal_branching` in
  `run_hiercos_lex.sh` and `run_hiercos_baselines.sh`; expose `WEIGHT_BETA`
  (mirroring `run_hiercos_lhdnn_projection.sh`) and tag cumulative output
  directories with the beta value. Blocks C.
- [ ] `run_hiercos_baselines.sh` appends `_d${FEATURE_DIM}` to no-projection
  output names; suppress it when the projection is off.
- [ ] `run_hiercos_cifar100_resnet50_pretrained.sh` prose says pretrained is the
  default while `PRETRAINED_MODE=false`. Only matters if that script is revived.
- [ ] Verify every generated output directory with `DRY_RUN=1` before training.
- [ ] For partial seeds pass the exact local `latest.pt`; never rely on an
  automatic resume guess or relaunch a completed seed.
- [ ] Nothing reports how much the LH projection actually removes. The `proj_*`
  keys are HCC, not LH, and the per-block parameter diagnostics are degenerate
  in LH runs (`delta_param_norm_t1 == t2t1 == t3t2t1` at every epoch, because
  the dense identity frame leaves every head non-`None` for every level loss).
  Log `||c_level|| / ||z||` per level before interpreting D, E or F.

## Reporting checks

- [ ] Require `test_metrics.yaml` and a final `test` event before marking a seed
  complete; treat partial logs as dynamics artifacts.
- [ ] Report the actual seed count; never report a single-seed cell as a finding.
- [ ] State the substrate on every mechanism table — it differs across datasets.
- [ ] Top-down rows from the top-down-selected checkpoint, independent rows from
  the independent-selected one.
- [ ] FPA, weighted AP, accuracy higher-is-better; AHD and TICE lower-is-better;
  percentage deltas in percentage points.
- [ ] Confirm lex activity from post-projection diagnostics and HCC from
  `proj_constraint_alpha`, not from directory names.
- [ ] When a lex row moves, read the unweighted per-level train CE before
  interpreting test metrics.
- [ ] Report the per-level regulariser fraction `alpha*R_l / loss_level_l`
  beside the unweighted per-level CE. Under `level` softmax it reaches
  .79/.86/.93 at epoch 100 on CIFAR-100, so `alpha` is not one number across
  levels and the level weights govern a shrinking share of the objective --
  they scale `CE_l` only, never `R_l`.
- [ ] A lex result is not evidence about LH. State this wherever a lex row is
  reported as a verdict on the approach: lex's projection reference is the
  instantaneous gradient, LH's is the accumulated head weights.
- [ ] Never mix archived `drop_last_eval=true` results into a headline aggregate.
