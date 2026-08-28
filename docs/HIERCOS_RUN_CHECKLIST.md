# Hier-COS Ablation Run Checklist

Audited 2026-08-27 from `config_resolved.yaml` and `test_metrics.yaml` under
`/scratch/g.saggini1/outputs`, not from directory names.

3 seeds (`0`,`1`,`2`) + split seed `0` for any thesis-table row; 2 seeds allowed
where marked *sensitivity*. Everything not named by the question stays fixed.

## Run queue

| P | Section | What | Seed-runs | Blocked by |
|---|---|---|---|---|
| 0 | C | `equal` x `{none, lex}` on all datasets | 10 | baseline dataset matrix; CIFAR lex seed 2 |
| 0 | C | CIFAR-100 block LH x `{equal, kl_leaf}` | 6 | — (LH launcher available) |
| 1 | C | cumulative beta=1 x `{none, lex}` on all datasets | 18 | weight-mode/beta launcher support |
| 1 | C | Aircraft marginal x `{none, lex}` | 6 | weight-mode launcher support |
| 2 | D | remaining LH cells at the weighting fixed after C | TBD | section C complete and analysed |
| 3 | E | LH representation-width sensitivity | 3 + 5 | section D reference row |
| 4 | F | CIFAR-100 WRN ladder at dense/global, 2 seeds | 8 | — |

Sections A and B are complete. Finish and analyse section C before launching
new LH-projection rows in D. Amended 2026-08-28: the CIFAR-100 weight x LH cells
moved *into* C, because the weight--LH interaction cannot be inferred from the
lex arm (see C, *Why LH is crossed here, not in D*).

Section F belongs to the Hier-COS ablation section of the thesis, but it is a
non-blocking capacity sensitivity: it may run after or alongside D and does not
choose D's frame, normaliser or weighting.

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

## C. Level-weight ablation — 40 outstanding

Isolates: the numerical level weights, crossed with **both** `{none,
lex_coarse_first}` at the structural substrate on all three datasets, and
additionally with `LH` on CIFAR-100. A lex-only sweep cannot distinguish a
generally better scalarisation from a weight--lex interaction; and it cannot be
transplanted to LH at all.

### Why LH is crossed here, not in D

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
on lex is not evidence about LH. Six CIFAR block cells settle it and
simultaneously close D's missing CIFAR row.

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
mode, not a nuisance parameter. Section C is therefore a prerequisite for the
matched LH comparison rather than an optional sensitivity appendix.

### Run matrix

All cells use seeds 0--2 and split seed 0.

**CIFAR-100 at block/level** — run in the order given:

- [ ] **(1)** `equal` x none: **3 seed-runs**. Blocking: without it the `equal`
  column has no baseline, so the 2x2 cannot separate a generally better
  scalarisation from a weight--lex interaction. Everything else in this section
  rests on a half-filled square until it lands.
- [x] **(1)** `equal` x lex: seeds 0--1 complete; **finish seed 2**.
- [ ] **(2)** `equal` x LH: **3 seed-runs**.
- [ ] **(2)** `kl_leaf` x LH: **3 seed-runs**. Also closes D's missing CIFAR
  row; the existing identity bundle is off-substrate.
- [x] `kl_leaf` x `{none, lex}`: seeds 0--2 complete. This is the failing cell:
  unweighted train CE 0.859/1.072/0.252 at epoch 50 against the baseline's
  0.124/0.196/0.261, FPA .7659 vs .7780+-.0018, TICE .0137 vs .0087. It is the
  predicted starvation result on the correct substrate, not a frame artifact.
- [ ] **(3)** cumulative beta=1 x `{none, lex}`: **6 seed-runs**. Refines a
  curve whose shape (1)--(2) already give; lowest priority in this section, and
  below the LH cells.

**Aircraft at identity/level:**

- [ ] `equal` x none: **3 seed-runs**.
- [x] `equal` x lex: seeds 0--2 complete.
- [x] `kl_leaf` x `{none, lex}`: seeds 0--2 complete.
- [ ] cumulative beta=1 x `{none, lex}`: **6 seed-runs**.
- [ ] marginal x `{none, lex}`: **6 seed-runs**.

**CUB-200 at identity/level:**

- [ ] `equal` x none: **3 seed-runs**.
- [x] `equal` x lex: seeds 0--2 complete.
- [x] `kl_leaf` x `{none, lex}`: seeds 0--2 complete.
- [ ] cumulative beta=1 x `{none, lex}`: **6 seed-runs**.

Prediction: under lex on CIFAR-100, coarse/mid CE and TICE degrade monotonically
in `w_2`; under the baseline they do not. Parallel curves would mean the weights
are a plain scalarisation choice and 4.1.4's framing needs weakening.

- [ ] Report `w_2` next to every rule name.
- [ ] Report unweighted per-level train CE, not only test metrics.

## D. LH-projection comparison — blocked by C

Isolates: none vs `coarse_first` lex vs LH at one matched structural substrate
and one matched weighting per reported comparison. Section C must be analysed
before the headline weighting is fixed; never compare the best lex weight with
an LH or baseline row trained under a different rule.

- [x] Aircraft identity/level/`kl_leaf`: none, lex, LH, seeds 0--2 available as
  the incumbent-weight reference.
- [x] CUB-200 identity/level/`kl_leaf`: none, lex, LH, seeds 0--2 available as
  the incumbent-weight reference.
- [x] CIFAR-100 block/level/`kl_leaf`: none and lex, seeds 0--2 complete.
- [x] CIFAR-100 block/level/`kl_leaf`/LH: seeds 0--2 remain missing. **Moved
  into section C** and paired there with `equal` x LH, because the weight--LH
  interaction cannot be inferred from the lex arm. The existing identity bundle
  is off-substrate and remains only an LH frame-sensitivity point.
- [ ] After C, record the rule used for the headline matched comparison and add
  any missing LH cells for that same rule on all three datasets.

The LH arm adds learnable per-level heads and a terminal PReLU as well as the
backward projection. Until an adapter-only control exists, D identifies the
complete LH adaptation package rather than the projection operator alone.

## E. LH representation budget (`projection.feature_dim`) — 3 + 5 outstanding

Isolates: LH width. **LH-only** — `feature_dim` is ignored when the projection
is off.

Widths are *already* matched in section D: `models/hiercos/model.py:172` gives
the baseline `total_nodes` too, and `feature_dim=0` resolves to the same. So
this is not a section D control; it asks whether tying LH's width to the
taxonomy handicaps it. Protected rows = `sum(C_l[:-1])`: 28/128 = 22% CIFAR-100,
51/251 = 20% CUB-200, **100/200 = 50% Aircraft** — and Aircraft is where LH's
deficit is largest (-3.0 pp vs -0.6 pp on CUB-200).

- [ ] Aircraft identity/level/`kl_leaf`/LH, `feature_dim=400`, seeds 0--2.
  **3 seed-runs.** Run first.
- [ ] CUB-200 `feature_dim=512`: resume seed 1 from its epoch-8 checkpoint, start
  seed 2, do not rerun seed 0 (0.7698/.0047 vs .7586/.0070 at auto width, n=1).
  **2 seed-runs**, only if Aircraft recovers.
- [ ] CIFAR-100 `feature_dim=256`, seeds 0--2. **3 seed-runs**, same condition.

- [ ] Keep these rows **out** of the section D table: at `feature_dim >
  total_nodes` the LH arm is wider than the other two arms. A matched version
  would need `model.py:172` changed to decouple backbone `out_dim` from
  `projection.enabled`; do not attempt unless E changes C's conclusion.

## F. CIFAR-100 backbone capacity ladder — 8 outstanding (*sensitivity*)

Isolates: capacity. Use
`NUM_RUNS=2 ./scripts/hiercos/run_hiercos_cifar100_backbone_ladder.sh`; it
shrinks the WRN at fixed implementation, 32 px, head, frame, and loss. The ResNet-50/224
script changes architecture, pretraining, and resolution at once and cannot
answer this — it stays permanently on hold.

Base arm dense/global x {none, lex}: both WRN-28-8 anchors already exist at
seeds 0--2 (.7752/.0104 and .7732/.0058), so that rung is free, and it is the
arm carrying the headline explicit-lex claim.

- [x] WRN-28-8 x {none, lex}: reuse the dense/global anchors.
- [ ] WRN-16-8 x {none, lex}, 2 seeds each. **4 seed-runs.**
- [ ] WRN-28-4 x {none, lex}, 2 seeds each. **4 seed-runs.**
- [ ] Do **not** run the ladder at the block/level substrate as well.
- [ ] Do **not** combine with the 224 px override: the hard-coded
  `avg_pool2d(out, 8)` then yields 7x7 and the embedding becomes `out_dim * 49`.

Prediction: less capacity means harder level competition, so the lex-minus-
baseline gap in coarse CE and TICE should widen monotonically as capacity falls.
A flat gap would mean the CIFAR-100 regression is purely the weight mechanism.

---

## Deferred

- [ ] **Adapter-only head control** — highest-value deferred item. Enabling LH
  adds learnable per-level heads, a terminal PReLU, *and* the backward
  projection, so section D identifies the whole **LH adaptation package**, not
  the projection operator. State that limitation wherever D is reported. A
  projection-only claim needs adapter-only none / adapter-only lex / adapter+LH.
- [ ] **LH advantage** (`ADVANTAGE_ENABLED=true`) — separate model variant.
- [ ] **Direct subspace supervision** — separate research question (tau=0.1).
- [ ] **CIFAR-100 ResNet-50 / 224 px** — superseded by F. The Aircraft
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
- [ ] Do not run C, D, E, or F on more than one structural substrate per
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
  structural substrates; Aircraft and CUB-200 `kl_leaf` LH bundles;
  dense/global `fine_first`, HCC, and `coarse_first` lex bundles.
- [x] `equal`/lex at Aircraft and CUB-200 identity, seeds 0--2; at CIFAR-100
  block, seeds 0--1. Finish only CIFAR-100 seed 2.
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
  Log `||c_level|| / ||z||` per level before interpreting D or E.

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
