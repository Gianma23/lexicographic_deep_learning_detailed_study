# Experiment Matrix: Hier-COS Ablations and Cross-Model Extension

Status of this document: it schematizes the trial space implied by the code, then
re-organizes it around research questions. Run status is inferred from the presence of
directories under `/scratch/g.saggini1/outputs` and from `scripts/hiercos/*.sh`; it is
**not** a verification that each run completed or that the intended mechanism actually
activated. Before citing any row, confirm against `config_resolved.yaml` and
`test_metrics.yaml`, and for HCC confirm activation from `proj_constraint_alpha` in
`run_log.jsonl` rather than from the directory name.

Convention used below:

- **[code]** — enforced or defined in the implementation, with a file reference.
- **[interp]** — my reading of what the trial is for. Open to revision.

---

# Part A — The trial space

## A.1 Axes (Hier-COS)

| # | Axis | Config key | Values | Source |
|---|---|---|---|---|
| 1 | Fixed frame | `model.fixed_frame_mode`, `model.fixed_frame_per_level` | `identity`, `orthonormal_block_random` (= random + per-level), `orthonormal_random` (dense) | [model.py:373-393](models/hiercos/model.py#L373-L393) |
| 2 | Loss / softmax scope | `model.loss` | `kl_reg`, `global_softmax_ce_reg`, `level_softmax_ce_reg` | [losses.py:11](models/orthonormal_plugin/losses.py#L11) |
| 3 | Level weighting | `model.weight_mode` | `equal`, `kl_leaf`, `kl_coarse` | [losses.py:12](models/orthonormal_plugin/losses.py#L12) |
| 4 | Learnable transform | `model.transform_mode` | `full`, `bn_linear`, `final_only` | [model.py:343-347](models/hiercos/model.py#L343-L347) |
| 5 | Gradient-space lex | `train.lexicographic.enabled` | `false`, `true` | [config.py:106-123](train/lexicographic/config.py#L106-L123) |
| 5a | Lex priority order | `.projection_mode` | `coarse_first`, `fine_first`, `pairwise_orthogonal` | [config.py:9](train/lexicographic/config.py#L9) |
| 5b | Lex projection rule | `.projection_rule` | `orthogonalize_all`, `conflict_only` | [config.py:10](train/lexicographic/config.py#L10) |
| 5c | Lex onset | `.start_epoch` | `0`, `80`, … | [config.py:112](train/lexicographic/config.py#L112) |
| 6 | LH-style projection | `model.projection.enabled` | `false`, `true` | [model.py:270](models/hiercos/model.py#L270) |
| 6a | PReLU ρ variant | `.rho_enabled` | `false`, `true` | [model.py:271-274](models/hiercos/model.py#L271-L274) |
| 6b | Parent advantage | `.advantage_enabled` | `false`, `true` | [model.py:275-278](models/hiercos/model.py#L275-L278) |
| 6c | Feature width | `.feature_dim` | `0` (= total nodes), `512`, … | [model.py:290-317](models/hiercos/model.py#L290-L317) |
| 7 | Output-space HCC | `hcc.enabled` | `false`, `true` | [config_validation.py:763](train/config_validation.py#L763) |
| 7a | α schedule | `hcc.alpha_schedule` | `exp`, `tanh`, `linear`, `step` | [config_validation.py:788-792](train/config_validation.py#L788-L792) |
| 7b | α onset | `hcc.alpha_start_epoch` | `0`, `80` | [run_hiercos_hcc_grid.sh](scripts/hiercos/run_hiercos_hcc_grid.sh) |
| 7c | Test-time only | `hcc.final_test_only` | `false`, `true` | [config_validation.py:764](train/config_validation.py#L764) |
| 8 | Backbone | `model.variant`, `model.pretrained`, `model.pool` | `haframe_wide_resnet` / `haframe_resnet50`; `true`/`false`; `max`/`average` | [model.py:349-366](models/hiercos/model.py#L349-L366) |
| 9 | Dataset | `dataset.name` | `cifar-100`, `cub-200-2011`, `fgvc-aircraft` (iNat21 supported by the framework, no Hier-COS runs) | `configs/hiercos/` |
| 10 | Seed | `train.seed` | 3 seeds default (`NUM_RUNS=3`) | `scripts/run_seed_utils.sh` |

Taking only axes 1–5 and 9 as a full factorial gives 3 × 3 × 3 × 3 × 7 × 3 = **1 701
configurations per seed**, before any head variant. The factorial is not the plan; the
constraints below and the research questions in Part B cut it to roughly 60–80 configs
that actually carry information.

## A.2 Constraints that prune the factorial [code]

These are hard validation errors, not conventions. They are what makes the matrix
tractable, and several of them *are themselves findings* about how the mechanisms relate.

1. **Lex needs exactly three differentiable level losses.**
   `kl_reg` therefore cannot be combined with lex — it exposes no per-level objectives.
   [config.py:160-181](train/lexicographic/config.py#L160-L181)
2. **LH-DNN admits no mechanism arms — it is a baseline only.** Lex is rejected
   ([config.py:132-133](train/lexicographic/config.py#L132-L133)) and HCC is rejected
   ([config_validation.py:756-759](train/config_validation.py#L756-L759)); the plugin is
   excluded by design. See C.0 for the argument.
3. **Lex on HRN requires `model.loss: level_marginal`.** [config.py:171-175](train/lexicographic/config.py#L171-L175)
4. **Lex on H-CAST requires `model.loss.globalkl: false`.** [config.py:152-158](train/lexicographic/config.py#L152-L158)
5. **LH projection requires `level_softmax_ce_reg`** — the per-level softmax is what gives
   each branch its own loss. [config_validation.py:836-846](train/config_validation.py#L836-L846)
6. **LH projection requires an identity or per-level block-diagonal frame**, so level heads
   stay independent. A dense random frame mixes levels and destroys the branch structure.
   [model.py:378-386](models/hiercos/model.py#L378-L386)
7. **LH projection is non-trivial only if `feature_dim > sum(classes[:-1])`.**
   [model.py:310-317](models/hiercos/model.py#L310-L317)
8. **HCC requires exactly three levels**; HRN + HCC requires `level_marginal`.
   [config_validation.py:759-775](train/config_validation.py#L759-L775)
9. **The orthonormal plugin requires `mixup=0` and `cutmix=0`.**
   [config_validation.py:806-810](train/config_validation.py#L806-L810)

Constraint 5 is worth stating in the thesis as a design consequence rather than a config
detail: *global* softmax couples all nodes into one normalizer, so there is no such thing
as a separable per-level gradient to project. Constraints 1 and 5 are the same fact seen
from the lex side and the LH side.

## A.2b The three hierarchy mechanisms are pairwise mutually exclusive

This is the structural fact that collapses the matrix, and it deserves to be stated as one
rule rather than three scattered constraints:

| Pair | Status | Reference |
|---|---|---|
| HCC × LH projection | **enforced** | [model.py:423-430](models/hiercos/model.py#L423-L430), [config_validation.py:780-787](train/config_validation.py#L780-L787) |
| LH projection × lex | **enforced** | [config_validation.py:887-891](train/config_validation.py#L887-L891) |
| HCC × lex | **intended, not currently enforced** | see below |

So the mechanism axis is a **4-way single choice — `{none, HCC, LH projection, lex}` — not a
2×2×2 factorial.** Every comparison between these three is a between-arms comparison against
the shared `none` baseline; no combination arm exists, and none should be planned.

**On HCC × lex.** I searched `train/config_validation.py`, `train/lexicographic/`, and
`train/engine.py` and found no check rejecting this pair. The exclusion is real but the
guard is missing, so today the config would be accepted and would train something
incoherent rather than erroring.

The mechanistic reason it must be excluded: HCC's least-squares projection onto
"children sum to parent" is a **joint** constraint across all three levels
([models/common/hcc.py](models/common/hcc.py#L4-L6)), and when HCC is on, the level losses
are computed on the HCC-corrected effective logits, not the raw ones
([losses.py:446-448](models/orthonormal_plugin/losses.py#L446-L448)). Each level's loss
therefore depends on *every* level's raw logits through the projection. Lex's entire premise
is that there are three separable level objectives whose gradients can be ordered — under
HCC there are not. The lexicographic ordering would be applied to gradients that HCC has
already cross-coupled, so the priority it imposes would not mean what it claims to mean.

This is the same separability premise as constraints 1 and 5, violated from a third
direction. Worth a validation guard mirroring the LH × lex one at line 887, and worth a
sentence in the thesis: output-space hierarchy constraints and gradient-space level ordering
are not composable, because the former destroys the objective separability the latter
requires.

## A.3 What has been run on Hier-COS

Grouped by the axis being varied. Each row is 3 seeds unless noted.

**Frame ablation (identity vs. dense random)** — CIFAR-100, CUB, Aircraft, in both
`global_softmax_ce_reg` and `level_softmax_ce_reg`, baseline and lex.
Directories: `*_baseline_kl_leaf` vs `*_baseline_kl_leaf_identity`,
`*_lex_orthogonalize_all_coarse_first` vs `*_..._coarse_first_identity`.

**Loss scope (global vs. level softmax)** — all three datasets, baseline and lex,
crossed with frame. Plus one `hiercos_cifar100_per_level_kl_reg_baseline_equal`.

**Level weights (`equal` vs `kl_leaf`)** — all three datasets, baseline and lex.
`kl_coarse` appears in the script enum but I found no run directory for it.

**Transform mode (`full` / `bn_linear` / `final_only`)** — all three datasets, mostly under
lex; `run_hiercos_transform_ablation.sh`.

**Lex variants** — `coarse_first` vs `fine_first` on all three datasets;
`conflict_only` on CIFAR-100 only (`hiercos_cifar100_global_softmax_ce_reg_lex_conflict_only_coarse_first`).
`pairwise_orthogonal` has a script path but no run directory.

**LH projection** — CIFAR-100, CUB, Aircraft with `level_softmax_ce_reg` + identity frame;
`_d512` width variant; `_rho` variant (CUB only); `_noadapt` (CIFAR-100 only).
`advantage_enabled` is implemented and script-exposed but I found no run directory.

**HCC on Hier-COS** — `run_hiercos_hcc_grid.sh` exists (step@0, step@80, T=10) but I found
no `hiercos_*hcc*` output directories. Treat as **specified, not yet run**.

Note: the working tree (uncommitted) removes two former preconditions on HCC + Hier-COS —
the `level_softmax_ce_reg` requirement and the per-level/block-diagonal frame requirement
(`git diff train/config_validation.py`, around line 773). HCC on Hier-COS is therefore now
legal with a global softmax and a dense frame as well, which widens this arm before it has
been run even once. Worth deciding deliberately which of those cells you actually want,
rather than inheriting the wider space by default.

**Backbone / pretraining** — `hiercos_cifar100_..._identity_resnet50` (pretrained ResNet-50
on CIFAR-100), `hiercos_aircraft_..._fromscratch_...` vs pretrained.

### Visible gaps

| Gap | Why it matters |
|---|---|
| `pairwise_orthogonal` never run | The order-free control for RQ4; without it "coarse-first helps" cannot be separated from "any projection helps" |
| `conflict_only` on one dataset only | Cannot tell whether the conflict-gating result generalizes |
| `kl_coarse` never run | Leaves the weight axis one-sided; see RQ3 |
| HCC on Hier-COS not run | RQ7 currently has no Hier-COS arm at all |
| `advantage_enabled` never run | One of the three LH-projection sub-variants is untested |
| No seed-matched lex × weight cross | The known `equal` / `kl_leaf` confound (RQ3) |
| **No HCC × lex validation guard** | The pair is incoherent (A.2b) but currently accepted; an invalid run could be produced and analysed in good faith |
| No LH vs. lex head-to-head at matched settings | RQ6's central comparison; the two implementations of one principle have not been run against each other on identical frame/loss/weights/seeds |

---

# Part B — Research questions

This is the part meant to be read linearly. Each question states what is being asked, the
minimal contrast that answers it, and what each outcome would license as a claim.

## RQ1 — Is Hier-COS's benefit the fixed frame, or the inference rule?

Hier-COS bundles three things that are usually credited as one: a **frozen orthonormal
frame** over taxonomy nodes, a **subspace-norm decoding rule** (predictions come from the
norm of the projection onto each taxonomy subspace, with no softmax over those scores,
[model.py:640-643](models/hiercos/model.py#L640-L643)), and a **loss with a level
regularizer**. The paper's contribution narrative attaches to the frame. The trial exists
to test whether that attribution survives. [interp]

The contrast is a three-point ladder on axis 1, holding everything else fixed:

| Setting | What survives | What is removed |
|---|---|---|
| `orthonormal_random` (dense) | frame + subspace decoding + node-block layout | — |
| `orthonormal_block_random` (per-level) | orthonormality within a level, node-block layout | cross-level mixing |
| `identity` | node-block layout + subspace decoding | the rotation entirely |

Reading the outcomes:

- **identity ≈ dense random** → the frame is not doing the work. What remains is the
  node-per-taxonomy-node parametrization and the subspace-norm inference. The honest claim
  becomes "Hier-COS's gain is attributable to its decoding and output parametrization,
  not to the fixed orthonormal frame."
- **identity ≪ dense random** → the frame carries the benefit, and the mechanism is
  cross-level coordinate mixing.
- **block-diagonal in between** → separates "orthonormality helps" from "cross-level
  mixing helps." This is the row that makes the ladder diagnostic rather than binary.

Note the ladder is also load-bearing elsewhere: constraint 6 forces identity or
block-diagonal whenever LH projection is on, so RQ1's middle rung is a prerequisite for
RQ6, not an optional extra.

Recorded caveat (memory: `hiercos-frame-headroom`): on CIFAR-100 the identity deficit
behaves as a three-way interaction with `coarse_first` lex, visible as node-magnitude
instability. RQ1 should therefore not be reported as a single main effect on CIFAR-100
without the lex arm alongside it.

## RQ2 — Where must the softmax live, and what does that make possible?

`global_softmax_ce_reg` normalizes across all taxonomy nodes at once;
`level_softmax_ce_reg` normalizes within each level. This looks like a loss detail and is
actually the structural gate for most of the rest of the matrix.

With a global softmax there is one coupled objective, so there is no separable per-level
loss to (a) weight meaningfully, (b) hand to the lexicographic projector as three ordered
objectives, or (c) attach to independent LH branch heads. The code enforces exactly this:
lex demands three level losses [config.py:177-181](train/lexicographic/config.py#L177-L181),
and LH projection demands `level_softmax_ce_reg`
[config_validation.py:836-846](train/config_validation.py#L836-L846).

So RQ2 has two distinct readings and both belong in the thesis:

1. **As a standalone ablation** — does per-level normalization cost or gain accuracy
   relative to global, at matched frame and weights? This is a fair-comparison question.
2. **As an enabling condition** — level softmax is the prerequisite that makes branching,
   per-level weighting, and lexicographic ordering definable at all. If it costs accuracy,
   that cost is the *entry price* for every gradient-space method in this thesis and must
   be reported as such, not hidden inside the lex-vs-baseline delta.

Reading 2 is the reason the baseline must be run in both scopes: a lex run at
`level_softmax_ce_reg` compared against a baseline at `global_softmax_ce_reg` confounds
the optimizer with the normalizer.

## RQ3 — Does the level weighting drive the conclusions?

`equal`, `kl_leaf` (leaf-heavy), `kl_coarse` (the flipped vector,
[losses.py:408](models/orthonormal_plugin/losses.py#L408)). This is the axis the supervisor
objected to (`docs/hiercos_level_weights.md`), on the grounds that the weights are
empirical rather than derived.

Two separate questions live here, and conflating them is what makes the objection stick:

- **Is the weight choice a large effect?** If moving `equal` → `kl_leaf` moves fine
  accuracy by ~2pp, then any method comparison at unmatched weights is uninterpretable.
- **Is the ranking of methods weight-dependent?** This is the harder and more important
  one. A method that only wins at one weight setting has not been shown to win.

There is a known confound to clear (memory: `lex-vs-baseline-weight-confound`): the lex
presets default to `equal` while the baselines default to `kl_leaf`, worth roughly 2pp of
fine accuracy. Until lex × weight is run seed-matched as a genuine 2 × 2, the headline
lex-vs-baseline number carries a weight effect inside it. **This is the single highest-value
gap in the current matrix** — it does not add a new mechanism, it decides whether the
existing lex results mean what they appear to mean.

The `kl_coarse` cell is the falsification arm. Memory (`lex-buys-consistency-not-accuracy`)
records that lex improves TICE in 6/6 settings while leaf-heavy weighting trades the
reverse — that is, weighting and lex push on the same accuracy/consistency tradeoff from
opposite ends. `kl_coarse` tests whether coarse-heavy weighting reproduces lex's consistency
gain *without* any gradient projection. If it does, the honest claim about lex weakens
sharply: it would mean lex is one way to buy consistency and simple reweighting is another,
cheaper one.

## RQ4 — Does gradient-space lexicographic optimization work, and is the *ordering* what matters?

Lex projects the per-level gradients so that higher-priority levels are protected. The
question is not only "does it help" but "does the lexicographic *structure* help, or is any
gradient decorrelation enough?" [interp]

The design has a control built in, which is why axis 5a has three values and not two:

| Arm | Role |
|---|---|
| `coarse_first` | the hypothesis — coarse levels take priority |
| `fine_first` | reversed priority; if it matches coarse-first, ordering is not the mechanism |
| `pairwise_orthogonal` | **order-free control** — decorrelates without imposing any priority |

`pairwise_orthogonal` is the arm that makes this a real test and it has not been run. Without
it, "coarse-first improves TICE" is compatible with "projecting gradients at all improves
TICE." I would prioritize this above any new dataset.

Axis 5b asks a different question: `orthogonalize_all` projects unconditionally,
`conflict_only` projects only when gradients actually conflict. If `conflict_only` matches
`orthogonalize_all`, the benefit is conflict resolution and the projection is inert most of
the time — which is a much more mechanistic claim, and testable directly from the
lexicographic diagnostics in `run_log.jsonl` (how often the conflict gate fires).

Axis 5c (`start_epoch` 0 vs 80) asks whether lex is a training-dynamics intervention or a
fine-tuning correction, mirroring the HCC step@0 / step@80 design so the two families stay
comparable.

## RQ5 — Does lex need a learnable transform to act on?

`transform_mode`: `full` → `bn_linear` → `final_only` progressively removes the learnable
transformation between backbone and fixed frame. At `final_only` there is no transform
module in the optimizer path at all
([model.py:456-460](models/hiercos/model.py#L456-L460)).

This is a mechanism question, not a capacity question [interp]. Gradient projection acts on
whatever parameters sit in the projected path. If lex's benefit shrinks as the transform
shrinks, the effect is localized to the transform and lex is essentially shaping a learned
re-embedding. If the benefit survives at `final_only`, lex is acting on the backbone itself
and the claim is much stronger and more general — it would mean the method transfers to any
architecture, which is what Part C is betting on.

## RQ6 — One principle, two implementations: LH projection vs. lexicographic projection

**These are not two complementary methods. They are the same idea implemented two ways.**
Both enforce "do not let the finer objective disturb what the coarser objective already
established," and both do it in gradient space. They differ only in *where the projection
is performed*, and they are mutually exclusive in code
([config_validation.py:887-891](train/config_validation.py#L887-L891)).

| | LH projection | Lexicographic projection |
|---|---|---|
| **Implementation** | **differentiable layer, inside the graph** | **explicit gradient surgery, outside the graph** |
| Where | `z - c + sg(c)` in forward ([model.py:590](models/hiercos/model.py#L590)) | on the three level-loss gradients, before the optimizer step |
| How the projection is obtained | autograd carries it automatically | computed by brute force and applied by hand |
| Granularity | per-sample, at branch points | per-level loss, whole gradient |
| Scope | first-order, branch-point only (memory: `lhdnn-theorem5-scope`) | full ordered objective set |
| Active at inference | no — forward value unchanged (memory: `lh-projection-backward-only`) | no |

The LH version is elegant and cheap: the constraint is expressed once in the forward pass
and the backward pass inherits it for free. Its cost is that the guarantee it gets is only
what the graph gives — first-order, per-sample, at branch points, and conditional on an
idempotent terminal activation. The lex version is the blunt instrument: it ignores how the
graph is built, takes whatever three gradients come out, and orthogonalizes them directly.
Its cost is compute and the need to hold three gradients at once; its benefit is that the
ordering holds by construction, over the whole objective, with no activation assumption.

So RQ6 is not "do they stack" — they cannot. It is: **does the in-graph differentiable
formulation buy the same ordering as explicit projection, more cheaply, or does its
first-order/branch-point restriction cost real hierarchy consistency?** That is a
head-to-head at matched frame, loss, weights, and seeds, scored primarily on TICE and AHD,
with the `none` baseline as the shared reference.

Sub-axes on the LH side probe exactly where its guarantee is thin: `rho_enabled` makes the
protected subspace sample-dependent through the PReLU derivative
([model.py:530-545](models/hiercos/model.py#L530-L545)), testing whether the idempotency
requirement bites in practice; `advantage_enabled` adds a parent-logit baseline (untested);
`feature_dim` (512 vs. total nodes) sets how much room exists above the protected subspace,
floored by constraint 7.

## RQ7 — Output-space constraints (HCC) vs. gradient-space (lex)

This is the contrast that keeps the thesis honest about what "lexicographic" means. HCC
constrains the *outputs* via an affine hierarchy projection; lex and LH constrain the
*gradients*. Per AGENTS.md, HCC is not an explicit lexicographic optimizer unless
`train.lexicographic.enabled` is set.

Per A.2b the three mechanisms are pairwise exclusive, so the achievable cells are exactly
**{none, HCC, LH, lex}**. RQ7 is therefore a clean between-arms comparison of an
output-space constraint against two gradient-space ones, all against a shared baseline —
and the absence of combination arms is itself the result: hierarchy consistency can be
bought in output space *or* in gradient space, but the two are not composable, because HCC
destroys the objective separability that gradient ordering requires.

The α-schedule axis (step@0 vs step@80) asks whether hierarchy consistency should be imposed
from initialization or applied as a late correction, and `final_test_only` isolates the
extreme case: constraint applied *only* at test time, which cleanly separates "the
constraint improved the learned representation" from "the constraint fixed up the
predictions." That is the cheapest and most decisive HCC arm, and the whole HCC-on-Hier-COS
family is currently unrun.

## RQ8 — Backbone and pretraining as threats to validity

`haframe_wide_resnet` (from scratch, CIFAR-native) vs. `haframe_resnet50` (ImageNet
pretrained). This axis is not a contribution — it exists to bound the others [interp]. If the
frame/lex/weight effects invert between a scratch WideResNet and a pretrained ResNet-50, then
every conclusion is conditional on the feature extractor.

Per AGENTS.md this is also where the local-extrapolation caveats belong: HRN on CIFAR-100,
Hier-COS on CUB, and this repo's CIFAR hierarchy construction are all deviations from the
paper-aligned settings and must be reported separately from paper-matched rows.

## RQ9 — Dataset generality

CIFAR-100 (coarse hierarchy, small images, scratch training), CUB-200 (fine-grained, shallow
semantic spread), FGVC-Aircraft (fine-grained, genuinely hierarchical manufacturer→family→
variant). The three differ in *how much the hierarchy actually constrains the fine label*,
which is the property every method here depends on. A method that works on Aircraft and not
CUB is telling you it needs a semantically informative hierarchy — that is a result, not a
failure.

---

# Part C — Cross-model extension

Everything above is Hier-COS-specific in *setting* but not in *mechanism*. Three mechanisms
generalize, by three different routes.

## C.0 Model × mechanism applicability

The governing rule first: **LH-DNN is a baseline only.** It carries no mechanism arms — not
lex, not HCC, not the plugin. Everything else below is scoped to the other four models.

| Model | Lex | HCC | Orthonormal plugin | Role in the thesis |
|---|---|---|---|---|
| H-CAST | yes (`globalkl: false`) | yes | yes | primary method model |
| Hier-COS | yes ({global,level}`_softmax_ce_reg`) | yes | n/a (native frame) | primary method model |
| HRN | yes (`level_marginal`) | yes (`level_marginal`) | yes | baseline + extension |
| HT-CapsNet | yes | yes | yes | baseline + extension |
| **LH-DNN** | **no — rejected** | **no — rejected** | **no — excluded by design** | **baseline only** |

Lex support: [config.py:132-139](train/lexicographic/config.py#L132-L139).
HCC support: `hcc_supported_models = {"hcast", "hrn", "ht_capsnet", "hiercos"}`
([config_validation.py:756-759](train/config_validation.py#L756-L759)), with a dedicated
regression test `test_hcc_is_rejected_for_lhdnn`
([tests/test_hcc_model_support.py:74-77](tests/test_hcc_model_support.py#L74-L77)).
LH-DNN also has no HCC code path at all.

### Why LH-DNN is baseline-only, as an argument rather than a limitation

The two exclusions have different standing and the thesis should not blur them.

**HCC × LH-DNN is code-enforced and mechanically grounded.** HCC constrains output logits
into hierarchy consistency; LH-DNN's outputs are produced through its own branch-point
projection. Imposing an output-space cascade on top would overwrite the structure LH-DNN
exists to demonstrate.

**Plugin × LH-DNN is technically possible but excluded by design.** The wrapper accepts any
non-Hier-COS model ([models/__init__.py:33-43](models/__init__.py#L33-L43)), so it *would*
load. But the plugin consumes the base model's per-level scores and re-classifies them
through the fixed frame, replacing the classification pathway and blanking
`effective_logits_per_level` ([wrapper.py:77-91](models/orthonormal_plugin/wrapper.py#L77-L91)).
Applied to LH-DNN, the loss becomes the plugin's, and what remains of LH-DNN is essentially
a backbone. A result from that arm could not be attributed to LH-DNN, so it would not
answer any question worth asking.

The unifying point: **LH-DNN's contribution *is* its head and projection mechanism.** Every
mechanism in this thesis acts on exactly that part of the network, so any arm applied to
LH-DNN either duplicates what it already does (lex — see below) or replaces it (HCC,
plugin). It is therefore the right control precisely because it admits no arms: it holds the
"hierarchy handled natively by the architecture" position fixed while the other four models
vary.

### The lex exclusion specifically
Per RQ6, LH projection *is* the differentiable-layer implementation of what lex does by
explicit projection. LH-DNN already applies that mechanism natively at its branch points, so
enabling lex on top would apply the same principle twice by two different routes — which is
why the same pair is also rejected inside Hier-COS
([config_validation.py:887-891](train/config_validation.py#L887-L891)). The exclusion is
consistent across both models and follows from the mechanism, not from missing support.

The comparison LH-DNN would have supplied is instead available *within* Hier-COS: the
`projection.enabled` arm and the `lex.enabled` arm are the two implementations of the one
principle, run on identical backbone, frame, loss, and weights. That is a cleaner
head-to-head than a cross-model LH-DNN-vs-lex comparison would have been, since it holds
everything but the implementation fixed.

## C.1 Lexicographic arms to run

Lex is the one mechanism that reaches four models, so it carries most of the
cross-architecture weight. Each new model arm should carry at minimum the RQ4 core:
`coarse_first` / `fine_first` / `pairwise_orthogonal` × `orthogonalize_all` /
`conflict_only`, at matched weights.

| Model | Status |
|---|---|
| H-CAST | run (`hcast_lex_*`, incl. step@80) |
| Hier-COS | run |
| HRN | **not run** — needs `model.loss: level_marginal` |
| HT-CapsNet | **not run** |

## C.2 The orthonormal plugin — ports the fixed frame onto three other models

`OrthonormalPluginWrapper` wraps a non-Hier-COS model
([models/__init__.py:33-43](models/__init__.py#L33-L43)) and exposes its own `loss`,
`weight_mode`, `transform_mode`, `fixed_frame_mode`, `fixed_frame_per_level`, and `alpha`
([config_validation.py:800-834](train/config_validation.py#L800-L834)).

This is the mechanism that makes **RQ1 and RQ2 model-independent**. The identity-vs-random
ladder and the global-vs-level softmax question can be replicated on **H-CAST, HRN, and
HT-CapsNet** — not LH-DNN, per C.0. If "identity ≈ random" replicates across three
backbones plus Hier-COS, the claim stops being about Hier-COS and becomes a claim about
fixed-frame hierarchical heads in general — a substantially stronger thesis contribution
than a single-model ablation.

One such run exists already:
`hrn_cub200_orthonormal_plugin_level_softmax_ce_reg_final_only_identity`.

Caveat: the plugin requires `mixup=0` and `cutmix=0`, so plugin arms are not directly
comparable to baseline runs of models whose default recipe uses either. H-CAST in particular
has `nomixup_nosmoothing` variants for this reason — use those as the matched baseline.

## C.3 HCC — cross-model over the same four

HCC lives in `models/common/hcc.py` and is supported for exactly
`{hcast, hrn, ht_capsnet, hiercos}` — the same four models as lex, and for the same reason:
LH-DNN is excluded (C.0). Requires exactly three levels; HRN additionally requires
`level_marginal`.

| Model | Status |
|---|---|
| H-CAST | run extensively (step@0, step@80, inversestep@80, linear, cond2, nokl) |
| Hier-COS | **not run** (see A.3 — arm recently widened) |
| HRN | **not run** — needs `level_marginal` |
| HT-CapsNet | **not run** |

That HCC and lex are supported on precisely the same four models is convenient: RQ7's
output-space-vs-gradient-space comparison can in principle be run on every model that admits
either, with LH-DNN as the untouched baseline throughout.

## C.4 Suggested order of work

Ranked by information gained per GPU-hour, not by convenience:

0. **Add the HCC × lex validation guard** — no GPU cost, prevents an incoherent run from
   entering the analysis. Mirror the LH × lex check at
   [config_validation.py:887-891](train/config_validation.py#L887-L891).
1. **Close the RQ3 confound** — lex × {`equal`, `kl_leaf`} seed-matched, all three datasets.
   Re-interprets results you already have; adds no new mechanism. Highest value.
2. **LH vs. lex head-to-head** (RQ6) — matched frame, loss, weights, seeds, plus the shared
   `none` baseline. This is the thesis's central mechanistic claim: whether the in-graph
   differentiable formulation buys the same ordering as explicit projection. Requires
   `level_softmax_ce_reg` + identity frame on both arms, which the existing runs partly cover.
3. **Run `pairwise_orthogonal`** — without the order-free control, RQ4's central claim is
   not identified.
4. **HCC on Hier-COS**, starting with `final_test_only` — cheapest decisive arm, and RQ7
   currently has no Hier-COS data at all.
5. **Lex on HRN and HT-CapsNet** — turns lex from a two-model observation into a
   cross-architecture claim.
6. **Plugin frame ladder on one of H-CAST / HRN / HT-CapsNet** — tests whether RQ1
   generalizes. Not LH-DNN (C.0).
7. **`kl_coarse` arm** — falsification test for the consistency claim in RQ3.
8. **`advantage_enabled`, `conflict_only` on remaining datasets** — completeness.

## C.5 Reporting rules to hold across the whole matrix

- Top-down rows use the top-down-selected checkpoint; independent rows use the
  independent-selected checkpoint. Never mix.
- FPA, weighted AP, accuracy: higher is better. AHD, TICE: lower is better.
- Percentage deltas in percentage points.
- Some `run_log.jsonl` files truncate early despite a finished run (memory:
  `truncated-run-logs`) — filter before averaging epoch curves.
- Confirm HCC activation from `proj_constraint_alpha`, never from a directory name.
