# HCC Research Report

Date: 2026-04-28

This report analyzes the H-CAST Hierarchical Constraint Cascade (HCC) runs in
`/scratch/g.saggini1/outputs` across CIFAR-100, CUB-200, and FGVC-Aircraft.
It uses the existing notebooks, resolved configs, `run_log.jsonl`,
`test_metrics.yaml`, HCC source code, and gradient diagnostics. No training was
rerun.

Primary artifacts:

- `notebooks/hcast_analysis.ipynb`
- `notebooks/hcc_internal_diagnostics.ipynb`
- `models/hcast/model.py`
- `models/hcast/hard_hierarchy.py`
- `models/hcast/losses.py`
- `train/eval.py`
- `train/metrics.py`
- `train/trunk_metrics.py`
- `docs/HCC_DIAGNOSTIC_LOGS.md`
- `docs/GRADIENT_PARAM_DIAGNOSTIC_LOGS.md`

## Executive Summary

HCC is not only an evaluation-time consistency fix. When active, it changes both
the scores used by metrics and the objective used by training.

The main finding is:

> In the earlier probability-space implementation, HCC behaved like a
> parent-budget clamp on child distributions. The current implementation
> projects logits directly, so new logs should be read through the
> `proj_logit_*` diagnostics.

The evidence is strongest in the `step_80epoch` runs:

- CIFAR-100 has exactly 5 fine leaves per middle parent. When HCC turns on, FPA
  independent improves from 72.21% to 74.10%, and TICE independent drops from
  8.45% to 3.15%.
- CUB-200 has middle-parent fan-out from 1 to 30 leaves. When HCC turns on, FPA
  independent collapses from 79.91% to 19.03%, but FPA top-down stays at
  80.89%.
- Aircraft has many singleton middle parents. When HCC turns on, FPA
  independent drops from 67.82% to 50.25%, while FPA top-down rises from
  71.48% to 73.02%.

The independent/top-down split is the key diagnostic:

- Independent decoding uses a global argmax at each level. At L2, every fine
  class competes with every other fine class.
- Top-down decoding first chooses the parent and then masks children to that
  parent. It removes cross-parent fine-class competition.

Therefore, CUB and Aircraft can have poor independent metrics while top-down
metrics remain strong. The correct leaf is often still recoverable inside the
right parent; it is losing the global L2 competition after HCC rescales subtrees.

The gradient diagnostics add a second, training-side explanation:

- Active HCC sharply reduces the fine-level gradient norm on the shared `t1`
  trunk.
- Active HCC also lowers the cosine alignment between level losses, especially
  `cos_fine_mid`.
- This happens on all three datasets.

At epoch 100, baseline vs `step0` `grad_norm_t1_fine` is:

| Dataset | Baseline | HCC step0 | Ratio |
| --- | ---: | ---: | ---: |
| CIFAR-100 | 6.392 | 1.311 | 0.21x |
| CUB-200 | 6.810 | 0.806 | 0.12x |
| Aircraft | 5.161 | 0.403 | 0.08x |

This means HCC from epoch 0 is not just changing the final probabilities. It is
also training the network under a much weaker and differently aligned fine-level
signal from the beginning.

## What HCC Actually Does

### Forward path

`HCASTModel.forward` returns raw `logits_per_level` for all runs. When HCC is
enabled and active, it also returns `effective_logits_per_level`.

The important branch is:

- If `alpha <= eps`, HCC emits diagnostics but leaves `effective_logits_per_level`
  absent. Losses and metrics use raw logits, exactly like baseline H-CAST.
- If `alpha > eps`, HCC projects per-level logits, blends raw/projected logits
  with `alpha`, and returns `effective_logits_per_level`.

In the step-schedule runs, the transition is hard:

- inactive: `alpha=0`, `temperature=10`
- active: `alpha=1`, `temperature=1`

Because `train/train.py` calls the model with zero-based `epoch` but logs
`epoch + 1`, a configured `alpha_start_epoch: 80` appears in the logs at
displayed epoch 81. This matters for the inverse-step spike.

### Projection path

`HierarchicalAffineProjector` enforces the three-level constraints:

```text
z1 = M12 @ z2
z2 = M23 @ z3
```

The implementation is stage-wise:

- `z1` is kept unchanged.
- `z2` is projected against a detached `z1` anchor.
- `z3` is projected against a detached `z2_hat` anchor.
- No simplex mass renormalization is applied in the logit-space projector.

The resulting fine probabilities are obtained by applying softmax to the final
projected logits. They are diagnostics/evaluation scores, not the constrained
object itself.

### Loss path

`models/hcast/losses.py` switches score source based on whether
`effective_logits_per_level` is present:

- HCC inactive or baseline: raw logits go into `cross_entropy`.
- HCC active: projected/blended logits go into `cross_entropy`.

So active-HCC loss is still a logit cross-entropy objective, but over the
projected logits rather than the raw unconstrained logits.

### Metrics path

`train/eval.py` makes the same switch:

- HCC inactive or baseline: metrics use raw logits.
- HCC active: metrics use `effective_logits_per_level`.

This means validation curves around a hard HCC switch contain a logit-source
discontinuity. At the switch, the model is evaluated under a projected logit
definition.

### Decoding path

`train/metrics.py` implements two decoders:

- Independent: argmax at each level independently.
- Top-down: argmax at L0, then mask L1 to children of L0, then mask L2 to
  children of L1.

Top-down TICE is always 0 when taxonomy mappings are complete, because the
decoder constructs a valid path. Independent TICE can increase under HCC because
sum-consistent probabilities do not guarantee that independent argmax choices
form a valid path.

## Tree Balance

The hierarchy imbalance is not incidental. In the earlier probability-space
implementation, it directly interacted with HCC's parent-budget rescaling.

| Dataset | Middle nodes | Leaves | Min leaves/parent | Mean | Max | Singleton parents | Fan-out histogram |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CIFAR-100 | 20 | 100 | 5 | 5.00 | 5 | 0 | 5:20 |
| CUB-200 | 38 | 200 | 1 | 5.26 | 30 | 12 | 1:12, 2:4, 3:3, 4:5, 5:2, 6:2, 7:3, 8:2, 11:1, 13:1, 15:1, 23:1, 30:1 |
| Aircraft | 70 | 100 | 1 | 1.43 | 8 | 54 | 1:54, 2:10, 3:2, 4:3, 8:1 |

CIFAR is perfectly balanced at L1 to L2. CUB and Aircraft are not. CUB has
families with 1, 23, and 30 species. Aircraft has 54 singleton families out of
70.

The bias is:

- Small or singleton parents can produce very high leaf probabilities after HCC.
- Large parents must distribute their budget across many leaves.
- Independent fine decoding compares all leaves globally, so it is exposed to
  this fan-out bias.
- Top-down decoding compares only siblings inside the selected parent, so it is
  much less exposed.

## Final Test Results

These are final `test_metrics` from the selected runs.

| Dataset | Run | FPA ind | FPA TD | L2 ind | L2 TD | wAP ind | wAP TD | TICE ind | AHD ind |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 | baseline | 72.39% | 74.28% | 75.20% | 74.28% | 77.63% | 76.91% | 8.32% | 1.059 |
| CIFAR-100 | step0 | 61.62% | 62.30% | 62.98% | 62.30% | 67.77% | 67.26% | 4.32% | 1.323 |
| CIFAR-100 | step0 noKL | 61.13% | 61.84% | 62.09% | 61.84% | 67.12% | 66.93% | 3.61% | 1.321 |
| CIFAR-100 | step80 | 73.57% | 74.23% | 74.55% | 74.23% | 77.10% | 76.87% | 3.14% | 1.039 |
| CIFAR-100 | inverse80 | 65.71% | 70.07% | 69.19% | 70.07% | 72.81% | 73.43% | 14.86% | 1.223 |
| CUB-200 | baseline | 79.27% | 80.96% | 81.48% | 80.96% | 84.17% | 83.73% | 6.10% | 0.587 |
| CUB-200 | step0 | 12.77% | 42.67% | 14.42% | 42.67% | 28.54% | 51.02% | 83.47% | 2.210 |
| CUB-200 | step0 noKL | 11.96% | 42.33% | 13.79% | 42.33% | 27.96% | 50.66% | 84.26% | 2.235 |
| CUB-200 | step80 | 19.68% | 81.83% | 20.97% | 81.83% | 36.04% | 84.52% | 78.01% | 1.767 |
| CUB-200 | inverse80 | 54.32% | 62.12% | 58.35% | 62.12% | 64.78% | 67.79% | 22.21% | 1.215 |
| CUB-200 | step0 condL2 | 12.35% | 40.50% | 14.23% | 40.50% | 28.00% | 48.88% | 84.14% | 2.274 |
| Aircraft | baseline | 69.41% | 72.31% | 74.02% | 72.31% | 78.62% | 77.23% | 13.32% | 1.282 |
| Aircraft | step0 | 37.57% | 54.20% | 39.78% | 54.20% | 55.76% | 62.89% | 46.76% | 2.237 |
| Aircraft | step0 noKL | 37.66% | 50.46% | 39.97% | 50.46% | 55.00% | 60.10% | 48.47% | 2.282 |
| Aircraft | step80 | 52.97% | 71.86% | 57.42% | 71.86% | 70.18% | 76.90% | 38.46% | 1.616 |
| Aircraft | inverse80 | 42.61% | 65.26% | 49.09% | 65.26% | 63.57% | 71.99% | 43.27% | 1.958 |

All three datasets show a large degradation when HCC is active from epoch 0.
The size and shape differ:

- CIFAR step0 loses about 11 to 12 pp in FPA.
- CUB step0 loses 66.50 pp FPA independent and 38.28 pp FPA top-down.
- Aircraft step0 loses 31.84 pp FPA independent and 18.11 pp FPA top-down.

No-KL runs do not recover the loss. Therefore the global KL term is not the
main cause.

CUB `step0 condL2` lowers the numeric L2 loss by changing the loss definition to
conditional `log p(child | parent)`, but final FPA remains almost identical to
bad step0. This is important: reducing the reported loss scale does not fix the
decoding failure.

## Switch Evidence

The `step80` runs isolate the evaluation-time effect of activating HCC after a
baseline-like model has already learned.

| Dataset | Epoch | alpha | FPA ind | FPA TD | TICE ind | flip L2 | cond ind | cond TD | pre mass | post mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 step80 | 80 | 0.0 | 72.21% | 74.33% | 8.45% | 0.00% | 85.78% | 88.06% | 0.581 | 0.581 |
| CIFAR-100 step80 | 81 | 1.0 | 74.10% | 74.79% | 3.15% | 6.92% | 87.98% | 88.29% | 0.779 | 0.374 |
| CIFAR-100 step80 | 100 | 1.0 | 74.38% | 74.88% | 2.95% | 6.78% | 88.04% | 88.30% | 0.823 | 0.379 |
| CUB-200 step80 | 80 | 0.0 | 79.91% | 81.04% | 5.58% | 0.00% | 85.07% | 86.11% | 0.619 | 0.619 |
| CUB-200 step80 | 81 | 1.0 | 19.03% | 80.89% | 79.23% | 78.52% | 20.31% | 85.60% | 0.703 | 0.232 |
| CUB-200 step80 | 100 | 1.0 | 20.66% | 81.16% | 77.51% | 77.05% | 21.99% | 85.82% | 0.888 | 0.229 |
| Aircraft step80 | 80 | 0.0 | 67.82% | 71.48% | 14.49% | 0.00% | 84.60% | 90.21% | 0.458 | 0.458 |
| Aircraft step80 | 81 | 1.0 | 50.25% | 73.02% | 41.13% | 41.94% | 62.20% | 90.16% | 0.515 | 0.326 |
| Aircraft step80 | 100 | 1.0 | 55.83% | 73.94% | 34.99% | 38.20% | 68.57% | 90.55% | 0.661 | 0.338 |

Reading:

- CIFAR: HCC flips few fine predictions and improves independent consistency.
- CUB: HCC flips almost 80% of fine predictions. Independent FPA collapses, but
  top-down FPA stays at baseline quality.
- Aircraft: same failure mode as CUB, but smaller.

The conditional accuracies are decisive. In CUB at epoch 81:

- `acc_l2_ind_given_l1_correct`: 20.31%
- `acc_l2_td_given_l1_correct`: 85.60%

So the leaf is still usually recoverable inside the selected top-down parent.
The independent global L2 argmax is the failing piece.

## Projection Residuals And Within-Parent Rank

HCC is numerically enforcing the sum constraints. The issue is not that the
projector fails to project.

| Run | Epoch | residual before | residual after | reduction | flip L2 | GT delta L2 | rank pre | rank post |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 step80 | 81 | 0.0181 | 0.0000 | 0.0181 | 6.92% | -0.5541 | 1.21 | 1.21 |
| CUB-200 step80 | 81 | 0.0094 | 0.0000 | 0.0094 | 78.52% | -0.4712 | 1.37 | 1.37 |
| Aircraft step80 | 81 | 0.0057 | 0.0000 | 0.0057 | 41.94% | -0.1735 | 1.13 | 1.13 |
| CIFAR-100 step0 | 100 | 0.0380 | 0.0000 | 0.0380 | 21.15% | -0.5034 | 1.59 | 1.58 |
| CUB-200 step0 | 100 | 0.0290 | 0.0000 | 0.0290 | 85.83% | -0.3809 | 3.95 | 3.87 |
| Aircraft step0 | 100 | 0.0208 | 0.0000 | 0.0208 | 93.44% | 0.0867 | 1.28 | 1.27 |

Within-parent rank barely changes in the hard-switch rows. In the earlier
probability-space implementation, this matched `_mass_renormalize` mostly
applying a common scale to children under the same parent.

Therefore the main failure is not sibling reordering. It is cross-parent
competition after projection.

## Why HCC From Epoch 0 Degrades Results

HCC from epoch 0 degrades all three datasets in the selected logs, but the
mechanism has two layers.

### 1. Early projected parent signals are unreliable

At epoch 0, the parent heads are not trained. With HCC active, child logits are
immediately projected against weak parent logits.

This makes the fine objective depend on the projected logit path from the first
minibatch, rather than only on the raw L2 head.

Because the projector uses detached parent anchors, the fine loss does not
directly repair the middle parent signal through the HCC path. The fine head can
learn within-parent ordering, but it cannot by itself force the middle head to
allocate more probability mass to that parent.

### 2. Active HCC changes the gradient geometry

The training logs show that active HCC strongly attenuates the fine-level
gradient and reduces gradient alignment between hierarchy levels.

Epoch 100 gradient/cosine summary:

| Dataset | Run | cos mid/coarse | cos fine/coarse | cos fine/mid | grad t1 fine |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | baseline | 0.744 | 0.476 | 0.637 | 6.392 |
| CIFAR-100 | step0 | 0.158 | 0.128 | 0.137 | 1.311 |
| CIFAR-100 | step80 | 0.269 | 0.247 | 0.336 | 1.137 |
| CUB-200 | baseline | 0.426 | 0.140 | 0.478 | 6.810 |
| CUB-200 | step0 | 0.024 | -0.023 | 0.007 | 0.806 |
| CUB-200 | step80 | 0.057 | 0.027 | 0.190 | 0.849 |
| Aircraft | baseline | 0.697 | 0.396 | 0.536 | 5.161 |
| Aircraft | step0 | 0.005 | 0.038 | 0.134 | 0.403 |
| Aircraft | step80 | 0.305 | 0.047 | 0.298 | 0.456 |

The pattern is all-dataset:

- `grad_norm_t1_fine` is much lower under active HCC.
- `cos_fine_mid` and `cos_fine_coarse` are much lower under active HCC.
- CUB step0 even has slightly negative `cos_fine_coarse` at epoch 100.

This says HCC makes the fine loss less able to reinforce the lower-level
representation in the same way as baseline H-CAST. It is not merely an
evaluation-time post-processing effect.

### 3. The tree decides how visible the damage is

On CIFAR, the balanced tree avoids the severe independent decoding bias. HCC
from epoch 0 still hurts because the model is trained under the constrained
objective too early, but later activation can help.

On CUB and Aircraft, HCC also creates a structural global-argmax bias:

- CUB: families range from 1 to 30 species.
- Aircraft: 54 families are singletons.

This makes the independent L2 argmax biased toward leaves under small parents.
Top-down decoding avoids most of this because it does not compare leaves across
different parents after selecting the parent.

## Why Active-HCC Loss Is Higher Than Baseline

The statement is true for late training in all three main datasets, but not
literally for every logged scalar at every epoch. For example, CIFAR step0 epoch
1 total loss is lower than baseline because the global KL part is lower. The
important stable pattern is that once training reaches the useful regime, active
HCC has higher `level_ce` and total loss than baseline.

Total training loss at selected epochs:

| Dataset | Run | epoch 1 | epoch 79 | epoch 80 | epoch 81 | epoch 100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 | baseline | 11.125 | 6.058 | 6.165 | 6.013 | 5.919 |
| CIFAR-100 | step0 | 10.841 | 7.485 | 7.560 | 7.445 | 7.360 |
| CIFAR-100 | step80 | 11.125 | 6.058 | 6.165 | 7.402 | 7.252 |
| CIFAR-100 | inverse80 | 10.841 | 7.485 | 7.560 | 8.605 | 6.223 |
| CUB-200 | baseline | 13.512 | 5.500 | 5.266 | 5.490 | 5.301 |
| CUB-200 | step0 | 15.939 | 9.093 | 8.968 | 9.108 | 9.021 |
| CUB-200 | step80 | 13.512 | 5.500 | 5.266 | 9.084 | 8.838 |
| CUB-200 | inverse80 | 15.939 | 9.093 | 8.968 | 14.838 | 6.994 |
| Aircraft | baseline | 13.392 | 5.177 | 6.109 | 6.022 | 6.030 |
| Aircraft | step0 | 13.669 | 6.942 | 7.690 | 7.598 | 7.606 |
| Aircraft | step80 | 13.392 | 5.177 | 6.109 | 7.503 | 7.439 |
| Aircraft | inverse80 | 13.669 | 6.942 | 7.690 | 15.182 | 7.109 |

At epoch 100, baseline vs active-HCC `level_ce`:

| Dataset | baseline | step0 | step0 noKL | step80 |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | 5.495 | 6.733 | 6.736 | 6.641 |
| CUB-200 | 4.981 | 8.187 | 8.189 | 8.030 |
| Aircraft | 5.654 | 7.021 | 7.057 | 6.878 |

No-KL stays high. Therefore the higher loss is not mainly from the global KL
term.

The reason is the score source:

- Baseline CE can increase the raw fine logit for the correct fine class
  independently of parent probability calibration.
- Active-HCC CE sees the projected logit. If projection lowers the correct fine
  logit, the softmax probability can be low even when its within-parent rank is
  good.
- HCC also attenuates fine gradients, so the model has less fine-level leverage
  to repair the representation.

This is why the active-HCC loss scale jumps as soon as `step80` turns on:

- CIFAR: 6.165 at epoch 80 to 7.402 at epoch 81.
- CUB: 5.266 to 9.084.
- Aircraft: 6.109 to 7.503.

The loss did not suddenly discover a worse model; the loss started reading a
different score source.

## Why Step0 And Step80 Converge To Similar Loss But Different FPA

By epoch 100, `step0` and `step80` active-HCC losses are close:

| Dataset | step0 total | step80 total | step0 FPA ind | step80 FPA ind | step0 FPA TD | step80 FPA TD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 | 7.360 | 7.252 | 61.62% | 73.57% | 62.30% | 74.23% |
| CUB-200 | 9.021 | 8.838 | 12.77% | 19.68% | 42.67% | 81.83% |
| Aircraft | 7.606 | 7.439 | 37.57% | 52.97% | 54.20% | 71.86% |

This is not a contradiction.

The loss is a projected-logit objective under HCC. FPA is an argmax path
metric. Two models can have similar constrained cross-entropy but very different
ranking behavior, especially near decision boundaries and especially under
unbalanced parent fan-out.

The histories are different:

- `step0` learns the whole representation under HCC from the first epoch. Its
  fine gradients are attenuated throughout training, and level-gradient cosine
  alignment stays weak.
- `step80` first learns a baseline-like representation using raw logits. When
  HCC turns on, the score source changes, but the model already has useful
  parent and within-parent leaf structure.

CUB makes this clearest. At `step80` epoch 100:

- independent FPA is only 20.66%;
- top-down FPA is 81.16%;
- `acc_l2_td_given_l1_correct` is 85.82%;
- within-parent GT rank is unchanged at 1.37 before and after projection.

So `step80` still contains a strong top-down classifier. The active-HCC loss
does not reveal that directly because it is not a top-down path metric. It is a
sum of level losses on projected logits.

## Why Inverse-Step Deactivation Spikes

The inverse-step logs are resume logs. They contain an HCC-active prelude
through displayed epoch 80, then the resumed run switches to a config with
`alpha_start_epoch: 999`. Because the loop logs `epoch + 1`, deactivation appears
at displayed epoch 81.

At deactivation:

- `effective_logits_per_level` disappears.
- The loss immediately switches from projected logits back to raw logits.
- Metrics also switch back to raw logits.
- The model has been trained under HCC, so the raw logits are not calibrated for
  direct use.

The loss spike and gradient spike happen together:

| Dataset | epoch | alpha | train total | FPA ind | FPA TD | cos fine/mid | grad t1 fine | delta t1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR inverse80 | 80 | 1.0 | 7.560 | 59.87% | 60.51% | 0.105 | 1.265 | 1.504 |
| CIFAR inverse80 | 81 | 0.0 | 8.605 | 55.03% | 63.07% | 0.410 | 6.511 | 4.841 |
| CUB inverse80 | 80 | 1.0 | 8.968 | 13.71% | 43.23% | 0.040 | 1.037 | 0.357 |
| CUB inverse80 | 81 | 0.0 | 14.838 | 16.14% | 30.80% | 0.233 | 16.887 | 3.035 |
| Aircraft inverse80 | 80 | 1.0 | 7.690 | 39.40% | 55.47% | 0.129 | 0.341 | 0.540 |
| Aircraft inverse80 | 81 | 0.0 | 15.182 | 5.13% | 42.66% | 0.352 | 11.899 | 2.975 |

This is the cleanest evidence that HCC changes optimization, not just
post-processing:

- Fine gradients jump by 5x on CIFAR, 16x on CUB, and 35x on Aircraft.
- Parameter movement jumps in the same epoch.
- The raw-logit objective is suddenly much harder for a model trained under the
  projected-logit objective.

The spike is therefore expected. It is an objective and score-source
discontinuity.

## Dataset-Specific Findings

### CIFAR-100

CIFAR is the only balanced tree in the analysis. HCC can improve independent
path consistency without imposing a strong fan-out bias.

At `step80` activation:

- FPA independent: 72.21% to 74.10%.
- TICE independent: 8.45% to 3.15%.
- L2 flip rate: only 6.92%.
- Top-down FPA is stable.

The main negative result is `step0`: HCC from epoch 0 loses about 12 pp top-down
FPA. The gradient table explains why: the fine-level gradient is much weaker
under active HCC, and level-gradient alignment drops sharply. Balanced fan-out
protects against the CUB-style global argmax collapse, but it does not remove
the early-training optimization damage.

### CUB-200

CUB is the strongest failure case.

At `step80` activation:

- FPA independent: 79.91% to 19.03%.
- FPA top-down: 81.04% to 80.89%.
- TICE independent: 5.58% to 79.23%.
- L2 flip rate: 78.52%.
- `acc_l2_ind_given_l1_correct`: 20.31%.
- `acc_l2_td_given_l1_correct`: 85.60%.

This says:

- The parent path remains usable.
- The correct child remains recoverable inside the parent.
- Independent global L2 argmax is the failing operation.

The tree explains why: CUB has 12 singleton families and one 30-species family.
HCC makes leaves under small parents much more competitive globally than leaves
under large parents.

Step0 is worse because the model never gets the baseline raw-logit learning
phase. The CUB `step0 condL2` ablation reduces the numerical L2 loss but does
not restore FPA, so the problem is not just that marginal fine CE is a bad loss
scale. The ranking/decoding geometry remains bad.

### FGVC-Aircraft

Aircraft has the same failure mode as CUB, but it is less extreme for top-down
metrics.

At `step80` activation:

- FPA independent: 67.82% to 50.25%.
- FPA top-down: 71.48% to 73.02%.
- TICE independent: 14.49% to 41.13%.
- L2 flip rate: 41.94%.
- `acc_l2_td_given_l1_correct`: 90.16%.

Aircraft has 54 singleton families out of 70. Under the earlier
probability-space interpretation, singleton parents were especially favored
because parent budget was not split among siblings.

The inverse deactivation spike is largest on Aircraft:

- total loss: 7.690 to 15.182;
- FPA independent: 39.40% to 5.13%;
- `grad_norm_t1_fine`: 0.341 to 11.899.

This suggests the raw logits are especially poorly calibrated after being
trained under the active-HCC objective.

## Answers To The Main Questions

### Why does HCC from epoch 0 degrade results so much?

Because it starts training with a projected-logit objective before the parent
heads are reliable. The fine loss is trained through the logit projection from
the first epoch, which changes the gradient path relative to baseline H-CAST.

Evidence across all datasets:

- Step0 test metrics are worse than baseline on CIFAR, CUB, and Aircraft.
- Step0 no-KL is still bad on all three, so global KL is not the cause.
- Fine-level gradient norms are much lower under active HCC on all three.
- Gradient cosines between fine/mid/coarse losses drop on all three.

Dataset-specific amplification:

- CIFAR: balanced tree, so the degradation is mostly optimization-side.
- CUB/Aircraft: optimization damage plus unbalanced fan-out bias in independent
  fine decoding.

### Why is loss with HCC higher than baseline H-CAST?

Because active HCC changes the loss score source. The model is no longer
optimizing raw logits with `cross_entropy`; it is optimizing projected logits
after projection/blending.

If projection lowers the correct fine logit, the correct fine probability can be
low even when the fine leaf is well-ranked inside that parent.

All three datasets show higher late-training active-HCC `level_ce` than
baseline. No-KL runs stay high, so the global KL term is not the main cause.

Important nuance: this is not literally true for every scalar at every epoch.
CIFAR step0 epoch 1 total loss is lower than baseline because its GK component
is lower. The robust claim is that active-HCC late-training `level_ce` and total
loss are higher than baseline on all three datasets.

### Why do step0 and step80 converge on similar loss even if FPA is different?

Because once HCC is active, both runs are measured under the same constrained
loss scale. Similar cross-entropy under projected logits does not imply
similar argmax path behavior.

The training histories are different:

- Step0 learns the representation under weak fine gradients from the beginning.
- Step80 learns a baseline-like representation first, then applies HCC.

This is why CUB step80 has poor independent FPA but excellent top-down FPA,
while CUB step0 is poor in both modes. The late losses are close, but the
ranking structure learned before the switch is not the same.

### In inverse-step 80, why is there a spike when HCC is deactivated?

The spike is at displayed epoch 81, which corresponds to internal epoch 80.
That is the first epoch where the resumed inverse config has `alpha=0`.

At that point, the loss and metrics switch from projected logits back to
raw logits. The raw logits have been trained under the projected objective and
are not ready to be used directly. This creates an immediate loss spike and a
large gradient spike, especially in the fine-level trunk gradient.

This happens on all three datasets, with the largest spike on Aircraft.

## Practical Interpretation

Current hard HCC is useful as a diagnostic and sometimes as a late-stage
top-down consistency mechanism. It is risky as an epoch-0 training objective.

The safest interpretation of the current logs is:

- HCC enforces probability-sum constraints correctly.
- HCC does not guarantee independent argmax path consistency.
- HCC can strongly damage global independent fine decoding in unbalanced trees.
- Top-down decoding is the right evaluation mode for active HCC on CUB and
  Aircraft.
- If independent fine predictions matter, hard HCC should not be used naively
  on unbalanced taxonomies.

## Recommendations

1. Do not use hard HCC from epoch 0 as the default training recipe.

2. Report independent and top-down metrics separately. For CUB and Aircraft,
   top-down metrics reveal that useful hierarchical classifiers still exist
   even when independent FPA collapses.

3. Treat HCC loss curves as a different loss scale from baseline H-CAST. Do not
   compare active-HCC CE to baseline CE without noting that active HCC trains on
   projected logits.

4. For unbalanced trees, consider fan-out-aware calibration before global
   independent decoding. Examples:
   - decode top-down only;
   - compare conditional probabilities inside the selected parent;
   - normalize fine scores by parent fan-out for independent analysis;
   - use a softer/ramped HCC schedule rather than a hard step.

   Training-loss fan-out correction has been removed from the code path.

5. If HCC remains in training, monitor gradient diagnostics:
   - `grad_norm_t1_fine`;
   - `cos_fine_mid`;
   - `cos_fine_coarse`;
   - `delta_param_norm_t1`.

   These metrics expose the objective discontinuity more directly than FPA
   alone.

## Verification Notes

- Final metrics were read from `test_metrics` events in `run_log.jsonl`.
- Validation switch tables were read from `val_metrics` in `run_log.jsonl`.
- Loss and gradient tables were read from `train_losses` and `train_metrics` in
  `run_log.jsonl`.
- Schedule interpretation was cross-checked against `config_resolved.yaml`.
- The epoch offset was cross-checked against `train/train.py`: the model receives
  zero-based `epoch`, while the logger writes `epoch + 1`.
- Fan-out was computed from `datasets/cub_tree.py`, `datasets/aircraft_tree.py`,
  and the known CIFAR-100 20-by-5 hierarchy.
- Gradient key semantics were cross-checked against
  `docs/GRADIENT_PARAM_DIAGNOSTIC_LOGS.md` and `train/trunk_metrics.py`.
