# HCC/H-CAST Research Report

Date: 2026-05-12

> **Status note (2026-08-06).** HCC is now a binary on/off switch in this
> codebase: the `alpha_schedule` / `alpha_start_epoch` / `temperature` knobs were
> removed, so the delayed arm (`step@80`) is no longer reproducible from a
> current config. The `step@0` arm is numerically identical to today's
> `hcc.enabled: true`. Findings below are retained as a record of the runs that
> were executed while the schedule existed; the delayed-HCC recommendation no
> longer corresponds to a supported setting.

This report analyzes whether, where, and why HCC helps H-CAST. It uses only the uncommented runs in `notebooks/model_analysis/hcast_analysis.ipynb`: CIFAR-100, CUB-200-2011, and FGVC-Aircraft, each with H-CAST, HCAST no-KL, HCC step@80, HCC step@80 no-KL, HCC step@0, and HCC step@0 no-KL.

The `hcc_internal_diagnostics` notebook is used only as a glossary for metric interpretation. Numeric HCC conclusions below come from the scoped run logs and `test_metrics.yaml` files under `/scratch/g.saggini1/outputs`.

Auxiliary `hcast_lex_*` runs are used only in the lexicographic-optimization section as a reference signature for explicit gradient projection. They are not treated as part of the main HCC benchmark.

## Executive Answer

HCC works, but not as a uniform accuracy booster. It is best understood as a hard hierarchy-consistency intervention that changes the effective logits, the optimization geometry, and the decoder-dependent tradeoff between coherent paths and independent fine-class ranking.

The strongest positive evidence is:

- FGVC-Aircraft with delayed HCC (`step@80`) improves H-CAST clearly: top-down FPA +1.65 pp, independent FPA +1.87 pp, top-down AHD -0.090, independent AHD -0.093, and independent TICE -2.60 pp.
- CUB-200 benefits mainly in top-down decoding: delayed HCC improves top-down FPA +0.46 pp versus H-CAST, and the no-KL matched comparison improves +0.61 pp. Independent fine behavior does not improve.
- CIFAR-100 benefits most from HCC active from the start with KL: `step@0` improves top-down FPA +0.73 pp and independent FPA +0.32 pp versus H-CAST. Matched no-KL HCC, however, does not beat the no-KL baseline.

The strongest negative evidence is:

- Early hard HCC is harmful on Aircraft top-down metrics and harmful on CUB when KL is enabled.
- Delayed HCC often improves top-down metrics while degrading or leaving independent metrics unchanged, especially on CUB.
- HCC reduces hierarchy residuals to zero, but this does not guarantee better fine discrimination. In several late-HCC windows it lowers the ground-truth fine probability relative to raw logits.

The practical conclusion is: HCC helps when hierarchy-enforced decoding is the desired operating mode and the constraint is introduced at a time compatible with the dataset. The best default from these runs is delayed HCC for Aircraft and CUB top-down behavior, and early HCC only looks attractive on CIFAR-100 with KL.

From a lexicographic optimization perspective, HCC is not an explicit lexicographic optimizer. True lexicographic training projects lower-priority gradients and drives post-projection cosines near zero. HCC has no post-gradient projection, but it often makes the raw gradients look more lexicographic: fine-level gradient norms shrink and fine/higher-level cosine alignment drops. So HCC is better described as an output-space hard-affine hierarchy constraint that implicitly changes the gradient field in a lex-like direction.

## Run Scope And Verification

All 18 scoped runs were found, each with 100 logged epochs. HCC activation was verified from `proj_constraint_alpha`, not inferred from directory names:

- `step@0`: active from logged epoch 1.
- `step@80`: active from logged epoch 81.

Final test values were recomputed from `test_metrics.yaml` using the same mode-specific checkpoint rule used by `hcast_analysis`: top-down rows use the top-down-selected checkpoint, and independent rows use the independent-selected checkpoint. This is necessary because several runs have different best epochs for top-down and independent selection.

## Final Test Behavior

Values are final test metrics using mode-specific best checkpoints. For FPA, wAP, and fine accuracy higher is better. For AHD and TICE lower is better.

| Dataset | Run | Best TD/Ind epoch | FPA TD | FPA Ind | wAP TD | wAP Ind | AHD TD | AHD Ind | TICE Ind | Fine TD | Fine Ind |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar100 | H-CAST | 100/100 | 74.13% | 72.29% | 76.74% | 77.23% | 1.021 | 1.066 | 8.04% | 74.13% | 74.72% |
| cifar100 | HCAST no-KL | 93/93 | 74.40% | 72.29% | 77.06% | 77.48% | 0.999 | 1.053 | 8.69% | 74.40% | 74.96% |
| cifar100 | HCC step@80 | 98/88 | 74.41% | 72.30% | 77.00% | 77.57% | 1.013 | 1.054 | 9.52% | 74.41% | 75.06% |
| cifar100 | HCC step@80 no-KL | 99/73 | 74.22% | 71.45% | 76.85% | 76.84% | 1.014 | 1.087 | 8.73% | 74.22% | 74.25% |
| cifar100 | HCC step@0 | 93/90 | 74.86% | 72.62% | 77.40% | 77.69% | 0.994 | 1.070 | 8.78% | 74.86% | 75.38% |
| cifar100 | HCC step@0 no-KL | 100/100 | 74.38% | 72.17% | 76.99% | 77.38% | 1.009 | 1.070 | 8.93% | 74.38% | 74.92% |
| cub200 | H-CAST | 99/97 | 80.96% | 79.44% | 83.73% | 84.21% | 0.550 | 0.582 | 5.85% | 80.96% | 81.51% |
| cub200 | HCAST no-KL | 99/73 | 80.94% | 79.04% | 83.75% | 84.02% | 0.544 | 0.594 | 6.35% | 80.94% | 81.31% |
| cub200 | HCC step@80 | 97/75 | 81.41% | 78.84% | 84.17% | 83.70% | 0.530 | 0.599 | 6.05% | 81.41% | 80.91% |
| cub200 | HCC step@80 no-KL | 97/73 | 81.55% | 79.04% | 84.28% | 84.02% | 0.527 | 0.594 | 6.35% | 81.55% | 81.31% |
| cub200 | HCC step@0 | 44/92 | 79.65% | 79.38% | 82.59% | 84.18% | 0.589 | 0.573 | 6.86% | 79.65% | 81.41% |
| cub200 | HCC step@0 no-KL | 97/97 | 81.26% | 79.14% | 84.04% | 83.69% | 0.534 | 0.579 | 6.57% | 81.26% | 80.82% |
| aircraft | H-CAST | 97/95 | 72.31% | 69.24% | 77.23% | 78.55% | 1.208 | 1.289 | 13.68% | 72.31% | 73.93% |
| aircraft | HCAST no-KL | 90/90 | 73.32% | 68.43% | 78.39% | 77.93% | 1.139 | 1.258 | 14.69% | 73.32% | 72.23% |
| aircraft | HCC step@80 | 96/96 | 73.96% | 71.12% | 78.79% | 78.49% | 1.118 | 1.196 | 11.08% | 73.96% | 73.43% |
| aircraft | HCC step@80 no-KL | 85/96 | 73.82% | 70.73% | 78.72% | 78.55% | 1.124 | 1.207 | 11.05% | 73.82% | 73.38% |
| aircraft | HCC step@0 | 74/96 | 71.26% | 68.74% | 76.65% | 76.62% | 1.231 | 1.294 | 12.06% | 71.26% | 71.20% |
| aircraft | HCC step@0 no-KL | 96/96 | 72.59% | 69.83% | 77.82% | 77.45% | 1.171 | 1.250 | 9.46% | 72.59% | 72.04% |

## Paired Effects

Deltas are relative to the comparison baseline. FPA, wAP, fine accuracy, and TICE are in percentage points. AHD is absolute.

| Dataset | Comparison | FPA TD | FPA Ind | wAP TD | wAP Ind | Fine TD | Fine Ind | AHD TD | AHD Ind | TICE Ind |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar100 | step@80 vs H-CAST | +0.28 | +0.01 | +0.25 | +0.34 | +0.28 | +0.34 | -0.009 | -0.012 | +1.48 |
| cifar100 | step@0 vs H-CAST | +0.73 | +0.32 | +0.66 | +0.46 | +0.73 | +0.66 | -0.027 | +0.004 | +0.74 |
| cifar100 | step@80 no-KL vs no-KL | -0.19 | -0.84 | -0.21 | -0.64 | -0.19 | -0.71 | +0.015 | +0.033 | +0.04 |
| cifar100 | step@0 no-KL vs no-KL | -0.03 | -0.12 | -0.07 | -0.10 | -0.03 | -0.04 | +0.011 | +0.017 | +0.23 |
| cifar100 | step@0 - step@80 | +0.45 | +0.31 | +0.41 | +0.12 | +0.45 | +0.32 | -0.018 | +0.016 | -0.74 |
| cifar100 | step@0 no-KL - step@80 no-KL | +0.16 | +0.72 | +0.14 | +0.54 | +0.16 | +0.67 | -0.005 | -0.016 | +0.20 |
| cub200 | step@80 vs H-CAST | +0.46 | -0.60 | +0.43 | -0.50 | +0.46 | -0.60 | -0.021 | +0.017 | +0.20 |
| cub200 | step@0 vs H-CAST | -1.31 | -0.06 | -1.14 | -0.03 | -1.31 | -0.10 | +0.039 | -0.010 | +1.00 |
| cub200 | step@80 no-KL vs no-KL | +0.61 | +0.00 | +0.53 | +0.00 | +0.61 | +0.00 | -0.018 | +0.000 | +0.00 |
| cub200 | step@0 no-KL vs no-KL | +0.32 | +0.11 | +0.29 | -0.33 | +0.32 | -0.50 | -0.011 | -0.014 | +0.23 |
| cub200 | step@0 - step@80 | -1.77 | +0.54 | -1.57 | +0.48 | -1.77 | +0.50 | +0.059 | -0.026 | +0.81 |
| cub200 | step@0 no-KL - step@80 no-KL | -0.29 | +0.11 | -0.24 | -0.33 | -0.29 | -0.50 | +0.007 | -0.014 | +0.23 |
| aircraft | step@80 vs H-CAST | +1.65 | +1.87 | +1.57 | -0.06 | +1.65 | -0.50 | -0.090 | -0.093 | -2.60 |
| aircraft | step@0 vs H-CAST | -1.05 | -0.50 | -0.57 | -1.93 | -1.05 | -2.73 | +0.023 | +0.005 | -1.62 |
| aircraft | step@80 no-KL vs no-KL | +0.50 | +2.29 | +0.33 | +0.61 | +0.50 | +1.15 | -0.015 | -0.051 | -3.63 |
| aircraft | step@0 no-KL vs no-KL | -0.73 | +1.40 | -0.58 | -0.48 | -0.73 | -0.19 | +0.032 | -0.008 | -5.23 |
| aircraft | step@0 - step@80 | -2.71 | -2.37 | -2.14 | -1.87 | -2.71 | -2.23 | +0.113 | +0.098 | +0.98 |
| aircraft | step@0 no-KL - step@80 no-KL | -1.23 | -0.89 | -0.91 | -1.10 | -1.23 | -1.34 | +0.046 | +0.043 | -1.60 |

### Interpretation

CIFAR-100 is the only dataset where early HCC with KL is clearly the best HCC setting. However, matched no-KL comparisons are negative, so the evidence does not isolate HCC alone as the cause of the CIFAR gain.

CUB-200 shows the cleanest top-down-only effect. Delayed HCC improves top-down FPA and AHD, including in the no-KL matched comparison, but independent FPA and independent fine accuracy do not improve. HCC is therefore improving hierarchy-enforced inference more than independent fine recognition.

FGVC-Aircraft shows the strongest delayed-HCC benefit. The top-down improvement is robust versus H-CAST and versus no-KL HCAST, and independent FPA also improves strongly. Early HCC is worse for top-down inference and fine accuracy, even though it can reduce independent TICE.

## What HCC Does Internally

The active-window diagnostics below average validation metrics only while HCC is active. `Resid after = 0` means the hard constraint is being imposed exactly in the logged logit residual metric. `Fine prob L1` and `Flip` measure how much the fine distribution changes. `GT prob delta` is post minus pre probability on the ground-truth fine class.

| Dataset | Run | Active epochs | Resid before | Resid after | Resid red | Fine prob L1 | Flip | GT prob delta | GT logit delta | Parent mass pre/post | Sibling rank pre/post | L2 ind given L1 ok | L2 TD given L1 ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar100 | HCC step@80 | 81-100 | 3.727 | 0.000 | 3.727 | 0.393 | 6.77% | -9.08% | -0.031 | 66.22%/52.90% | 1.22/1.22 | 85.13% | 87.98% |
| cifar100 | HCC step@80 no-KL | 81-100 | 1.431 | 0.000 | 1.431 | 0.395 | 7.14% | -9.45% | -1.041 | 69.47%/55.70% | 1.22/1.22 | 84.81% | 87.81% |
| cifar100 | HCC step@0 | 1-100 | 1.610 | 0.000 | 1.610 | 0.370 | 8.60% | +11.89% | +0.910 | 36.00%/50.79% | 1.28/1.28 | 81.19% | 84.81% |
| cifar100 | HCC step@0 no-KL | 1-100 | 1.128 | 0.000 | 1.128 | 0.337 | 8.20% | +10.68% | +0.648 | 40.16%/53.35% | 1.29/1.29 | 81.46% | 84.95% |
| cub200 | HCC step@80 | 81-100 | 5.435 | 0.000 | 5.435 | 0.614 | 5.39% | -19.28% | -0.533 | 75.19%/48.31% | 1.36/1.36 | 83.17% | 85.79% |
| cub200 | HCC step@80 no-KL | 81-100 | 2.317 | 0.000 | 2.317 | 0.621 | 7.38% | -19.98% | -1.675 | 76.39%/49.44% | 1.37/1.37 | 81.25% | 85.49% |
| cub200 | HCC step@0 | 1-100 | 2.231 | 0.000 | 2.231 | 0.364 | 18.13% | +10.45% | +1.200 | 36.12%/48.56% | 1.65/1.65 | 76.42% | 79.66% |
| cub200 | HCC step@0 no-KL | 1-100 | 1.386 | 0.000 | 1.386 | 0.288 | 15.62% | +9.13% | +0.875 | 38.66%/50.10% | 1.63/1.63 | 76.70% | 80.19% |
| aircraft | HCC step@80 | 81-100 | 1.136 | 0.000 | 1.136 | 0.517 | 10.90% | -9.17% | -0.299 | 58.20%/45.84% | 1.13/1.13 | 87.34% | 90.43% |
| aircraft | HCC step@80 no-KL | 81-100 | 0.755 | 0.000 | 0.755 | 0.555 | 10.24% | -11.49% | -1.136 | 56.25%/41.88% | 1.13/1.13 | 87.39% | 90.33% |
| aircraft | HCC step@0 | 1-100 | 1.025 | 0.000 | 1.025 | 0.971 | 69.63% | +31.40% | +3.436 | 9.09%/42.02% | 1.16/1.16 | 81.38% | 87.07% |
| aircraft | HCC step@0 no-KL | 1-100 | 0.887 | 0.000 | 0.887 | 0.895 | 67.93% | +28.66% | +2.890 | 8.72%/39.33% | 1.16/1.16 | 82.06% | 86.98% |

The residual result is unambiguous: when HCC is active, it enforces the logged hierarchy constraints exactly. The performance question is not whether HCC applies the constraint; it is whether that constrained representation is a better representation for the metric and decoder.

The diagnostic pattern splits into two regimes:

- Delayed HCC often rewrites fine probabilities while lowering the ground-truth fine probability relative to raw logits. This is especially strong on CUB, where `step@80` averages about -19 pp GT probability delta. That explains why CUB independent metrics do not improve even though top-down metrics do.
- Early HCC usually increases ground-truth parent mass and ground-truth fine probability, but it can be a very large intervention. Aircraft `step@0` flips around 68-70% of fine argmaxes during the active window. That scale of rewriting is consistent with worse top-down FPA and worse fine accuracy despite lower independent TICE.

Sibling-rank averages are almost unchanged in these scoped runs. The failure mode here is therefore not mainly "GT child rank worsens among siblings"; it is more about the amount and timing of probability mass redistribution, the selected decoder, and the way optimization adapts to the hard constraint.

## Step@0 Versus Step@80

Delayed HCC lets H-CAST learn mostly unconstrained until logged epoch 81, then imposes the constraint. The immediate validation shock at activation is dataset-dependent.

| Dataset | Run | FPA TD 80->81 | FPA TD 80->100 | FPA Ind 80->81 | FPA Ind 80->100 | Fine TD 80->81 | Fine Ind 80->81 | GT prob delta e81 | Flip e81 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar100 | HCC step@80 | -0.45 | +0.37 | -2.11 | -0.05 | -0.45 | -1.50 | -12.54% | 8.83% |
| cifar100 | HCC step@80 no-KL | +0.50 | +0.68 | -0.47 | +0.37 | +0.50 | -0.40 | -12.10% | 8.28% |
| cub200 | HCC step@80 | -0.23 | +0.04 | -2.64 | -1.60 | -0.23 | -2.72 | -20.22% | 6.31% |
| cub200 | HCC step@80 no-KL | -0.15 | -0.80 | -4.32 | -3.73 | -0.15 | -4.55 | -21.20% | 9.31% |
| aircraft | HCC step@80 | +1.71 | +2.63 | +2.10 | +3.21 | +1.71 | +1.07 | -9.74% | 10.66% |
| aircraft | HCC step@80 no-KL | -0.11 | +0.81 | +0.39 | +1.40 | -0.11 | -0.59 | -12.86% | 10.63% |

This table explains why step@80 is attractive for Aircraft but ambiguous elsewhere. Aircraft gets an immediate positive activation response in the KL run. CUB gets a strong independent-decoding hit at activation, and the no-KL run never recovers its independent fine behavior. CIFAR has mixed immediate behavior but can recover top-down by epoch 100.

Best-epoch selection also matters. CUB `step@80` has best independent epochs 75 and 73 for KL/no-KL, before HCC activation, while top-down best epochs are after activation. This means the independent test results for delayed CUB HCC do not provide evidence that active HCC improves independent inference. They mainly show that the independent-selected checkpoint avoided the HCC phase.

## Loss Dynamics

The table averages the last five training epochs. These are training losses, not validation or test losses.

| Dataset | Run | total | level_ce | gk_loss | L0 | L1 | L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar100 | H-CAST | 6.056 | 5.617 | 0.878 | 1.215 | 1.743 | 2.658 |
| cifar100 | HCAST no-KL | 5.606 | 5.606 | 0.000 | 1.214 | 1.739 | 2.653 |
| cifar100 | HCC step@80 | 5.979 | 5.528 | 0.902 | 1.161 | 1.714 | 2.652 |
| cifar100 | HCC step@80 no-KL | 5.775 | 5.775 | 0.000 | 1.216 | 1.793 | 2.766 |
| cifar100 | HCC step@0 | 6.174 | 5.701 | 0.946 | 1.223 | 1.795 | 2.683 |
| cifar100 | HCC step@0 no-KL | 5.688 | 5.688 | 0.000 | 1.221 | 1.784 | 2.683 |
| cub200 | H-CAST | 5.270 | 4.951 | 0.638 | 0.874 | 1.512 | 2.565 |
| cub200 | HCAST no-KL | 4.951 | 4.951 | 0.000 | 0.874 | 1.513 | 2.564 |
| cub200 | HCC step@80 | 5.581 | 5.213 | 0.734 | 0.876 | 1.575 | 2.762 |
| cub200 | HCC step@80 no-KL | 5.222 | 5.222 | 0.000 | 0.875 | 1.556 | 2.791 |
| cub200 | HCC step@0 | 5.374 | 5.039 | 0.671 | 0.884 | 1.569 | 2.586 |
| cub200 | HCC step@0 no-KL | 5.022 | 5.022 | 0.000 | 0.880 | 1.537 | 2.605 |
| aircraft | H-CAST | 6.101 | 5.717 | 0.768 | 1.468 | 1.900 | 2.349 |
| aircraft | HCAST no-KL | 5.725 | 5.725 | 0.000 | 1.470 | 1.903 | 2.352 |
| aircraft | HCC step@80 | 6.119 | 5.737 | 0.765 | 1.472 | 1.949 | 2.316 |
| aircraft | HCC step@80 no-KL | 5.777 | 5.777 | 0.000 | 1.473 | 1.965 | 2.338 |
| aircraft | HCC step@0 | 5.927 | 5.572 | 0.710 | 1.478 | 1.906 | 2.189 |
| aircraft | HCC step@0 no-KL | 5.604 | 5.604 | 0.000 | 1.473 | 1.915 | 2.215 |

Loss does not alone explain success. CUB HCC has higher last-epoch fine loss and still improves top-down metrics under delayed HCC. Aircraft early HCC has lower final fine loss but worse top-down and fine accuracy. HCC changes the objective surface because training and evaluation use effective constrained logits when active; a lower constrained training loss is not equivalent to better unconstrained fine ranking.

The best loss-aligned case is CIFAR `step@80` with KL: lower last-window total and level CE than H-CAST, with a modest top-down gain. The clearest loss/accuracy mismatch is Aircraft `step@0`: lower final fine loss but worse fine accuracy.

## Gradients And Parameters

The most consistent optimization signal is that HCC reduces fine-level update pressure and reduces positive alignment between fine and higher-priority gradients. This is strongest on Aircraft, moderate on CUB, and weaker on CIFAR.

Whole-run relative changes are used for `step@0`, because HCC is active from epoch 1.

| Dataset | Comparison | fine grad | mid grad | coarse grad | fine/higher cos | delta all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar100 | step@0 vs H-CAST | -6.7% | -6.2% | -1.2% | -33.1% | +2.0% |
| cifar100 | step@0 no-KL vs no-KL | -1.9% | -17.3% | -0.9% | -32.3% | +2.4% |
| cub200 | step@0 vs H-CAST | -8.4% | -3.0% | +2.8% | -45.6% | +2.5% |
| cub200 | step@0 no-KL vs no-KL | -7.0% | -15.8% | +4.1% | -43.2% | +1.1% |
| aircraft | step@0 vs H-CAST | -50.4% | -21.4% | +0.4% | -65.9% | +1.7% |
| aircraft | step@0 no-KL vs no-KL | -55.4% | -28.2% | -2.8% | -66.1% | +2.1% |

Late-window relative changes are used for `step@80`, because HCC is active only from epoch 81 onward.

| Dataset | Comparison | fine grad | mid grad | coarse grad | fine/higher cos | delta all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar100 | step@80 vs H-CAST | -9.4% | -6.8% | -0.7% | -39.7% | -0.8% |
| cifar100 | step@80 no-KL vs no-KL | -4.5% | -16.5% | +1.3% | -37.9% | -2.7% |
| cub200 | step@80 vs H-CAST | -10.3% | -6.9% | -1.2% | -65.8% | -5.4% |
| cub200 | step@80 no-KL vs no-KL | -10.7% | -18.8% | -2.4% | -61.5% | -9.7% |
| aircraft | step@80 vs H-CAST | -47.2% | -24.1% | +1.9% | -78.1% | -19.2% |
| aircraft | step@80 no-KL vs no-KL | -52.0% | -31.4% | -2.6% | -76.7% | -20.7% |

This is the strongest "why" evidence in the logs. HCC is not just a post-processing correction. When active during training, it changes gradient magnitudes and the relative geometry of level losses. In particular:

- The fine gradient on shared trunk `t1` is reduced, especially on Aircraft.
- Fine/higher gradient cosine drops substantially, meaning fine updates become less aligned with the combined higher-level direction.
- Late HCC reduces per-epoch parameter movement, especially on Aircraft and no-KL CUB.

The parameter norms themselves stay stable; the important change is not parameter scale but update direction and update size.

## Lexicographic Optimization Perspective

The research goal is to understand whether lexicographic methods can improve H-CAST. HCC is a first step because it constrains H-CAST with a HardNet-Aff-style hierarchy-affine projection at the output/logit level. The useful question is therefore not only "does HCC improve metrics?", but "does HCC behave like a lexicographic method in gradient space?"

The answer is partial. HCC is lex-like in its effects, not in its mechanism.

Explicit lexicographic training computes level gradients, then projects lower-priority gradients away from higher-priority directions. In the logs this produces `post_*` diagnostics. The defining signature is that `post_cos_t1_fine_proj_higher` and `post_cos_t1_mid_proj_coarse` move to approximately zero. HCC has no `post_*` projection diagnostics because it does not project gradients. Instead, it changes the logits used by the loss; the gradients are then ordinary gradients of a constrained objective.

The table below compares the no-KL H-CAST family, because all available `hcast_lex_*` reference runs are no-KL. Windows are chosen to match the active intervention: whole-run windows for start@0, and epoch 81+ windows for start@80 or step@80. `post` columns exist only for lex runs.

| Dataset | Run | Window | n | raw fine/higher cos | post fine/higher cos | raw mid/coarse cos | post mid/coarse cos | raw fine grad | post fine grad | proj applied fine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar100 | HCAST no-KL | 1-100 | 100 | 0.6557 | NA | 0.7531 | NA | 6.1290 | NA | NA |
| cifar100 | HCC step@80 no-KL | 81-100 | 20 | 0.3906 | NA | 0.4002 | NA | 6.2698 | NA | NA |
| cifar100 | HCC step@0 no-KL | 1-100 | 100 | 0.4440 | NA | 0.3918 | NA | 6.0120 | NA | NA |
| cifar100 | LEX start@0 | 1-100 | 79 | 0.6615 | 0.0000 | 0.7332 | -0.0000 | 6.4994 | 4.4559 | 100.0% |
| cifar100 | LEX start@80 | 81-100 | 6 | 0.6283 | 0.0000 | 0.7536 | -0.0000 | 6.6070 | 4.8627 | 100.0% |
| cub200 | HCAST no-KL | 1-100 | 100 | 0.4092 | NA | 0.4845 | NA | 7.4525 | NA | NA |
| cub200 | HCC step@80 no-KL | 81-100 | 20 | 0.1329 | NA | 0.1938 | NA | 6.3024 | NA | NA |
| cub200 | HCC step@0 no-KL | 1-100 | 100 | 0.2326 | NA | 0.2378 | NA | 6.9304 | NA | NA |
| cub200 | LEX start@0 | 1-100 | 100 | 0.4071 | -0.0000 | 0.4602 | -0.0000 | 8.7737 | 7.4578 | 100.0% |
| cub200 | LEX start@80 | 81-100 | 20 | 0.3508 | 0.0000 | 0.4931 | -0.0000 | 7.1573 | 6.3901 | 100.0% |
| aircraft | HCAST no-KL | 1-100 | 100 | 0.6124 | NA | 0.6720 | NA | 6.0065 | NA | NA |
| aircraft | HCC step@80 no-KL | 81-100 | 20 | 0.1270 | NA | 0.3243 | NA | 2.7245 | NA | NA |
| aircraft | HCC step@0 no-KL | 1-100 | 100 | 0.2075 | NA | 0.4198 | NA | 2.6803 | NA | NA |
| aircraft | LEX start@0 | 1-100 | 100 | 0.6269 | 0.0000 | 0.6363 | 0.0000 | 7.0012 | 4.8385 | 100.0% |
| aircraft | LEX start@80 | 81-100 | 20 | 0.5300 | 0.0000 | 0.6561 | 0.0000 | 5.5959 | 4.3164 | 100.0% |

This comparison gives three important insights.

First, explicit lexicographic training does exactly what it should in gradient space. Even when raw fine/higher cosine remains high, the post-projection fine/higher cosine is essentially zero and the fine projection is applied in all measured epochs. That is the clean signature of real lexicographic optimization.

Second, HCC does not reproduce that mechanism. There is no post-gradient operation. Instead, HCC reduces the raw cosine before any gradient projection would be needed. For example, Aircraft no-KL has raw fine/higher cosine 0.6124 in HCAST, 0.1270 in HCC step@80, but 0.5300 raw and 0.0000 post in LEX start@80. The lex run says "the raw objective still couples fine and higher gradients, so I project the update." The HCC run says "the constrained logits changed the objective so the raw fine gradient is already much less coupled to the higher-level direction."

Third, HCC can be more aggressive than lex on gradient magnitude. On Aircraft, HCC step@80 no-KL cuts raw fine gradient norm to 2.7245, while LEX start@80 has raw fine norm 5.5959 and post fine norm 4.3164. This is one reason Aircraft delayed HCC helps: it does not merely orthogonalize fine updates; it strongly damps fine-level pressure while preserving or improving hierarchy-enforced path metrics.

### Does HCC Follow A Lexicographic Principle?

Only weakly, and only as an implicit effect.

If lexicographic priority means "higher-level objectives should be protected from lower-level updates by explicit gradient projection," then HCC is not lexicographic. It never guarantees that fine updates are orthogonal to higher-priority gradients.

If lexicographic priority means "the optimization should become less dominated by fine-level pressure on shared parameters," then HCC is partially lexicographic. Across datasets it lowers fine gradient norms and lowers fine/higher cosine alignment, especially when it works well:

- Aircraft delayed HCC no-KL: fine/higher cosine drops from 0.5449 to 0.1270 in the late window, and total parameter movement drops about 20.7%.
- CUB delayed HCC no-KL: fine/higher cosine drops from 0.3451 to 0.1329 in the late window, and total parameter movement drops about 9.7%.
- CIFAR early HCC no-KL: fine/higher cosine drops from 0.6557 to 0.4440 over the run, but this does not translate into a no-KL performance gain.

So lower cosine is not sufficient. It is a necessary-looking signature for HCC's intended effect, but it must be paired with tolerable probability rewriting, stable fine accuracy, and the right decoder.

### What The Lex Runs Suggest About Future H-CAST Research

The lex reference runs are not the main benchmark, but they are informative:

- CIFAR lex start@0 reaches 74.51% top-down FPA and 72.80% independent FPA in the no-KL family, with independent TICE 6.54%. That is better than the no-KL HCC variants on structural error, suggesting explicit lexicographic projection may help CIFAR more cleanly than HCC alone.
- CUB lex start@0 gives stronger independent behavior than no-KL HCC: independent FPA 79.56% and independent fine accuracy 81.76%. HCC step@80 no-KL gives stronger top-down FPA, 81.55%, but not stronger independent fine recognition. This supports the idea that HCC and lex optimize different parts of the hierarchy/fine tradeoff.
- Aircraft delayed HCC no-KL is better than the auxiliary lex runs on both top-down and independent FPA. This suggests that, for Aircraft, hard output consistency is more useful than pure gradient priority projection, at least with these schedules.

The research implication is that HCC is a useful first HardNet-Aff constraint, but it is not a substitute for lexicographic optimization. A natural next hypothesis is that H-CAST may need both: HCC to make logits hierarchy-feasible, and lexicographic gradient projection to protect higher-level objectives when fine-level loss dominates shared parameters.

## Dataset-Level Conclusions

### CIFAR-100

HCC with KL gives small but real-looking gains, strongest for `step@0`. The best top-down result is 74.86% FPA versus 74.13% for H-CAST. The best independent result is also `step@0`, but the independent AHD is slightly worse than baseline and TICE is worse. HCC no-KL does not beat the no-KL baseline, which weakens the claim that HCC alone is the causal improvement on CIFAR.

The internal diagnostics are favorable for early HCC: positive GT probability deltas, positive GT parent mass shifts, and moderate flip rates around 8%. CIFAR appears able to train under the hard constraint from the beginning without severe fine-class disruption.

### CUB-200-2011

CUB is the clearest example of HCC improving top-down path metrics while failing to improve independent fine recognition. Delayed HCC with no-KL gives the strongest top-down FPA, 81.55% versus 80.94% for matched no-KL HCAST. But independent FPA and independent fine accuracy are unchanged in that matched comparison, and delayed HCC with KL is worse independently than H-CAST.

The reason is visible in the active-window diagnostics: late HCC imposes exact residual satisfaction but strongly lowers GT fine probability and GT parent mass relative to raw logits. Top-down decoding can still benefit from the path constraint, but independent fine ranking does not.

Early HCC with KL is actively bad for top-down metrics on CUB. The top-down best epoch is 44, suggesting that the best checkpoint occurs well before the end of training even though HCC is always active.

### FGVC-Aircraft

Aircraft is where delayed HCC works best. `step@80` improves top-down FPA by +1.65 pp and independent FPA by +1.87 pp versus H-CAST. It also improves AHD and independent TICE substantially. The matched no-KL comparison still favors delayed HCC for top-down FPA, independent FPA, wAP, AHD, and TICE.

Early HCC is too strong on Aircraft. It flips about 68-70% of fine argmaxes over the active window and strongly rewrites the fine distribution. That can reduce TICE, but it damages top-down FPA, wAP, and fine accuracy. Aircraft therefore benefits from HCC as a late regularizing or consistency intervention, not as an always-on training regime.

## Overall Mechanistic Interpretation

HCC works by enforcing hierarchy-compatible effective logits. That reliably improves structural consistency, but the resulting performance depends on whether the dataset and decoder benefit from that constraint.

The evidence supports five claims:

1. HCC is mechanically effective: active HCC drives the logged residual after projection to zero in all scoped HCC runs.
2. HCC is decoder-sensitive: top-down decoding benefits more often than independent decoding, especially on CUB.
3. Timing matters: early HCC is beneficial on CIFAR with KL, harmful on Aircraft, and mixed on CUB. Delayed HCC is strongest on Aircraft and useful for CUB top-down metrics.
4. HCC changes optimization, not only inference: it reduces fine-level gradient magnitudes, lowers fine/higher gradient alignment, and often reduces late parameter movement.
5. HCC is lex-like but not lexicographic: it indirectly reduces raw lower/higher gradient coupling, while explicit lexicographic training directly projects lower-priority gradients and drives post-projection cosines to zero.

The best conceptual model is that HCC trades independent fine-class freedom for hierarchy-consistent path structure while also damping fine-level pressure on shared parameters. If the raw fine classifier is already well calibrated inside the hierarchy, this trade can improve path metrics. If the projection rewrites fine probabilities too aggressively, it can harm fine recognition even while making the hierarchy cleaner.

## Limitations

Each condition appears to be a single run, so these are empirical run-level conclusions, not statistical claims over seeds.

The main HCC conclusions intentionally exclude commented-out runs, inverse-step runs, linear schedules, and conditional variants. Lexicographic runs are included only as an auxiliary gradient-geometry reference, not as a primary benchmark.

The CIFAR lex logs are incomplete relative to their test checkpoint metadata, so CIFAR lex performance should be treated as side evidence. The gradient-geometry conclusion remains useful because the available logged lex epochs still show the expected post-projection cosine behavior.

The diagnostics are epoch-level means. They cannot show per-class or per-instance failure cases. A useful next analysis would break HCC effects down by parent node, sibling count, and whether HCC flips an originally correct fine prediction into an incorrect one.
