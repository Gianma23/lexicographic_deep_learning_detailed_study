# Hier-COS Weight-Mode Lex Analysis

Date: 2026-05-23

This report analyzes completed Hier-COS runs with `model.loss=per_level_kl_reg` and `train.lexicographic.enabled=true`. The goal is to understand what explicit lexicographic gradient projection contributes under different Hier-COS weight modes. The scope is intentionally limited to existing completed runs; it does not claim causal lex-vs-nonlex performance gains because matched complete non-lex per-level runs are not available.

Generated CSV tables are under `/scratch/g.saggini1/outputs/analysis/hiercos_weight_modes`.

## Executive Answer

- Lex projection behaved as intended in finite epochs: intended post-projection cosines were numerically near zero after projecting mid off coarse and fine off the composed higher-priority direction.
- Weight mode mainly changed the raw gradient field and the stability/performance tradeoff before projection. `kl_leaf` made the fine objective strongest, `kl_coarse` made the coarse objective strongest, and `equal` kept the three KL target-path weights balanced.
- CIFAR-100 `kl_coarse` first shows non-finite diagnostics at logged epoch 10, so its late-window losses, gradients, and parameter norms are not interpretable as a stable training regime.
- Negative raw cosine similarity is not an error. It means the local level-gradient directions conflict on shared trainable parameters, so improving one level can locally oppose another.
- Negative component-wise post cosines are also possible because the implementation projects fine against one composed higher-priority vector, not independently against coarse and mid as separate constraints.
- Final test selection is mode-specific: CIFAR-100: best top-down FPA `kl_leaf`, best independent FPA `kl_leaf`, lowest independent TICE `equal`; CUB-200-2011: best top-down FPA `kl_leaf`, best independent FPA `kl_leaf`, lowest independent TICE `kl_leaf`; FGVC-Aircraft: best top-down FPA `kl_leaf`, best independent FPA `kl_leaf`, lowest independent TICE `equal`.

## Scope And Verification

The script includes exactly the nine approved run directories `hiercos_{dataset}_per_level_kl_reg_{equal,kl_leaf,kl_coarse}`. It validates `model.name=hiercos`, `model.loss=per_level_kl_reg`, the expected `model.weight_mode`, `train.lexicographic.enabled=true`, `train.lexicographic.start_epoch=0`, and 100 logged epochs.

| Dataset | Weight | Epochs | Best TD/Ind | First Non-Finite Epoch |
| --- | --- | --- | --- | --- |
| CIFAR-100 | equal | 100 | 96/97 | - |
| CIFAR-100 | kl_leaf | 100 | 91/91 | - |
| CIFAR-100 | kl_coarse | 100 | 2/2 | 10 |
| CUB-200-2011 | equal | 100 | 93/93 | - |
| CUB-200-2011 | kl_leaf | 100 | 98/95 | - |
| CUB-200-2011 | kl_coarse | 100 | 94/94 | - |
| FGVC-Aircraft | equal | 100 | 71/71 | - |
| FGVC-Aircraft | kl_leaf | 100 | 97/98 | - |
| FGVC-Aircraft | kl_coarse | 100 | 60/73 | - |

The `per_level_kl_reg` objective exposes three differentiable losses for lexicographic projection. In this implementation, the weight mode changes the KL target-path mass. The per-level regularization terms remain separately logged and enter each `loss_level_*` through `kl_level_* + alpha * reg_level_*`. The auxiliary `kl` and `reg` columns are useful diagnostics, but the logged `total` in this mode is the sum of the three exposed level losses.

| Mode | Coarse | Mid | Fine |
| --- | --- | --- | --- |
| equal | 0.333333 | 0.333333 | 0.333333 |
| kl_leaf | 0.161570 | 0.225489 | 0.612942 |
| kl_coarse | 0.612942 | 0.225489 | 0.161570 |

## Final Test Metrics

Values use the same checkpoint-selection rule as training: top-down columns come from the top-down-selected checkpoint, and independent columns come from the independent-selected checkpoint. FPA, weighted AP, and accuracy are higher-is-better; AHD and TICE are lower-is-better.

| Dataset | Weight | Best TD/Ind | FPA TD | FPA Ind | wAP TD | wAP Ind | AHD TD | AHD Ind | TICE Ind | Fine TD | Fine Ind |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-100 | equal | 96/97 | 77.08% | 77.08% | 79.47% | 79.61% | 0.450 | 0.448 | 0.42% | 77.08% | 77.23% |
| CIFAR-100 | kl_leaf | 91/91 | 77.71% | 77.61% | 79.96% | 80.01% | 0.445 | 0.447 | 0.66% | 77.71% | 77.77% |
| CIFAR-100 | kl_coarse | 2/2 | 6.32% | 5.06% | 10.69% | 10.69% | 2.309 | 2.343 | 27.81% | 6.32% | 6.24% |
| CUB-200-2011 | equal | 93/93 | 74.76% | 74.72% | 78.23% | 78.40% | 0.383 | 0.383 | 0.61% | 74.76% | 74.95% |
| CUB-200-2011 | kl_leaf | 98/95 | 76.01% | 75.69% | 79.30% | 79.18% | 0.363 | 0.366 | 0.56% | 76.01% | 75.83% |
| CUB-200-2011 | kl_coarse | 94/94 | 28.61% | 28.39% | 38.95% | 38.87% | 1.041 | 1.043 | 1.49% | 28.61% | 28.51% |
| FGVC-Aircraft | equal | 71/71 | 78.76% | 78.70% | 81.94% | 82.05% | 0.491 | 0.492 | 0.57% | 78.76% | 78.91% |
| FGVC-Aircraft | kl_leaf | 97/98 | 80.95% | 81.07% | 83.60% | 83.62% | 0.449 | 0.450 | 0.78% | 80.95% | 81.10% |
| FGVC-Aircraft | kl_coarse | 60/73 | 79.45% | 80.11% | 82.20% | 82.89% | 0.486 | 0.474 | 0.69% | 79.45% | 80.44% |

Across these existing lex runs, no single weight mode dominates all datasets and metrics. `kl_leaf` is often competitive for independent fine behavior because it gives the leaf KL term the largest target-path mass. `kl_coarse` can improve hierarchy-prioritized behavior on some datasets, but CIFAR-100 shows a clear stability failure in this mode. `equal` is the most conservative weighting and avoids the CIFAR-100 non-finite failure observed for `kl_coarse`.

## Loss Dynamics

Early window means use epochs 1-10. Late window means use epochs 91-100. Entries such as `value (n/10)` mean only `n` finite epoch values were available in that window.

### Early Loss Window

| Dataset | Weight | Total | KL Aux | Reg Aux | Loss L0 | Loss L1 | Loss L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-100 | equal | 3.665 | 1.950 | 12.323 | 0.959 | 1.146 | 1.560 |
| CIFAR-100 | kl_leaf | 3.734 | 2.188 | 12.304 | 0.738 | 1.002 | 1.994 |
| CIFAR-100 | kl_coarse | 3.671 (9/10) | 2.139 (9/10) | 12.032 (9/10) | 1.394 (9/10) | 1.065 (9/10) | 1.213 (9/10) |
| CUB-200-2011 | equal | 3.029 | 0.819 | 11.119 | 0.490 | 0.841 | 1.698 |
| CUB-200-2011 | kl_leaf | 2.885 | 0.825 | 11.298 | 0.423 | 0.672 | 1.791 |
| CUB-200-2011 | kl_coarse | 3.957 | 1.924 | 11.020 | 0.991 | 1.240 | 1.726 |
| FGVC-Aircraft | equal | 3.640 | 1.241 | 13.002 | 0.999 | 1.229 | 1.412 |
| FGVC-Aircraft | kl_leaf | 3.438 | 1.185 | 13.226 | 0.786 | 1.073 | 1.579 |
| FGVC-Aircraft | kl_coarse | 4.196 | 1.905 | 13.605 | 1.506 | 1.330 | 1.360 |

### Late Loss Window

| Dataset | Weight | Total | KL Aux | Reg Aux | Loss L0 | Loss L1 | Loss L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-100 | equal | 1.698 | 0.462 | 2.750 | 0.117 | 0.565 | 1.016 |
| CIFAR-100 | kl_leaf | 1.461 | 0.399 | 2.615 | 0.138 | 0.291 | 1.032 |
| CIFAR-100 | kl_coarse | NaN | NaN | NaN | NaN | NaN | NaN |
| CUB-200-2011 | equal | 1.921 | 0.519 | 3.036 | 0.109 | 0.605 | 1.207 |
| CUB-200-2011 | kl_leaf | 1.736 | 0.511 | 2.945 | 0.126 | 0.309 | 1.301 |
| CUB-200-2011 | kl_coarse | 2.081 | 0.708 | 4.426 | 0.096 | 0.653 | 1.332 |
| FGVC-Aircraft | equal | 1.673 | 0.305 | 2.693 | 0.181 | 0.573 | 0.920 |
| FGVC-Aircraft | kl_leaf | 1.470 | 0.284 | 2.555 | 0.189 | 0.342 | 0.940 |
| FGVC-Aircraft | kl_coarse | 1.421 | 0.194 | 2.968 | 0.152 | 0.592 | 0.677 |

The loss tables show why CIFAR-100 `kl_coarse` should be treated as a stability result rather than a valid late-training comparison: the run becomes non-finite early and all late loss diagnostics are NaN. For stable runs, the late `loss_level_*` pattern reflects both the KL target-path weights and the unweighted per-level regularization pressure.

## Gradient Geometry

Raw cosines are measured before lexicographic projection. Post cosines are measured after the projected-gradient composition. The intended post constraints are `post_cos_t1_mid_proj_coarse` and `post_cos_t1_fine_proj_higher`; these should be near zero when projection is active and numerically stable.

### Early Gradient Cosines

| Dataset | Weight | Raw Mid/Coarse | Raw Fine/Higher | Raw Fine/Coarse | Raw Fine/Mid | Post Mid/Coarse | Post Fine/Higher |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-100 | equal | 0.227 | 0.207 | 0.089 | 0.246 | -0.000000 | 0.000000 |
| CIFAR-100 | kl_leaf | 0.282 | 0.234 | 0.123 | 0.243 | -0.000000 | -0.000000 |
| CIFAR-100 | kl_coarse | 0.136 (9/10) | 0.033 (9/10) | -0.051 (9/10) | 0.249 (9/10) | -0.000000 (9/10) | 0.000000 (9/10) |
| CUB-200-2011 | equal | -0.227 | 0.018 | -0.175 | 0.183 | 0.000000 | 0.000000 |
| CUB-200-2011 | kl_leaf | -0.017 | -0.191 | -0.219 | -0.064 | -0.000000 | -0.000000 |
| CUB-200-2011 | kl_coarse | 0.199 | 0.004 | -0.048 | 0.199 | -0.000000 | -0.000000 |
| FGVC-Aircraft | equal | 0.052 | 0.158 | 0.000 | 0.258 | -0.000000 | -0.000000 |
| FGVC-Aircraft | kl_leaf | 0.137 | -0.020 | -0.079 | 0.053 | -0.000000 | 0.000000 |
| FGVC-Aircraft | kl_coarse | 0.158 | 0.168 | 0.058 | 0.392 | -0.000000 | 0.000000 |

### Late Gradient Cosines

| Dataset | Weight | Raw Mid/Coarse | Raw Fine/Higher | Raw Fine/Coarse | Raw Fine/Mid | Post Mid/Coarse | Post Fine/Higher |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-100 | equal | -0.855 | 0.196 | -0.569 | 0.339 | 0.000000 | 0.000000 |
| CIFAR-100 | kl_leaf | -0.673 | -0.437 | -0.603 | 0.084 | 0.000000 | 0.000000 |
| CIFAR-100 | kl_coarse | NaN | NaN | NaN | NaN | NaN | NaN |
| CUB-200-2011 | equal | -0.885 | 0.005 | -0.496 | 0.154 | 0.000000 | -0.000000 |
| CUB-200-2011 | kl_leaf | -0.685 | -0.541 | -0.563 | -0.089 | 0.000000 | -0.000000 |
| CUB-200-2011 | kl_coarse | -0.862 | 0.133 | -0.430 | 0.233 | 0.000000 | 0.000000 |
| FGVC-Aircraft | equal | -0.817 | -0.126 | -0.538 | 0.155 | 0.000000 | 0.000000 |
| FGVC-Aircraft | kl_leaf | -0.529 | -0.680 | -0.530 | -0.230 | 0.000000 | 0.000000 |
| FGVC-Aircraft | kl_coarse | -0.828 | 0.087 | -0.571 | 0.331 | 0.000000 | 0.000000 |

The key lexicographic contribution is visible in the gap between raw and post cosines. Raw mid/coarse and fine/higher cosines can be positive, weak, or negative depending on dataset and weight mode. After projection, the intended post cosines are approximately zero in finite epochs. That is the explicit gradient-space behavior that ordinary weighted training does not guarantee.

## Gradient Norms

The norm table uses `t1`, which is the relevant shared-parameter block for these Hier-COS runs. In practice the Hier-COS diagnostics mostly collapse to `t1` because all trainable parameters receive all three level losses.

### Early Gradient Norms

| Dataset | Weight | Coarse Grad | Mid Grad | Fine Grad | Fine/Coarse | Post Mid/Raw Mid | Post Fine/Raw Fine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-100 | equal | 1.921 | 1.858 | 1.707 | 0.888 | 0.952 | 0.965 |
| CIFAR-100 | kl_leaf | 1.022 | 1.299 | 3.095 | 3.028 | 0.951 | 0.963 |
| CIFAR-100 | kl_coarse | 2.537 (9/10) | 0.882 (9/10) | 0.588 (9/10) | 0.232 | 0.964 | 0.993 |
| CUB-200-2011 | equal | 2.825 | 2.955 | 2.999 | 1.062 | 0.958 | 0.981 |
| CUB-200-2011 | kl_leaf | 1.694 | 2.098 | 4.737 | 2.797 | 0.993 | 0.963 |
| CUB-200-2011 | kl_coarse | 5.366 | 1.563 | 1.304 | 0.243 | 0.930 | 0.913 |
| FGVC-Aircraft | equal | 3.483 | 2.961 | 2.823 | 0.811 | 0.985 | 0.976 |
| FGVC-Aircraft | kl_leaf | 1.977 | 2.089 | 4.631 | 2.343 | 0.985 | 0.984 |
| FGVC-Aircraft | kl_coarse | 5.429 | 1.715 | 1.262 | 0.232 | 0.978 | 0.975 |

### Late Gradient Norms

| Dataset | Weight | Coarse Grad | Mid Grad | Fine Grad | Fine/Coarse | Post Mid/Raw Mid | Post Fine/Raw Fine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-100 | equal | 0.973 | 2.828 | 3.758 | 3.863 | 0.472 | 0.645 |
| CIFAR-100 | kl_leaf | 0.918 | 1.426 | 4.823 | 5.253 | 0.722 | 0.578 |
| CIFAR-100 | kl_coarse | NaN | NaN | NaN | NaN | NaN | NaN |
| CUB-200-2011 | equal | 0.591 | 1.961 | 3.080 | 5.211 | 0.462 | 0.606 |
| CUB-200-2011 | kl_leaf | 0.598 | 1.075 | 4.239 | 7.092 | 0.725 | 0.497 |
| CUB-200-2011 | kl_coarse | 0.577 | 1.843 | 1.582 | 2.740 | 0.492 | 0.866 |
| FGVC-Aircraft | equal | 0.720 | 1.605 | 2.175 | 3.022 | 0.576 | 0.691 |
| FGVC-Aircraft | kl_leaf | 0.609 | 0.908 | 2.672 | 4.390 | 0.848 | 0.597 |
| FGVC-Aircraft | kl_coarse | 0.733 | 1.529 | 1.480 | 2.019 | 0.558 | 0.819 |

`Post Mid/Raw Mid` and `Post Fine/Raw Fine` quantify how much lexicographic projection reduces lower-priority update components. A low ratio means much of that level's raw gradient was aligned with, or conflicting along, a protected higher-priority direction and was removed by projection. This is the most direct answer to what lex mode brings: it changes the update direction even when scalar weighting leaves raw gradients coupled.

## Parameter Movement

`delta_param_norm_t1` is the norm of the end-of-epoch minus start-of-epoch parameter vector. The cumulative value below is the sum of logged epoch deltas, not a single end-to-end displacement norm.

| Dataset | Weight | Param Norm Mean | Delta Early | Delta Late | Sum Epoch Deltas | Finite Delta Epochs |
| --- | --- | --- | --- | --- | --- | --- |
| CIFAR-100 | equal | 63.723 | 6.635 | 0.069 | 349.014 | 100 |
| CIFAR-100 | kl_leaf | 65.407 | 6.917 | 0.071 | 374.425 | 100 |
| CIFAR-100 | kl_coarse | 125.470 (13/100) | 8.752 | NaN | 87.516 | 13 |
| CUB-200-2011 | equal | 86.332 | 5.305 | 0.045 | 153.043 | 100 |
| CUB-200-2011 | kl_leaf | 86.592 | 5.654 | 0.043 | 161.288 | 100 |
| CUB-200-2011 | kl_coarse | 88.123 | 6.219 | 0.038 | 196.219 | 100 |
| FGVC-Aircraft | equal | 88.814 | 4.754 | 0.032 | 134.605 | 100 |
| FGVC-Aircraft | kl_leaf | 88.835 | 4.954 | 0.032 | 135.160 | 100 |
| FGVC-Aircraft | kl_coarse | 89.406 | 5.160 | 0.030 | 154.264 | 100 |

Late parameter movement is small for stable runs because the cosine scheduler is near zero at the end of training. Early movement is more diagnostic of how strongly each weight mode drives shared parameters before the schedule decays. CIFAR-100 `kl_coarse` leaves only a short finite prefix for parameter diagnostics, so its cumulative movement is not comparable to the stable 100-epoch runs.

## Negative Cosine Similarity

| Dataset | Weight | Raw Cos Negative | Fine/Higher Negative | Component Post Negative | Max Abs Intended Post |
| --- | --- | --- | --- | --- | --- |
| CIFAR-100 | equal | 22.75% | 13.00% | 50.00% | 0.00000004 |
| CIFAR-100 | kl_leaf | 33.25% | 38.00% | 50.00% | 0.00000002 |
| CIFAR-100 | kl_coarse | 36.11% | 22.22% | 50.00% | 0.00000001 |
| CUB-200-2011 | equal | 71.50% | 86.00% | 50.00% | 0.00000010 |
| CUB-200-2011 | kl_leaf | 97.75% | 98.00% | 50.00% | 0.00000008 |
| CUB-200-2011 | kl_coarse | 39.25% | 9.00% | 50.50% | 0.00000008 |
| FGVC-Aircraft | equal | 69.50% | 89.00% | 50.00% | 0.00000009 |
| FGVC-Aircraft | kl_leaf | 93.25% | 95.00% | 50.00% | 0.00000007 |
| FGVC-Aircraft | kl_coarse | 43.75% | 0.00% | 50.00% | 0.00000010 |

Negative raw cosine similarity is normal in multi-objective hierarchical training. A negative value means the two level losses would update shared parameters in opposing local directions. With `per_level_kl_reg`, this is especially expected when the KL target-path mass emphasizes one level while regularization and classification pressure still act across all levels.

For lexicographic mode, the important check is not whether every cosine is non-negative. The important check is whether the protected post-projection relationships are near zero. Component-wise post cosines such as `post_cos_t1_fine_proj_coarse` can be negative because the current implementation projects fine once against the composed higher-priority direction `coarse + projected_mid`, not separately against every higher-priority component. Therefore a negative component-wise post cosine is not by itself a bug.

## Interpretation For Lex Mode

These runs support a gradient-space interpretation, not a causal performance claim. Lex mode brings an update rule that protects higher-priority objectives by removing lower-priority gradient components along protected directions. This is visible regardless of whether the raw cosine is positive or negative: positive alignment is reduced because it would move along a higher-priority direction, and negative alignment is removed because it would oppose that direction.

Weight modes still matter because they shape the raw gradients that lex projection receives. `kl_leaf` tends to make fine gradients larger relative to coarse gradients and can produce strong fine/higher conflict. `kl_coarse` tends to protect coarse behavior through scalar weighting, but it is not automatically safer; CIFAR-100 diverges under the completed `kl_coarse` lex run. `equal` is the cleanest neutral reference because it avoids imposing a strong scalar priority before lex projection.

The practical conclusion is that lexicographic mode and scalar weight mode answer different questions. Weight mode changes the objective being differentiated. Lex mode changes how the resulting level gradients are composed into the optimizer step. The useful thesis claim is therefore: in these Hier-COS runs, lex projection provides explicit gradient-space priority enforcement, while weight mode controls the raw level-gradient magnitudes and conflicts that the projection must manage.

## Limitations

- The analysis uses existing completed lex runs only. It does not include a complete matched non-lex grid, so it should not be framed as proof that lex improves performance over non-lex.
- The CIFAR-100 hierarchy is this repository's 3-level hierarchy, not the upstream Hier-COS 5-level CIFAR protocol.
- CUB-200-2011 Hier-COS is a local extrapolation because upstream Hier-COS does not report CUB experiments.
- All results are single-run observations under the available seed/configuration and should be treated as evidence, not final statistical conclusions.
