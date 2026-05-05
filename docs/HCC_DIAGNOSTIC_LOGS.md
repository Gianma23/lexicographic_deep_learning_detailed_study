# HCC Internal Diagnostic Logs (English Glossary)

This document explains the HCC-specific diagnostic metrics written by the training/evaluation pipeline.

## Where these metrics are logged

- `run_log.jsonl`:
  - `train_metrics` (per epoch)
  - `val_metrics` (per epoch)
- `test_metrics.yaml` (final test snapshot)

## Scope and conventions

- `pre`: `softmax(logits)` before HCC projection/blending.
- `post`: final probabilities used by probability diagnostics (`softmax(effective_logits_per_level)` for logit-space HCC, else `pre`).
- `ind`: independent decoding (argmax per level).
- `td`: top-down decoding (hierarchy-enforced decoding).
- Level naming:
  - `L0`: coarse
  - `L1`: middle
  - `L2`: fine

Notes:
- Fine-level diagnostics below are computed when at least 3 hierarchy levels are present.
- Parent-mass and sibling-rank metrics require a valid taxonomy mapping for `level=2` (`fine -> middle`).

## Metric-by-metric explanation

### 1) HCC schedule state

- `proj_constraint_alpha`
  - Meaning: blend coefficient between raw logits and projected logits for logit-space HCC.
  - Typical range: `[0, 1]`.
  - Interpretation:
    - `0`: no projection effect.
    - `1`: full hard projection effect.

- `proj_temperature`
  - Meaning: current temperature from the HCC schedule.
  - Interpretation:
    - Higher values: softer regime.
    - Lower values (near 1 in this implementation): harder regime.

### 2) Projector constraint residuals

These come from the hierarchical affine projector, using:

- `r12 = M12 * z2 - z1`
- `r23 = M23 * z3 - z2`
- Residual vector is `[r12, r23]` concatenated.

- `proj_logit_residual_before_l1`
  - Definition: `mean(abs(residual_before))`.
  - Interpretation: average logit-space constraint violation before correction.

- `proj_logit_residual_after_l1`
  - Definition: `mean(abs(residual_after))`.
  - Interpretation: average logit-space constraint violation after correction.

- `proj_logit_residual_reduction`
  - Definition: `proj_logit_residual_before_l1 - proj_logit_residual_after_l1`.
  - Interpretation:
    - Positive: projector reduced violation.
    - Near zero: little/no change.
    - Negative: correction worsened residuals (unexpected, worth checking).

### 3) Fine-level logit shift (`L2`)

- `proj_logit_delta_l1_level_2`
  - Definition: `mean(sum(abs(fine_post_logits - fine_pre_logits)))`.
  - Interpretation: overall magnitude of the fine-level logit rewrite caused by HCC.

- `proj_gt_logit_delta_level_2`
  - Definition: `mean(fine_post_logits[gt_fine] - fine_pre_logits[gt_fine])`.
  - Interpretation:
    - Positive: GT fine class logit increased after projection.
    - Negative: GT fine class logit decreased after projection.

### 4) Fine-level probability shift (`L2`)

- `proj_delta_l1_level_2`
  - Definition: `mean(sum(abs(fine_post - fine_pre)))`.
  - Range: `[0, 2]` for probability vectors.
  - Interpretation: post-softmax fine-level redistribution caused by HCC.

- `proj_flip_rate_level_2`
  - Definition: fraction of samples where `argmax(fine_pre) != argmax(fine_post)`.
  - Range: `[0, 1]`.
  - Interpretation: how often HCC changes the predicted fine class.

- `proj_gt_prob_delta_level_2`
  - Definition: `mean(fine_post[gt_fine] - fine_pre[gt_fine])`.
  - Interpretation:
    - Positive: GT fine class gained probability after projection.
    - Negative: GT fine class lost probability.

### 5) Fine accuracy conditioned on correct parent (`L1`)

The condition mask is `parent_pred == parent_target`.

Important:
- The `ind` and `td` versions use different masks, because each decoder conditions on its own parent prediction.
- So `acc_l2_ind_given_l1_correct` and `acc_l2_td_given_l1_correct` are not guaranteed to be measured on the same subset of samples.

- `acc_l2_ind_given_l1_correct`
  - Definition: fine independent accuracy on samples where independent `L1` is correct.
  - Intuition: "When independent decoding enters the right parent, how often does it still pick the right fine class?"

- `acc_l2_td_given_l1_correct`
  - Definition: fine top-down accuracy on samples where top-down `L1` is correct.
  - Intuition: "When top-down decoding enters the right parent, how often does it pick the right fine class inside that parent?"

- `support_l1_ind_correct`
  - Definition: fraction of the batch where independent `L1` prediction is correct.
  - Note: this is a ratio, not an absolute count.

- `support_l1_td_correct`
  - Definition: fraction of the batch where top-down `L1` prediction is correct.
  - Note: this is a ratio, not an absolute count.

Reading tip:
- Always read `acc_l2_*_given_l1_correct` together with `support_l1_*_correct`.
- A high conditional accuracy with very low support is not stable evidence.
- Read them in this order:
  - First ask whether support is high or low.
  - Then ask whether conditional leaf accuracy is high or low on that support set.
- Typical interpretations:
  - High support + low `acc_l2_ind_given_l1_correct`: parent is usually right, but independent fine argmax still fails.
  - High support + high `acc_l2_td_given_l1_correct`: once the hierarchy enforces the correct parent, the correct fine child is still recoverable.
  - Low support + high conditional accuracy: parent prediction is the main bottleneck.
  - Low support + low conditional accuracy: both parent and fine discrimination are weak.

### 6) GT parent mass and sibling ranking (`L2`)

- `gt_parent_mass_pre_l2`
  - Definition: mean probability mass assigned (pre) to all fine classes under the GT middle parent.
  - Range: `[0, 1]`.

- `gt_parent_mass_post_l2`
  - Definition: same as above, but post projection.
  - Interpretation: increase means more mass concentrated under the correct middle parent subtree.

- `gt_child_rank_within_parent_pre_l2`
  - Definition: mean rank of GT fine child among siblings in the GT parent, computed as:
    - `1 + (#siblings with probability > GT probability)`
  - Best value is `1` (GT is top-ranked among siblings).

- `gt_child_rank_within_parent_post_l2`
  - Definition: same rank after projection.
  - Interpretation:
    - Lower is better.
    - Higher means GT ranking among siblings worsened.

## Practical reading patterns

- Healthy hard-switch behavior:
  - `proj_constraint_alpha` rises to `1`, `proj_temperature` drops to hard regime.
  - `proj_logit_residual_after_l1` stays clearly below `proj_logit_residual_before_l1`.
  - Fine metrics do not collapse (limited `flip_rate`, stable conditional accuracies).

- Typical over-constraining pattern (observed in hard cases):
  - `gt_parent_mass_post_l2` increases.
  - `gt_child_rank_within_parent_post_l2` gets worse (higher).
  - `acc_l2_ind_given_l1_correct` drops while `acc_l2_td_given_l1_correct` stays relatively high.
  - Interpretation: hierarchy path consistency improves, but global fine discrimination degrades.
