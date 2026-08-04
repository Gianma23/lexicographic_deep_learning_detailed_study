# Checkpoint-only inference evaluation

This package evaluates existing checkpoints without constructing a loss,
optimizer, scheduler, or training loop.

```bash
python -m evaluation.evaluate_checkpoints \
  --run-dir /scratch/g.saggini1/outputs/<experiment>/seed_0 \
  --inference-mode all \
  --checkpoint-mode both
```

## The inference grid

An inference rule is a **readout** optionally preceded by a **transform**:

| | `node_score` | `subspace_norm` |
|---|---|---|
| **no transform** | `node_score` | `subspace_norm` |
| **HCC projection** | `hcc_node_score` | `hcc_subspace_norm` |

- **`node_score`** ranks each taxonomy node by its own coordinate.
- **`subspace_norm`** ranks it by the L2 norm over its ancestors+self+descendants
  subspace.
- **`hcc_`** applies the HCC affine hierarchy projection
  (`HierarchicalAffineProjector`) to the node coordinates before the readout.

Both readouts consume the same per-level node coordinates: the fixed-layer
`node_logits` for native Hier-COS, and the native per-level scores for
classifier-head models. For the latter, `subspace_norm` treats those per-level
scores as coordinates in an identity taxonomy frame — an inference-only
assumption.

`--inference-mode all` evaluates all four cells; `both` evaluates the two
untransformed readouts. Every cell is defined for every model, so no combination
is rejected. All cells are scored from one shared forward pass.

## Each model's own inference is one cell

The tool does not have a separate "native" rule: it recomputes the checkpoint's
own readout as one grid cell and uses it as the paired reference.

| Model | Native cell |
|---|---|
| H-CAST, HRN, LH-DNN, HT-CapsNet | `node_score` |
| Native Hier-COS | `subspace_norm` |
| Any of the above trained with `hcc.enabled: true` | the `hcc_`-prefixed cell of the same readout |

`native_inference_mode` records this per run, and every other evaluated cell gets
a `<cell>_minus_<native>` delta block.

Reproduction is exact for H-CAST (the same tensor) and Hier-COS (the same
subspace norms from the same masks), and metric-exact for HRN and the other
classifier heads: `node_score` reads the raw logits rather than the sigmoid or
softmax the model applies, and those are monotone per level, so every reported
metric is unchanged because all of them are computed from argmax decoding.

## Sign conventions

`node_score` reads a **signed** value for classifier logits, where a large
negative logit means *not this class*, and the **magnitude** for Hier-COS frame
coordinates, because Hier-COS trains `log_softmax(|node_logits|)` and its
objective never constrains a coordinate's sign. In both cases the rule ranks by
the quantity that model's training objective drives up for the correct node.
`coordinate_evidence` in the payload records which convention was used.

`subspace_norm` has no such choice: an L2 norm squares its inputs, so it always
discards the sign, including for signed classifier logits. That is the
substantive content of the identity-frame assumption.

## Properties worth knowing before reading the deltas

- **`hcc_node_score` cannot move top-down metrics for signed readouts.** The
  correction is `z_hat = z - coeff @ M`, one constant per parent broadcast to
  that parent's children, and top-down decoding is a masked argmax within the
  predicted parent's children. Sibling rankings therefore survive, and the coarse
  level is anchored outright. Only independent decoding can move. This is a
  property of the rule, not an empirical result. It does **not** hold for
  Hier-COS, where `abs` is not monotone in the coordinate. (The one edge case is
  a predicted parent with no children, where `_hierarchical_argmax_preds` falls
  back to an unmasked argmax; the dataset adapters build taxonomies from observed
  labels, so it does not arise for CIFAR-100, CUB, or Aircraft.)
- **`hcc_subspace_norm` moves both decoders for every model**, since the norms
  mix levels.
- **`hcc_` cells always project at `alpha=1`** and do not replicate a run's
  activation schedule. For a run trained with HCC, compare against
  `model_hcc_diagnostics.proj_constraint_alpha`, which records the alpha the
  model's own forward pass applied at test time.
- **HRN's native level-2 score is its auxiliary leaf CE head**, not the sigmoid
  tree branch that HCC corrects during training. The `hcc_` cells project what
  the readout consumes, so on HRN they are not a reconstruction of its
  training-time HCC path. Note separately that `evaluate_batch` scores HRN from
  `effective_probs_per_level`, built from raw sigmoids, so training-time HCC
  never reaches HRN's metrics at all.

## Legacy mode names

The previous flat names still work on the command line and resolve per model:

| Legacy | Resolves to |
|---|---|
| `normal` | `subspace_norm` for Hier-COS, `node_score` otherwise |
| `hiercos` | `subspace_norm` |
| `node_softmax` | `node_score` |
| `hcc` | `hcc_subspace_norm` for Hier-COS, `hcc_node_score` otherwise |

Reading a YAML written before the rename needs the other direction, which is not
a clean inverse — `normal` and `node_softmax` both resolve to `node_score`.
`legacy_mode_name()` spells it out, and `legacy_mode_names` in the payload
records the mapping for that run:

| Cell | H-CAST / HRN | Hier-COS | H-CAST trained with HCC |
|---|---|---|---|
| `node_score` | `normal` | `node_softmax` | — |
| `subspace_norm` | `hiercos` | `normal` | `hiercos` |
| `hcc_node_score` | `hcc` | — | `normal` |
| `hcc_subspace_norm` | — | `hcc` | — |

`normal` always named whatever the checkpoint did natively, which is why it moves
with the model and with `hcc.enabled`. Dashes are cells the old scheme could not
express. One detail of the remap: the old `node_softmax` applied a global
`softmax(|node_logits|)`, whose ordering — and therefore every reported metric —
matches `node_score`.

## Cost

All selected cells are scored from one shared forward pass, so adding a cell
costs only its readout and its metrics, never another pass over the test set.
On Aircraft the forward pass is a small part of the total; the metric
computation dominates. Two things keep it cheap, and both must stay in place:

- `train/metrics.py` caches the per-level `allowed` mask and child→parent
  lookup. Rebuilding them element by element on the accelerator each batch cost
  51 ms per top-down decode versus 0.11 ms cached.
- `evaluate_batch` decodes once per decoder and passes the predictions to every
  metric, instead of each metric decoding again, and this CLI passes
  `include_diagnostics=False` because `_outcome_metrics` discards the level-3
  diagnostics anyway. Training keeps them, since they are logged.

Together these took the four-cell evaluation from 45 s to 2.1 s per 20 batches
of 64 images. Skipping a cell whose numbers already exist in the run's
`test_metrics.yaml` would save little by comparison, and is unsound for any run
whose `test_split_source` is the adapter fallback.

## Output contract

By default the CLI evaluates both validation-selected checkpoints and preserves
the repository rule that the top-down row comes from `best_topdown.pt` while the
independent row comes from `best_independent.pt`.

The CLI treats `config_resolved.yaml` as an immutable run artifact and does not
apply the current training-entrypoint policy validator. Historical checkpoints
may contain generated provenance fields or protocol settings that are no
longer accepted for new training runs. It constructs only the test loader,
forces `drop_last_eval=false` so every model sees its complete test split, and
falls back to the official dataset adapter if a legacy configured test manifest
no longer exists. These in-memory changes are recorded under
`evaluation_config_adjustments`; no saved config is rewritten.

**The annotation fallback can invalidate a run's numbers.** If the configured
manifest is gone, the adapter may rebuild a different label space than the run
trained on, which surfaces as plausible-looking but meaningless parent-level
metrics rather than as an error. Observed on
`hcast_hcc_aircraft_step_80epoch_nokl/seed_0`: the same checkpoint scores 88.4%
coarse in its own `test_metrics.yaml` and 9.6% here, while leaf accuracy matches
(73.0% vs 72.9%) because only the parent levels are remapped. The CLI prints a
warning and sets `test_split_source: official_dataset_adapter_fallback`. When
that appears, check the native row against the run's own `test_metrics.yaml`
before using any row from that file.

The default output is `posthoc_inference_test_metrics.yaml` in the run
directory. Existing output is not replaced unless `--overwrite` is provided.

Use `notebooks/posthoc_hiercos_inference_comparison.ipynb` to invoke this CLI
for every completed seed of the H-CAST/HRN/Hier-COS baselines across all three
datasets. Each YAML remains in the corresponding seed run directory. The
notebook aggregates both absolute metrics and paired inference gains using the
mean, sample standard deviation (`ddof=1`), and seed count.
