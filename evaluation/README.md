# Checkpoint-only inference evaluation

This package evaluates existing checkpoints without constructing a loss,
optimizer, scheduler, or training loop.

```bash
python -m evaluation.evaluate_checkpoints \
  --run-dir /scratch/g.saggini1/outputs/<experiment>/seed_0 \
  --inference-mode both \
  --checkpoint-mode both
```

`--inference-mode both` resolves by model family:

| Model | Reference inference | Distance inference |
|---|---|---|
| H-CAST, HRN, and other non-Hier-COS models | `normal` | `hiercos` |
| Native Hier-COS | `node_softmax` | `normal` |

For non-Hier-COS models, `hiercos` concatenates native raw per-level logits,
assumes an identity taxonomy frame, computes ancestors+self+descendants
projection norms, and predicts by raw-score argmax. For native Hier-COS,
`node_softmax` selects each level's values directly from one global
`softmax(abs(node_logits))`, without constructing projection-norm scores.

Both inference rules are evaluated from the same forward output. By default,
the CLI evaluates both validation-selected checkpoints and preserves the
repository rule that the top-down row comes from `best_topdown.pt` while the
independent row comes from `best_independent.pt`.

The CLI treats `config_resolved.yaml` as an immutable run artifact and does not
apply the current training-entrypoint policy validator. Historical checkpoints
may contain generated provenance fields or protocol settings that are no
longer accepted for new training runs. It constructs only the test loader,
forces `drop_last_eval=false` so every model sees its complete test split, and
falls back to the official dataset adapter if a legacy configured test manifest
no longer exists. These in-memory changes are recorded under
`evaluation_config_adjustments`; no saved config is rewritten.

The default output is `posthoc_inference_test_metrics.yaml` in the run
directory. Existing output is not replaced unless `--overwrite` is provided.

Use `notebooks/posthoc_hiercos_inference_comparison.ipynb` to invoke this CLI
for every completed seed of the H-CAST/HRN/Hier-COS baselines across all three
datasets. Each YAML remains in the corresponding seed run directory. The
notebook aggregates both absolute metrics and paired inference gains using the
mean, sample standard deviation (`ddof=1`), and seed count.
