# Repository correctness and fidelity audit

Audit date: 20 July 2026  
Protocol: `corrected_unified_v1`

## Scope and method

The audit covered tracked Python, YAML, shell, Markdown, and analysis-helper
sources. Notebook files were inventoried but not rewritten. External datasets
and experiment outputs were read only.

The method combined:

- static source/config inspection;
- resolution of every runnable YAML and template schema;
- differential review against pinned H-CAST, HT-CapsNet, HRN, and Hier-COS
  sources and the LH-DNN paper;
- synthetic numerical/unit contracts;
- real metadata/image checks for all four datasets;
- shell syntax and dry-run matrix checks;
- checkpoint and metric oracles.

## Correctness findings

| Severity | Finding | Correction |
|---|---|---|
| High | Each split independently remapped labels, so the same integer could represent different classes across train/validation/test. | Added one `DatasetLabelSpace` shared by every split; unseen later-split labels fail. |
| High | Explicit missing annotation paths silently fell back to another dataset protocol. | Explicit paths are mandatory; native presets no longer contain placeholder JSON paths. |
| High | “Lexicographic” checkpoint selection packed metrics into one float, allowing a sufficiently large TICE change to override a small real FPA improvement. | Compare the exact tuple `(FPA, -TICE, weighted_AP)` and persist it in v2 checkpoints. |
| High | HRN and most Hier-COS presets dropped incomplete validation/test batches. | Corrected presets and validator require `drop_last_eval: false`. |
| Medium | CIFAR HT-CapsNet silently resumed from an output checkpoint by default. | All presets start with `train.resume: ""`; resume is explicit. |
| Medium | HT-CapsNet configs used 100 epochs while the upstream launcher uses 200. | Corrected HT-CapsNet presets use 200 epochs. |
| Medium | LH-DNN CIFAR-100 claimed paper fidelity but used 30 epochs instead of the large-network 15. | Corrected to 15 epochs. |
| Medium | Applying the CIFAR LH-DNN topology directly at 224 px created a very large flattened dense layer. | CUB/Aircraft explicitly adapt the last map to 2×2 before the shared layer. |
| Medium | HT-CapsNet could run without a complete taxonomy and lose its defining routing constraint. | Factory now requires every child at every transition to have a valid parent. |
| Medium | CUB validation ratios and split seeds differed by model family. | Corrected presets use CUB ratio 0.15, CIFAR ratio 0.10, and split seed 0. |
| Medium | Hier-COS baseline docs claimed `kl_reg`, while presets actually selected a local decomposed CE loss. | Baseline presets now use `kl_reg`; lex launchers explicitly override decomposed loss modes. |
| Low | Launcher comments advertised matrices that hard-coded arrays could not reach. | Matrices are validated environment lists; actual narrow defaults are printed. |
| Low | README and file documentation referenced nonexistent parity configs, templates, TODO, and notebooks. | Documentation was rebuilt from the current tree and protected by path tests. |

## Dataset evidence

Real adapters were constructed without modifying source data:

| Dataset | Train | Validation | Test | Class counts |
|---|---:|---:|---:|---|
| CIFAR-100 | 45,000 | 5,000 | 10,000 | 8 / 20 / 100 |
| CUB-200-2011 | 5,194 | 800 | 5,794 | 13 / 38 / 200 |
| FGVC-Aircraft | 3,334 | 3,333 | 3,333 | 30 / 70 / 100 |

For each dataset, validation/test construction reused training metadata,
taxonomy sizes matched, and a transformed sample was loaded from every split.

## Fidelity classification

| Family | Classification | Key remaining limitation |
|---|---|---|
| H-CAST | source-aligned core plus verified Aircraft correction and local HCC/lex extensions | vendored shape guards prevent byte-identical upstream code |
| LH-DNN | paper-derived CIFAR implementation | no official code; CUB/Aircraft are extrapolations |
| HT-CapsNet | source-aligned TensorFlow-to-PyTorch port with Keras-shaped SDPA attention | framework numerical equivalence, unified CUB taxonomy, and Aircraft extrapolation |
| HRN | source-aligned full-label architecture/loss | unified validation/full-eval protocol differs from upstream |
| Hier-COS | source-aligned fixed-frame/KL core plus local CE/lex modes | local hierarchy depths and ambiguous upstream Aircraft script |

Detailed evidence is in the
[model fidelity and divergence log](model_repo_differences.md).

## Verification status

Required automated checks:

```bash
python -m unittest discover -s tests -v
python -m compileall -q datasets models train gridsearch notebooks scripts docs
for f in scripts/*.sh scripts/*/*.sh; do bash -n "$f"; done
git diff --check
```

The final implementation handoff records the exact test count and any skipped
checks. A working NVIDIA device was not visible during the audit
(`nvidia-smi` could not communicate with a driver), so representative
one-batch GPU contracts remain environment-dependent. No long CPU fallback
training is substituted for that missing GPU evidence.

## Interpretation boundary

This audit demonstrates contract coverage and removes identified silent
correctness failures. It does not prove that every model reaches a published
score, that local extrapolation hyperparameters are optimal, or that results
from different protocol tags are directly comparable.
