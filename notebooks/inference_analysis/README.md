# Post-hoc inference analysis

Inference-only comparisons on frozen checkpoints. No model is retrained here: every notebook
re-reads existing checkpoints through a grid of inference rules and reports what the rule buys.

Every rule is one cell of a **readout x transform** grid:

|  | `node_score` | `subspace_norm` |
|---|---|---|
| **no transform** | rank each node by its own coordinate | rank it by the L2 norm over its ancestors+self+descendants subspace |
| **HCC projection** | `hcc_node_score` | `hcc_subspace_norm` |

Each checkpoint's own inference is one of those cells — `node_score` for H-CAST, HT-CapsNet and
HRN, `subspace_norm` for Hier-COS, and the HCC-prefixed cell for an HCC-trained checkpoint.

## The notebooks

| Notebook | Question it answers |
|---|---|
| `baseline.ipynb` | Does any cell beat what a baseline-trained checkpoint already does? |
| `lexmode.ipynb` | The same, for lexicographic (coarse-first) gradient projection. |
| `hcc.ipynb` | The same, for HCC-trained checkpoints, whose native cell is HCC-prefixed. |
| `lhprojection.ipynb` | The same, for the LH-DNN branch-point projection (Hier-COS only). |
| `subspace_supervision.ipynb` | The same, for direct subspace-norm supervision (Hier-COS only). |
| `all_mechanics.ipynb` | Which (mechanism, inference) combination is best overall, and how mechanisms compare at cells that are not their own. |

The five per-mechanism notebooks report the **within-checkpoint** gain against that checkpoint's
native cell: one set of weights, two inference rules, so the effect isolates the readout. Their
numbers are each measured against a different zero point, so they cannot be compared across
notebooks — that is what `all_mechanics.ipynb` is for.

`all_mechanics.ipynb` compares training runs against each other, which is a weaker comparison: the
two rows share only the seed, the dataset and the test split, so a difference mixes the training
mechanism with the readout, and each mechanism's checkpoint was selected using its own native
validation inference. The notebook states this next to every table that crosses mechanisms.

## Running them

Each notebook discovers its own runs, then calls `evaluation/evaluate_checkpoints.py` for any run
that has no `posthoc_inference_test_metrics.yaml` yet, writing the result into that seed's run
directory. Set `RUN_EVALUATION = True` in the setup cell to fill in missing runs, and run only one
kernel at a time: the evaluator checks for an existing output before it starts and again when it
saves, so two concurrent sweeps make the slower one fail at save time. Because the file lands in the
run directory, work done by one notebook is reused by all the others — `all_mechanics.ipynb`
defaults to `RUN_EVALUATION = False` for that reason.

Figures are written under `/scratch/g.saggini1/outputs/analysis/inference_analysis/<notebook>/`, as
PDF for LaTeX plus PNG for previews, and each call prints a `figure` environment to paste into the
thesis.

## Shared code

All six notebooks are thin: run discovery, the evaluator sweep, the loaders, the gain tables and
every figure live in [`../utils/posthoc_inference_utils.py`](../utils/posthoc_inference_utils.py).
Edit that module to change a figure everywhere; edit a notebook only to change what it selects.
