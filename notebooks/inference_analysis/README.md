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
| `all_mechanics.ipynb` | Which (mechanism, inference) combination is best overall, and how mechanisms compare at cells that are not their own. |

The four per-mechanism notebooks report the **within-checkpoint** gain against that checkpoint's
native cell: one set of weights, two inference rules, so the effect isolates the readout. Their
numbers are each measured against a different zero point, so they cannot be compared across
notebooks — that is what `all_mechanics.ipynb` is for.

Direct subspace supervision remains available as a family in `all_mechanics.ipynb`, but it has no
dedicated per-mechanism notebook.

`all_mechanics.ipynb` compares training runs against each other, which is a weaker comparison: the
two rows share only the seed, the dataset and the test split, so a difference mixes the training
mechanism with the readout, and each mechanism's checkpoint was selected using its own native
validation inference. The notebook states this next to every table that crosses mechanisms. Its
soft-readout-against-top-down section is the exception: both sides of every bar there come from the
same training run, so that comparison is within-checkpoint like the per-mechanism notebooks.

Every notebook asks, in a section of its own, whether the soft `subspace_norm` readout under
independent decoding already does what hard top-down decoding does. Both readouts come from one
training run, so the comparison isolates the readout against the decoder; only checkpoint selection
differs, since each side is read from its own validation-selected checkpoint.

## Running them

Each notebook discovers its own runs, then calls `evaluation/evaluate_checkpoints.py` for any run
that has no `posthoc_inference_test_metrics.yaml` yet, writing the result into that seed's run
directory. Set `RUN_EVALUATION = True` in the setup cell to fill in missing runs, and run only one
kernel at a time: the evaluator checks for an existing output before it starts and again when it
saves, so two concurrent sweeps make the slower one fail at save time. Because the file lands in the
run directory, work done by one notebook is reused by all the others — `all_mechanics.ipynb`
defaults to `RUN_EVALUATION = False` for that reason.

Every notebook exposes `RUN_ROOTS` in its setup cell. A per-mechanism notebook maps
`dataset -> model -> experiment directory`; `all_mechanics.ipynb` uses the corresponding full
`family -> dataset -> model -> experiment directory` mapping. A directory may be an absolute path
or a name relative to `/scratch/g.saggini1/outputs`. This makes it possible to switch among
automatic-width, `d512`, `equal`, `kl_leaf`, or future runs without editing the shared utility.

After changing `RUN_ROOTS`, re-run the discovery cell and every cell below it. The discovery cell
prints every selected root and whether it exists before reporting completed/evaluated coverage, so
a missing or incomplete selection cannot silently disappear from the analysis.

Figures are written under `/scratch/g.saggini1/outputs/analysis/inference_analysis/<notebook>/`, as
PDF for LaTeX plus PNG for previews. Plotting calls print the saved paths but do not emit LaTeX
figure environments.

## Shared code

All five notebooks are thin: run discovery, the evaluator sweep, the loaders, the gain tables and
every figure live in [`../utils/posthoc_inference_utils.py`](../utils/posthoc_inference_utils.py).
Edit that module to change a figure everywhere; edit a notebook only to change what it selects.
