"""One print style for every figure that goes into the thesis.

Figures are authored at their final printed width and must be included with
``width=\\linewidth`` and no extra scaling: any additional resizing in LaTeX
shrinks the font sizes below along with the artwork, so a figure authored here
and scaled there no longer matches the body text.

The values were first written in ``notebooks/datasets_analysis.ipynb`` and then
copied into ``notebooks/utils/current_run_plot_utils.py``. This module is the
single definition both of those now read, so the dataset figures, the trade-off
figures and the model-analysis figures sit side by side in the document without
a visible style break.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
THESIS_DOCS_DIR = REPO_ROOT / "docs"


# The thesis text block, from the `geometry` options in main.tex:
#
#     a4paper, top=3cm, bottom=3cm, inner=3cm, outer=2.5cm, bindingoffset=0.5cm
#
# The binding offset is added to the inner margin, so
#     width  = 210 - 30 - 25 - 5 = 150 mm = 5.906 in
#     height = 297 - 30 - 30     = 237 mm = 9.331 in
# Confirm with \the\textwidth in the real document if the geometry changes;
# 150 mm prints as 426.79 pt at 1 pt = 1/72.27 in.
TEXT_WIDTH_IN = 5.906
HALF_WIDTH_IN = 2.85
TEXT_HEIGHT_IN = 9.331

# Usable height for one figure: the text block minus room for a caption, which
# at 12 pt with \onehalfspacing runs about 0.75 in for three lines. A grid
# taller than this is split across consecutive figures rather than squeezed.
PAGE_HEIGHT_IN = 8.4

# Height of a single panel row. The title, tick labels and (on a bottom row) the
# axis label all come out of this budget, so the drawn axes are roughly 0.5 in
# shorter than the number here.
#
# Panels are laid out two to a row by default, as most papers do, which is what
# lets a row be this tall: a half-width panel ends up near a 3:2 aspect ratio
# instead of the wide strip a full-width row has to settle for to fit the page.
# The one-column value is the fallback for a section with a single panel.
PANEL_HEIGHT_IN = 1.95
PANEL_HEIGHT_2COL_IN = 2.1

# The document sets `fontsize=12pt`, where LaTeX's own steps are
# \footnotesize = 10 pt and \small = 10.95 pt.
#
# The title and the legend are read, so they sit at about \footnotesize. Tick
# labels and axis labels are reference furniture consulted only when a number
# matters, so they sit lower - and in a half-width panel they are also what the
# axes area is paying for, since every point of tick label is a point of gutter.
PAPER_RCPARAMS = {
    "figure.dpi": 140, "savefig.dpi": 400,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 8, "legend.title_fontsize": 8,
    "axes.axisbelow": True, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.constrained_layout.use": True,
}


# Dataset keys as they come out of the run logs, and how they are written in the
# thesis. The slugs match the run-directory naming used by scripts/.
DATASET_SHORT_NAMES = {
    "cifar-100": "CIFAR-100",
    "cub-200-2011": "CUB-200",
    "fgvc-aircraft": "Aircraft",
}
DATASET_SLUGS = {
    "cifar-100": "cifar100",
    "cub-200-2011": "cub200",
    "fgvc-aircraft": "aircraft",
}

# Hierarchy levels, as the dataset adapters and the resolved configs write them,
# and as the thesis writes them.
#
# `mid` is the gradient-diagnostic log key and `middle` is the thesis spelling.
# CIFAR-100's adapter names its hierarchy `coarse1/coarse2/fine`, which reads as
# two sibling coarse levels rather than a chain, so the figures write
# `coarse/middle/fine` instead. CUB-200 (`order/family/species`) and Aircraft
# (`manufacturer/family/variant`) already name their levels after the taxonomy
# and pass through unchanged.
LEVEL_DISPLAY_NAMES = {
    "coarse": "coarse",
    "coarse1": "coarse",
    "coarse2": "middle",
    "mid": "middle",
    "middle": "middle",
    "fine": "fine",
}

# Training-loss keys as the run logs write them, and how a figure names them.
# The keys come from each family's loss module (`models/<family>/losses.py`);
# where a family writes two names for the same quantity both are listed, so the
# figure never depends on which alias a run happened to log.
LOSS_DISPLAY_NAMES = {
    "total": "Total objective",
    # H-CAST and LH-DNN: the summed level cross-entropies, plus H-CAST's
    # coupled global KL term over the whole hierarchy.
    "level_ce": "Summed level cross-entropy",
    "gk_loss": "Global KL term",
    # HT-CapsNet: the summed level capsule margin losses.
    "margin": "Summed capsule margin loss",
    # HRN: the tree loss over the hierarchy and the separate leaf
    # cross-entropy. `hier_loss`/`tree_loss` and `ce_loss_leaf`/`fine_ce` are
    # aliases of one quantity each.
    "tree_loss": "Hierarchical tree loss",
    "hier_loss": "Hierarchical tree loss",
    "ce_loss_leaf": "Leaf cross-entropy",
    "fine_ce": "Leaf cross-entropy",
    # Hier-COS: the node-wide KL objective, the decomposed classification term
    # that replaces it, and the coordinate regulariser.
    "kl": "Node-wide KL term",
    "ce": "Classification term",
    "reg": "Regularisation term",
    # Subspace supervision.
    "subspace_soft_cross_entropy": "Subspace soft cross-entropy",
    "subspace_target_kl": "Subspace target KL",
    "subspace_score_l2": "Subspace score norm",
    "aux_loss": "Auxiliary term",
}


def use_paper_style() -> None:
    """Apply the thesis figure style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(PAPER_RCPARAMS)


def dataset_display_name(dataset_key) -> str:
    key = str(dataset_key)
    return DATASET_SHORT_NAMES.get(key, key)


def dataset_slug(dataset_key) -> str:
    key = str(dataset_key)
    return DATASET_SLUGS.get(key, key.replace("-", "").replace("_", ""))


def level_display_name(level_key) -> str:
    key = str(level_key)
    return LEVEL_DISPLAY_NAMES.get(key, key)


# Levels are numbered from the root in docs/04-methodology.tex: `l = 1, ..., L`
# with `l = 1` the coarsest and highest-priority level, and `\ell_l` its level
# objective. The logs and the code index the same levels from zero, so a display
# label adds one. `L0/L1/L2` is defined nowhere in the thesis and reads as a
# norm rather than a depth, so it is never used as a level name in a figure.
def level_index_symbol(level_idx) -> str:
    """The level's methodology name: `$l=1$` for the coarsest level."""
    return f"$l={int(level_idx) + 1}$"


def level_loss_symbol(level_idx) -> str:
    """The level objective's methodology name: `$\\ell_1$` for the coarsest."""
    return f"$\\ell_{{{int(level_idx) + 1}}}$"


def loss_display_name(loss_key) -> str:
    """How a figure names one training-loss key.

    Unmapped keys are de-underscored rather than dropped: a family can log a
    term these notebooks have not seen, and a readable fallback is better than
    hiding the panel.
    """
    key = str(loss_key)
    if key in LOSS_DISPLAY_NAMES:
        return LOSS_DISPLAY_NAMES[key]
    return key.replace("_", " ").strip().capitalize()


def legend_height_in(nrow) -> float:
    """Vertical inches a bottom legend needs, so panels can be sized around it."""
    return 0.0 if nrow <= 0 else 0.22 + 0.155 * nrow


def grid_page_rows(ncols: int, n_legend_rows: int, panel_height: float) -> int:
    """How many panel rows fit on one page once the legend has taken its space."""
    available = PAGE_HEIGHT_IN - legend_height_in(n_legend_rows)
    return max(1, int(available // max(panel_height, 1e-6)))


def save_figure(fig, figure_dir, stem, formats=("pdf", "png")):
    """Write a figure at its authored size and report the geometry.

    ``bbox_inches='tight'`` is deliberately not used: it re-crops the canvas and
    changes the physical width, which would break the one-to-one relationship
    between the authored figure size and the printed text width.

    Runtime output directories retain every requested format. A destination
    below ``docs/`` is a thesis asset directory and receives PDF only.

    Returns ``(stem, width_in, height_in)`` so a caller can assert that nothing
    drifted off the page.
    """
    width_in, height_in = (float(value) for value in fig.get_size_inches())
    if figure_dir is not None:
        figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        try:
            figure_dir.resolve().relative_to(THESIS_DOCS_DIR.resolve())
            selected_formats = ("pdf",)
        except ValueError:
            selected_formats = formats
        for suffix in selected_formats:
            fig.savefig(figure_dir / f"{stem}.{suffix}")
    return (str(stem), width_in, height_in)


def latex_block(stem, caption, label=None, width=r"\linewidth", subdir="") -> None:
    """Print the figure environment to paste into the thesis."""
    label = label or str(stem).replace("_", "-")
    path = f"{subdir.rstrip('/')}/{stem}" if subdir else str(stem)
    print("\n% ---- LaTeX ----------------------------------------------------")
    print("\\begin{figure}[tbp]\n  \\centering")
    print(f"  \\includegraphics[width={width}]{{images/{path}.pdf}}")
    print(f"  \\caption{{{caption}}}")
    print(f"  \\label{{fig:{label}}}\n\\end{{figure}}")


def report_exported(records, page_height=PAGE_HEIGHT_IN, width=TEXT_WIDTH_IN) -> None:
    """Print every saved figure with its geometry and flag anything off-page."""
    if not records:
        print("No figures were exported.")
        return
    print(f"{len(records)} figures exported (authored size, in inches):")
    for stem, width_in, height_in in records:
        flags = []
        if not math.isclose(width_in, width, abs_tol=0.02):
            flags.append(f"WIDTH != {width}")
        if height_in > page_height + 0.02:
            flags.append(f"TALLER THAN {page_height} in")
        note = f"   <-- {', '.join(flags)}" if flags else ""
        print(f"  {stem:<52s} {width_in:.2f} x {height_in:.2f}{note}")
