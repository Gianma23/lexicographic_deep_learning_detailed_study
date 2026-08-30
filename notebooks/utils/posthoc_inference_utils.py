"""Shared machinery for the post-hoc inference notebooks.

The notebooks under ``notebooks/inference_analysis`` all ask the same question of
frozen checkpoints: **what does the inference rule buy, once training is over?**
Every rule is one cell of a readout x transform grid,

===================  ===============================  ==============================
                     ``node_score``                   ``subspace_norm``
===================  ===============================  ==============================
no transform         rank a node by its coordinate    rank it by the L2 norm over
                                                      ancestors+self+descendants
HCC projection       ``hcc_node_score``               ``hcc_subspace_norm``
===================  ===============================  ==============================

and every checkpoint's own inference is one of those cells, recorded per run in
``posthoc_inference_test_metrics.yaml`` by ``evaluation.evaluate_checkpoints``.

One notebook per *training mechanism* (family) asks whether any cell beats that
mechanism's native cell, which is a within-checkpoint paired comparison. The
``all_mechanics`` notebook loads several families at once and compares them at
*any* cell, e.g. HCC under ``hcc + subspace`` against the baseline under
``subspace``. That comparison is not within-checkpoint: it pairs two different
training runs by seed only, so it mixes the training mechanism with the readout.
Every function that crosses families says so in its docstring, and the notebook
prints the caveat next to the table.

Module-level settings (``THESIS_STYLE``, ``TEXT_WIDTH_IN``, ...) are read at call
time, so a notebook can override them with ``posthoc.THESIS_STYLE = False``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from multiseed_utils import discover_seed_dirs, sample_stats


# --------------------------------------------------------------------------
# repository layout
# --------------------------------------------------------------------------
def resolve_repo_root(start: Path | None = None) -> Path:
    """The repository root, found from this file or from ``start``."""
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / 'evaluation').is_dir() and (candidate / 'configs').is_dir():
            return candidate
    raise RuntimeError('Could not locate the repository root from ' + str(here))


REPO_ROOT = resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate_checkpoints import (  # noqa: E402  (needs sys.path above)
    canonical_inference_rule,
    native_inference_rule,
)

OUTPUTS_ROOT = Path('/scratch/g.saggini1/outputs')
RESULT_FILENAME = 'posthoc_inference_test_metrics.yaml'
DATASETS = ('cifar100', 'cub200', 'aircraft')
DECODERS = ('independent', 'topdown')
LOWER_IS_BETTER = ('tice', 'ahd')


# --------------------------------------------------------------------------
# training mechanisms (families) and their run directories
# --------------------------------------------------------------------------
FAMILY_ORDER = ('baseline', 'lexmode', 'hcc', 'lhprojection', 'subspace_supervision')
FAMILY_LABELS = {
    'baseline': 'Baseline',
    'lexmode': 'Lexicographic',
    'hcc': 'HCC',
    'lhprojection': 'LH-projection',
    'subspace_supervision': 'Subspace supervision',
}
FAMILY_SHORT = {
    'baseline': 'base',
    'lexmode': 'lex',
    'hcc': 'hcc',
    'lhprojection': 'lh-proj',
    'subspace_supervision': 'subsp-sup',
}
# One line each, printed by the notebooks so the mechanism is stated with its results.
FAMILY_DESCRIPTIONS = {
    'baseline': 'plain hierarchical training, no gradient-space or output-space constraint',
    'lexmode': 'lexicographic gradient projection, coarse-first',
    'hcc': 'Hierarchical Constraint Cascade in the output space during training',
    'lhprojection': 'LH-DNN backward-only projection at the branch points (Hier-COS only)',
    'subspace_supervision': 'direct supervision of the taxonomy-subspace norm (Hier-COS only)',
}


def experiment_roots(outputs_root: Path = OUTPUTS_ROOT) -> dict:
    """``{family: {dataset: {model: run_root}}}`` for every mechanism.

    Paths are the experiment directories that hold ``seed_<n>`` subdirectories.
    LH-projection and subspace supervision are Hier-COS-only by construction.
    """
    return {
        'baseline': {
            'cifar100': {
                'hcast': outputs_root / 'hcast_cifar100',
                'htcapsnet': outputs_root / 'capsnet_cifar100',
                'hrn': outputs_root / 'hrn_cifar100',
                'hiercos': outputs_root / 'hiercos_cifar100_global_softmax_ce_reg_baseline_kl_leaf',
            },
            'cub200': {
                'hcast': outputs_root / 'hcast_cub200',
                'htcapsnet': outputs_root / 'capsnet_cub200',
                'hrn': outputs_root / 'hrn_cub200',
                'hiercos': outputs_root / 'hiercos_cub200_global_softmax_ce_reg_baseline_kl_leaf',
            },
            'aircraft': {
                'hcast': outputs_root / 'hcast_aircraft',
                'htcapsnet': outputs_root / 'capsnet_aircraft',
                'hrn': outputs_root / 'hrn_aircraft',
                'hiercos': outputs_root / 'hiercos_aircraft_global_softmax_ce_reg_baseline_kl_leaf',
            },
        },
        'lexmode': {
            'cifar100': {
                'hcast': outputs_root / 'hcast_cifar100_lex_coarse_first',
                'htcapsnet': outputs_root / 'ht_capsnet_cifar100_lex_coarse_first',
                'hrn': outputs_root / 'hrn_cifar100_level_marginal_lex_coarse_first',
                'hiercos': outputs_root / 'hiercos_cifar100_global_softmax_ce_reg_lex_coarse_first_kl_leaf',
            },
            'cub200': {
                'hcast': outputs_root / 'hcast_cub200_lex_coarse_first',
                'htcapsnet': outputs_root / 'ht_capsnet_cub200_lex_coarse_first',
                'hrn': outputs_root / 'hrn_cub200_level_marginal_lex_coarse_first',
                'hiercos': outputs_root / 'hiercos_cub200_global_softmax_ce_reg_lex_coarse_first_kl_leaf',
            },
            'aircraft': {
                'hcast': outputs_root / 'hcast_aircraft_lex_coarse_first',
                'htcapsnet': outputs_root / 'ht_capsnet_aircraft_lex_coarse_first',
                'hrn': outputs_root / 'hrn_aircraft_level_marginal_lex_coarse_first',
                'hiercos': outputs_root / 'hiercos_aircraft_global_softmax_ce_reg_lex_coarse_first_kl_leaf',
            },
        },
        'hcc': {
            'cifar100': {
                'hcast': outputs_root / 'hcast_cifar100_hcc',
                'htcapsnet': outputs_root / 'capsnet_cifar100_hcc',
                'hrn': outputs_root / 'hrn_cifar100_level_marginal_hcc',
                'hiercos': outputs_root / 'hiercos_cifar100_global_softmax_ce_reg_hcc',
            },
            'cub200': {
                'hcast': outputs_root / 'hcast_cub200_hcc',
                'htcapsnet': outputs_root / 'capsnet_cub200_hcc',
                'hrn': outputs_root / 'hrn_cub200_level_marginal_hcc',
                'hiercos': outputs_root / 'hiercos_cub200_global_softmax_ce_reg_hcc',
            },
            'aircraft': {
                'hcast': outputs_root / 'hcast_aircraft_hcc',
                'htcapsnet': outputs_root / 'capsnet_aircraft_hcc',
                'hrn': outputs_root / 'hrn_aircraft_level_marginal_hcc',
                'hiercos': outputs_root / 'hiercos_aircraft_global_softmax_ce_reg_hcc',
            },
        },
        'lhprojection': {
            dataset: {
                'hiercos': outputs_root / (
                    f'hiercos_{dataset}_level_softmax_ce_reg_projection_kl_leaf_identity'
                ),
            }
            for dataset in DATASETS
        },
        'subspace_supervision': {
            dataset: {'hiercos': outputs_root / f'hiercos_{dataset}_subspace'}
            for dataset in DATASETS
        },
    }


def completed_seed_dirs(root: Path):
    """Seed directories under ``root`` that hold both selected checkpoints."""
    seed_dirs = discover_seed_dirs(root, require_log=False, require_config=True)
    candidates = seed_dirs or ([root] if root.is_dir() else [])
    required = ('config_resolved.yaml', 'best_topdown.pt', 'best_independent.pt')
    return [path for path in candidates if all((path / name).is_file() for name in required)]


def _family_names(families):
    return [families] if isinstance(families, str) else list(families)


def resolve_run_roots(families, roots=None, outputs_root=OUTPUTS_ROOT):
    """Normalize a notebook's configurable run-root mapping.

    ``roots`` may have the full ``{family: {dataset: {model: root}}}`` shape or,
    when exactly one family is requested, the shorter
    ``{dataset: {model: root}}`` shape. A relative root is resolved underneath
    ``outputs_root``; an absolute root is preserved.
    """
    families = _family_names(families)
    configured = experiment_roots(outputs_root) if roots is None else roots
    if len(families) == 1 and families[0] not in configured:
        configured = {families[0]: configured}

    unknown = [family for family in families if family not in configured]
    if unknown:
        raise ValueError(
            f'No run roots configured for {unknown}; available families: '
            f'{list(configured)}.'
        )

    outputs_root = Path(outputs_root)
    resolved = {}
    for family in families:
        resolved[family] = {}
        for dataset, model_roots in configured[family].items():
            resolved[family][dataset] = {}
            for model, root in model_roots.items():
                path = Path(root)
                resolved[family][dataset][model] = (
                    path if path.is_absolute() else outputs_root / path
                )
    return resolved


def run_roots_table(families, datasets=DATASETS, roots=None, outputs_root=OUTPUTS_ROOT):
    """One row per configured run root, including roots that do not exist yet."""
    families = _family_names(families)
    resolved = resolve_run_roots(families, roots=roots, outputs_root=outputs_root)
    rows = []
    for family in families:
        for dataset in datasets:
            model_roots = resolved[family].get(dataset, {})
            if not model_roots:
                rows.append({
                    'family': family,
                    'dataset_slug': dataset,
                    'model': None,
                    'run_root': None,
                    'root_exists': False,
                })
                continue
            for model, root in model_roots.items():
                rows.append({
                    'family': family,
                    'dataset_slug': dataset,
                    'model': model,
                    'run_root': str(root),
                    'root_exists': root.is_dir(),
                })
    return pd.DataFrame(
        rows,
        columns=('family', 'dataset_slug', 'model', 'run_root', 'root_exists'),
    )


def discover_runs(families, datasets=DATASETS, outputs_root=OUTPUTS_ROOT, verbose=True,
                  roots=None):
    """One row per completed run directory, for every requested family.

    Returns a frame with ``family``, ``dataset_slug``, ``model``, ``run_dir`` and
    ``has_result`` (whether the post-hoc YAML already exists). ``model`` is the
    directory-level key (``htcapsnet``); the loader later uses the model name the
    evaluator writes into the YAML (``ht_capsnet``).

    ``roots`` follows :func:`resolve_run_roots`: a single-family notebook may
    pass its shorter dataset/model mapping, while a multi-family notebook passes
    the full mapping. Relative roots are interpreted underneath ``outputs_root``.
    """
    families = _family_names(families)
    roots = resolve_run_roots(families, roots=roots, outputs_root=outputs_root)

    rows = []
    for family in families:
        for dataset in datasets:
            model_roots = roots[family].get(dataset, {})
            if not model_roots and verbose:
                print(f'[missing config] {family}/{dataset}: no run root configured')
            for model, root in model_roots.items():
                paths = completed_seed_dirs(root)
                if not paths and verbose:
                    print(f'[missing] {family}/{dataset}/{model}: no completed run under {root}')
                for path in paths:
                    rows.append({
                        'family': family,
                        'dataset_slug': dataset,
                        'model': model,
                        'run_root': root,
                        'run_dir': path,
                        'has_result': (path / RESULT_FILENAME).is_file(),
                    })
    frame = pd.DataFrame(
        rows,
        columns=('family', 'dataset_slug', 'model', 'run_root', 'run_dir', 'has_result'),
    )
    if frame.empty:
        print('No completed runs found. Check the configured roots against the output directories.')
    return frame


def coverage_table(run_table):
    """Runs and already-evaluated runs per (family, dataset, model)."""
    if run_table.empty:
        return pd.DataFrame(
            columns=('family', 'dataset_slug', 'model', 'runs', 'evaluated')
        )
    return (
        run_table.groupby(['family', 'dataset_slug', 'model'], as_index=False)
        .agg(runs=('run_dir', 'size'), evaluated=('has_result', 'sum'))
        .sort_values(['family', 'dataset_slug', 'model'])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# running the evaluator
# --------------------------------------------------------------------------
def evaluate_runs(run_table, run_evaluation=True, overwrite=False, device=None,
                  inference_mode='all', subspace_score_space='probability',
                  repo_root=REPO_ROOT):
    """Fill in the missing ``posthoc_inference_test_metrics.yaml`` files.

    Run this from one kernel at a time: the evaluator checks for an existing
    output before it starts and again when it saves, so a second concurrent copy
    of the sweep makes whichever process is slower fail at save time.

    One unevaluable run does not discard the sweep — the evaluator writes its
    YAML only at the end, so every already-written file stays intact and the
    failures are reported instead.
    """
    if device is None:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
    if device == 'cpu':
        print('CUDA is unavailable; inference will run on CPU and may be slow.')

    result_files, failed = [], []
    for row in run_table.itertuples():
        output_path = Path(row.run_dir) / RESULT_FILENAME
        result_files.append(output_path)
        if not run_evaluation:
            continue
        if output_path.exists() and not overwrite:
            print('[reuse]', output_path)
            continue
        command = [
            sys.executable, '-m', 'evaluation.evaluate_checkpoints',
            '--run-dir', str(row.run_dir),
            '--inference-mode', inference_mode,
            '--subspace-score-space', subspace_score_space,
            '--checkpoint-mode', 'both',
            '--device', device,
        ]
        if overwrite:
            command.append('--overwrite')
        print('[run]', ' '.join(command))
        try:
            subprocess.run(command, cwd=repo_root, check=True)
        except subprocess.CalledProcessError as error:
            print(f'[failed] exit status {error.returncode}: {row.run_dir}')
            failed.append((Path(row.run_dir), error.returncode))

    available = [path for path in result_files if path.is_file()]
    print(f'Available result files: {len(available)}/{len(result_files)}')
    if not run_evaluation and not available:
        print('Set RUN_EVALUATION=True and run this cell to generate results.')
    if failed:
        print(f'\n[error] {len(failed)} run(s) did not produce a result file. '
              'The rows below are missing from every table in this notebook:')
        for run_dir, returncode in failed:
            print(f'    exit {returncode}  {run_dir}')

    _report_result_health(available, subspace_score_space)
    return available


def _report_result_health(paths, subspace_score_space):
    """Warn about files that are readable but not comparable with the rest."""
    partial, fallback, other_space = [], [], []
    for path in paths:
        payload = yaml.safe_load(Path(path).read_text())
        if len(payload.get('resolved_inference_modes', [])) < 4:
            partial.append(path)
        if payload.get('test_split_source') == 'official_dataset_adapter_fallback':
            fallback.append(path)
        # Absent key = written before the option existed, i.e. the coordinate space.
        if payload.get('subspace_score_space', 'coordinate') != subspace_score_space:
            other_space.append(path)
    if partial:
        print(f'\n{len(partial)}/{len(paths)} files predate the four-cell grid or were '
              'written with --inference-mode both.')
        print('Set OVERWRITE=True and re-run this cell to fill in the missing cells.')
    if other_space:
        print(f'\n{len(other_space)}/{len(paths)} files were written in a different '
              f'subspace score space than the selected {subspace_score_space!r}.')
        print('Their subspace_norm rows are not comparable with the rest. Set '
              'OVERWRITE=True and re-run this cell to regenerate them.')
    if fallback:
        print(f'\n[warning] {len(fallback)} file(s) were evaluated with the official-adapter '
              "fallback because the run's configured test manifest is missing. Their label "
              'space may differ from what the run trained on; check the native row against '
              "the run's own test_metrics.yaml before trusting them:")
        for path in fallback:
            print('   ', path)


# --------------------------------------------------------------------------
# loading results
# --------------------------------------------------------------------------
def split_metric(metric):
    """Return ``(metric_family, decoder)`` for one metric key, or ``(None, None)``."""
    for decoder in DECODERS:
        if metric.startswith(f'acc_level_{decoder}_'):
            return f"acc_level_{metric.rsplit('_', 1)[-1]}", decoder
        if metric.endswith(f'_{decoder}'):
            return metric[: -(len(decoder) + 1)], decoder
    return None, None


def native_cell_of(payload):
    """The grid cell that reproduces this checkpoint's own inference."""
    return payload.get('native_inference_mode') or native_inference_rule(
        payload['model'], bool(payload.get('hcc_trained_run', False))
    )


def canonical_cell(row_name, payload):
    """Map one YAML row onto a grid cell, for files written before or after the rename.

    ``normal`` named whatever the checkpoint did natively, so it follows the
    model; every other legacy name maps through the CLI's own alias table.
    """
    if row_name == 'normal':
        return native_cell_of(payload)
    return canonical_inference_rule(row_name, payload['model'])


def load_results(run_table=None, paths=None, verbose=True):
    """Long-form table of every metric, for every family, cell, decoder and seed.

    Both decoders are always loaded; selection happens later, so switching
    between independent, top-down or both never needs a reload. ``family`` comes
    from the run table, which is what makes a cross-mechanism comparison
    possible; everything else is read from the YAML the evaluator wrote.
    """
    if run_table is not None:
        sources = [
            (Path(row.run_dir) / RESULT_FILENAME, row.family)
            for row in run_table.itertuples()
        ]
    elif paths is not None:
        sources = [(Path(path), None) for path in paths]
    else:
        raise ValueError('Pass either run_table or paths.')

    records = []
    for result_path, family in sources:
        if not result_path.is_file():
            continue
        payload = yaml.safe_load(result_path.read_text())
        native_cell = native_cell_of(payload)
        for checkpoint_mode, checkpoint_payload in payload['checkpoints'].items():
            for row_name, metrics in checkpoint_payload['inference'].items():
                inference = canonical_cell(row_name, payload)
                for metric, value in metrics.items():
                    metric_family, decoder = split_metric(metric)
                    if metric_family is None:
                        continue
                    records.append({
                        'family': family,
                        'dataset': payload['dataset'],
                        'model': payload['model'],
                        'seed': int(payload['seed']),
                        'checkpoint_mode': checkpoint_mode,
                        'decoder': decoder,
                        'inference': inference,
                        'is_native': inference == native_cell,
                        'native_inference': native_cell,
                        'metric_family': metric_family,
                        'metric': metric,
                        'value': float(value),
                        'run_dir': str(result_path.parent),
                    })

    results = pd.DataFrame(records)
    if results.empty:
        print('No result rows loaded yet.')
        return results
    if verbose:
        print('Loaded rows:', len(results))
    return results


# --------------------------------------------------------------------------
# labels, ordering and the row "series"
# --------------------------------------------------------------------------
INFERENCE_ORDER = CELL_ORDER = (
    'node_score', 'subspace_norm', 'hcc_node_score', 'hcc_subspace_norm')
INFERENCE_LABELS = {
    'node_score': 'node_score',
    'subspace_norm': 'subspace_norm',
    'hcc_node_score': 'hcc + node_score',
    'hcc_subspace_norm': 'hcc + subspace_norm',
}
CELL_LABELS = {'node_score': 'node\nscore', 'subspace_norm': 'subspace\nnorm',
               'hcc_node_score': 'hcc +\nnode', 'hcc_subspace_norm': 'hcc +\nsubspace'}
MODEL_ORDER = ('hcast', 'ht_capsnet', 'hrn', 'hiercos')
MODEL_LABELS = {'hcast': 'H-CAST', 'ht_capsnet': 'HT-CapsNet',
                'hrn': 'HRN', 'hiercos': 'Hier-COS'}
MODEL_COLORS = {'hcast': '#2a78d6', 'ht_capsnet': '#9467bd',
                'hrn': '#eb6834', 'hiercos': '#1baf7a'}
DATASET_ORDER = ('cifar-100', 'cub-200-2011', 'fgvc-aircraft')
DATASET_LABELS = {'cifar-100': 'CIFAR-100', 'cub-200-2011': 'CUB-200-2011',
                  'fgvc-aircraft': 'FGVC-Aircraft'}
DATASET_SHORT = {'cifar-100': 'CIFAR-100', 'cub-200-2011': 'CUB-200',
                 'fgvc-aircraft': 'Aircraft'}
DECODER_LABELS = {'independent': 'Independent decoding', 'topdown': 'Top-down decoding'}
DECODER_SHORT = {'independent': 'independent', 'topdown': 'top-down'}
READOUT_LABELS = {'node_score': 'node score', 'subspace_norm': 'subspace norm'}
# (title, short name, unit, higher_is_better, symlog linear threshold)
METRIC_SPECS = {
    'fpa':         ('Full-path accuracy', 'FPA', 'pp', True,  1.0),
    'tice':        ('Tree inconsistency', 'TICE', 'pp', False, 1.0),
    'weighted_ap': ('Weighted per-level accuracy', 'wAP', 'pp', True, 1.0),
    'ahd':         ('Average hierarchy distance', 'AHD', 'raw', False, 0.05),
    'acc_level_0': ('Coarse accuracy', 'acc@0', 'pp', True, 1.0),
    'acc_level_1': ('Middle accuracy', 'acc@1', 'pp', True, 1.0),
    'acc_level_2': ('Fine accuracy', 'acc@2', 'pp', True, 1.0),
}

SERIES_SEPARATOR = ' | '


def add_series(frame, by='model'):
    """Add the ``series`` column the grids use for their rows.

    ``by='model'`` is the single-mechanism view, one row per model.
    ``by='model_family'`` is the cross-mechanism view, one row per
    (model, mechanism) pair, which is what puts "HCC under hcc+subspace" and
    "baseline under subspace" on the same grid. ``by='family'`` is the same for a
    single model.
    """
    frame = frame.copy()
    if frame.empty:
        # nothing loaded yet: keep the empty frame usable by every caller below
        frame['series'] = pd.Series(dtype=object)
        return frame
    if by == 'model':
        frame['series'] = frame['model']
    elif by == 'family':
        frame['series'] = frame['family']
    elif by == 'model_family':
        frame['series'] = frame['model'] + SERIES_SEPARATOR + frame['family']
    else:
        raise ValueError(f"by must be 'model', 'family' or 'model_family', not {by!r}")
    return frame


def series_model(series):
    """The model a series belongs to, for colour and ordering."""
    head = str(series).split(SERIES_SEPARATOR)[0]
    return head if head in MODEL_LABELS else None


def series_family(series):
    parts = str(series).split(SERIES_SEPARATOR)
    tail = parts[-1]
    return tail if tail in FAMILY_LABELS else None


def series_label(series, short=False):
    """'H-CAST', 'HCC', or 'H-CAST · HCC' depending on how the series was built."""
    parts = str(series).split(SERIES_SEPARATOR)
    labels = FAMILY_SHORT if short else FAMILY_LABELS
    out = [MODEL_LABELS.get(part, labels.get(part, part)) for part in parts]
    return ' · '.join(out)


def series_color(series):
    model = series_model(series)
    return MODEL_COLORS.get(model, '#52514e')


def order_series(values):
    """Model order first, mechanism order inside it; unknown keys keep their order."""
    values = list(dict.fromkeys(values))

    def rank(series):
        model, family = series_model(series), series_family(series)
        return (
            MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER),
            FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER),
            str(series),
        )

    return sorted(values, key=rank)


# --------------------------------------------------------------------------
# selection and aggregation
# --------------------------------------------------------------------------
def _as_list(value, available, quiet=False):
    """Normalize a selector into a concrete list of values that exist in the data."""
    if value is None or (isinstance(value, str) and value == 'all'):
        return list(available)
    if isinstance(value, str):
        value = [value]
    unknown = [item for item in value if item not in available]
    if unknown and not quiet:
        print(f'[note] not present in the loaded data, ignored: {unknown}')
    return [item for item in value if item in available]


def select(frame, inferences='all', decoders='both', datasets='all', models='all',
           families='all', metrics='all', matched_checkpoint=True, quiet=False):
    """Return the subset of rows to display.

    ``matched_checkpoint=True`` keeps the repository rule that a decoder is read
    from its own validation-selected checkpoint. Setting it False also shows the
    crossed combinations, which are not comparable to the reported results.
    ``inferences='native'`` keeps only each checkpoint's own cell.
    """
    if frame.empty:
        return frame

    selected = frame
    if isinstance(inferences, str) and inferences == 'native':
        selected = selected[selected['is_native']]
    else:
        available = [cell for cell in INFERENCE_ORDER if cell in set(frame['inference'])]
        selected = selected[selected['inference'].isin(_as_list(inferences, available, quiet))]

    decoders = list(DECODERS) if decoders == 'both' else decoders
    selected = selected[selected['decoder'].isin(_as_list(decoders, DECODERS, quiet))]
    selected = selected[selected['dataset'].isin(
        _as_list(datasets, sorted(set(frame['dataset'])), quiet))]
    selected = selected[selected['model'].isin(
        _as_list(models, sorted(set(frame['model'])), quiet))]
    if 'family' in frame.columns and frame['family'].notna().any():
        selected = selected[selected['family'].isin(
            _as_list(families, sorted(set(frame['family'].dropna())), quiet))]
    selected = selected[selected['metric_family'].isin(
        _as_list(metrics, sorted(set(frame['metric_family'])), quiet))]
    if matched_checkpoint:
        selected = selected[selected['decoder'].eq(selected['checkpoint_mode'])]
    return selected.reset_index(drop=True)


def describe_selection(view, matched_checkpoint=True):
    """One-line-per-axis summary of what a view holds."""
    if view.empty:
        print('Nothing selected. Widen the selection.')
        return
    print(f'Selected {len(view)} rows')
    print('  inference cells :', ', '.join(c for c in INFERENCE_ORDER
                                           if c in set(view['inference'])))
    print('  decoders        :', ', '.join(sorted(set(view['decoder']))))
    print('  datasets        :', ', '.join(sorted(set(view['dataset']))))
    print('  models          :', ', '.join(sorted(set(view['model']))))
    if 'family' in view.columns and view['family'].notna().any():
        print('  mechanisms      :', ', '.join(sorted(set(view['family'].dropna()))))
    print('  metrics         :', ', '.join(sorted(set(view['metric_family']))))
    print('  checkpoint rule :',
          'matched to decoder' if matched_checkpoint else 'ALL (includes crossed pairs)')


def display_value(frame, column='value'):
    """AHD stays in raw units; every other metric is a percentage."""
    return np.where(frame['metric_family'].eq('ahd'), frame[column], 100.0 * frame[column])


def summarize_samples(frame, group_columns, value_column, mean_column, std_column):
    """Mean, sample standard deviation and seed count per group."""
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mean, std, count = sample_stats(group[value_column].tolist())
        rows.append({
            **dict(zip(group_columns, keys)),
            mean_column: mean,
            std_column: std,
            'seeds': count,
        })
    return pd.DataFrame(rows)


def _series_group_columns(frame):
    columns = ['dataset', 'series', 'decoder', 'inference']
    return [column for column in columns if column in frame.columns]


def absolute_summary(view):
    """Per-cell absolute test values: mean, sample SD, seed count.

    The frame keeps ``series``, ``model``, ``family`` and ``is_native`` so the
    figures can label rows and mark each checkpoint's own cell.
    """
    if view.empty:
        return pd.DataFrame()
    view = view.copy()
    view['value_display'] = display_value(view)
    keys = ['dataset', 'series', 'model', 'decoder', 'inference', 'is_native', 'metric_family']
    if 'family' in view.columns:
        keys.insert(3, 'family')
    return summarize_samples(view, keys, 'value_display', 'mean_value', 'std_value')


PAIR_KEYS = ['dataset', 'model', 'seed', 'checkpoint_mode', 'decoder',
             'metric_family', 'metric']


def paired_gains(view, reference=None):
    """Gain of each selected cell against the **same checkpoint's** native cell.

    This is the within-checkpoint comparison: the two rows come from one set of
    weights, so the only thing that differs is the inference rule. Positive
    always means the selected cell is better, including for the lower-is-better
    AHD and TICE. The native row is taken from the full loaded table, so
    deselecting it does not remove the reference.
    """
    if view.empty:
        return pd.DataFrame()
    reference = view if reference is None else reference
    if reference.empty:
        return pd.DataFrame()

    pair_keys = list(PAIR_KEYS)
    if 'family' in view.columns and 'family' in reference.columns:
        pair_keys.append('family')
    native = (
        reference[reference['is_native']][pair_keys + ['value', 'inference']]
        .rename(columns={'value': 'native_value', 'inference': 'native_cell'})
        .drop_duplicates(subset=pair_keys)
    )
    merged = view[~view['is_native']].merge(native, on=pair_keys, how='inner')
    if merged.empty:
        return merged
    sign = np.where(merged['metric_family'].isin(LOWER_IS_BETTER), -1.0, 1.0)
    merged['gain'] = sign * (merged['value'] - merged['native_value'])
    merged['gain_display'] = np.where(
        merged['metric_family'].eq('ahd'), merged['gain'], 100.0 * merged['gain'])
    return merged


def gain_summary(paired):
    """Mean paired gain per (dataset, series, decoder, cell, metric)."""
    if paired is None or paired.empty:
        return pd.DataFrame()
    keys = ['dataset', 'series', 'model', 'decoder', 'inference', 'metric_family']
    if 'family' in paired.columns:
        keys.insert(3, 'family')
    keys = [key for key in keys if key in paired.columns]
    return summarize_samples(paired, keys, 'gain_display', 'mean_gain', 'std_gain')


# --------------------------------------------------------------------------
# cross-mechanism comparison
# --------------------------------------------------------------------------
def cross_family_gains(frame, anchor_family='baseline', anchor_cell='native',
                       matched_checkpoint=True):
    """Delta of every (mechanism, cell) against one chosen anchor.

    This is what the per-mechanism notebooks cannot do: it compares, say, HCC
    read with ``hcc + subspace`` against the baseline read with ``subspace``,
    instead of comparing each checkpoint only with itself.

    **It is not a within-checkpoint comparison.** The two rows come from two
    different training runs that share only the seed, the dataset and the test
    split, so the delta mixes the training mechanism with the readout and its
    spread is the spread of that combined effect. Report it as mean +- sample SD
    over matched seeds, never as the paired gain of an inference rule.

    ``anchor_cell='native'`` uses the anchor mechanism's own inference; pass a
    cell name to anchor on a specific readout instead. Positive always means the
    listed row is better, including for the lower-is-better AHD and TICE.
    """
    if frame.empty:
        return pd.DataFrame()
    if 'family' not in frame.columns or frame['family'].isna().all():
        raise ValueError('The loaded table has no family column; load with a run table.')
    if anchor_family not in set(frame['family']):
        raise ValueError(f'Anchor family {anchor_family!r} is not loaded; '
                         f'available: {sorted(set(frame["family"]))}')

    anchor_rows = frame[frame['family'].eq(anchor_family)]
    if anchor_cell == 'native':
        anchor_rows = anchor_rows[anchor_rows['is_native']]
    else:
        anchor_rows = anchor_rows[anchor_rows['inference'].eq(anchor_cell)]
    if anchor_rows.empty:
        raise ValueError(f'No rows for anchor {anchor_family}/{anchor_cell}.')

    keys = ['dataset', 'model', 'seed', 'checkpoint_mode', 'decoder',
            'metric_family', 'metric']
    anchor = (anchor_rows[keys + ['value', 'inference']]
              .rename(columns={'value': 'anchor_value', 'inference': 'anchor_cell'})
              .drop_duplicates(subset=keys))
    merged = frame.merge(anchor, on=keys, how='inner')
    if matched_checkpoint:
        merged = merged[merged['decoder'].eq(merged['checkpoint_mode'])]
    if merged.empty:
        return merged

    merged['anchor_family'] = anchor_family
    merged['is_anchor'] = (merged['family'].eq(anchor_family)
                           & merged['inference'].eq(merged['anchor_cell']))
    sign = np.where(merged['metric_family'].isin(LOWER_IS_BETTER), -1.0, 1.0)
    merged['gain'] = sign * (merged['value'] - merged['anchor_value'])
    merged['gain_display'] = np.where(
        merged['metric_family'].eq('ahd'), merged['gain'], 100.0 * merged['gain'])
    return merged.reset_index(drop=True)


def rank_cells(absolute_table, metric='fpa', decoder='independent', by=('dataset', 'model')):
    """Every (mechanism, cell) ranked inside each group, best first.

    Answers "what is the best inference across all modes and all mechanisms",
    with the mechanism's own native cell and the gap to the winner alongside, so
    a win by less than the seed spread is visible as such.
    """
    if absolute_table.empty:
        return pd.DataFrame()
    higher_better = METRIC_SPECS[metric][3]
    table = absolute_table[
        absolute_table['metric_family'].eq(metric)
        & absolute_table['decoder'].eq(decoder)
    ].copy()
    if table.empty:
        print(f'No rows for {metric} under {decoder} decoding.')
        return table

    by = [key for key in by if key in table.columns]
    table['rank'] = (table.groupby(by)['mean_value']
                     .rank(ascending=not higher_better, method='min').astype(int))
    best = (table.sort_values('rank').groupby(by)['mean_value']
            .first().rename('best_value'))
    table = table.merge(best, on=by, how='left')
    sign = 1.0 if higher_better else -1.0
    table['gap_to_best'] = sign * (table['mean_value'] - table['best_value'])
    columns = [key for key in ('dataset', 'model', 'family', 'series', 'inference',
                               'is_native', 'mean_value', 'std_value', 'seeds',
                               'rank', 'gap_to_best') if key in table.columns]
    return (table[columns].sort_values(by + ['rank']).reset_index(drop=True))


def best_cells(absolute_table, metric='fpa', decoder='independent',
               by=('dataset', 'model')):
    """The winning (mechanism, cell) per group, with its margin over the runner-up.

    ``margin`` is the distance to the second-best row in the same group, in the
    same units as the metric; compare it with ``std_value`` before calling a
    winner. ``native_value`` is the winning row's own native readout, so the
    part of the win that the readout contributes can be read off directly.
    """
    ranked = rank_cells(absolute_table, metric=metric, decoder=decoder, by=by)
    if ranked.empty:
        return ranked
    by = [key for key in by if key in ranked.columns]
    rows = []
    for keys, group in ranked.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        group = group.sort_values('rank')
        winner = group.iloc[0]
        runner_up = group.iloc[1] if len(group) > 1 else None
        if 'family' in group.columns:
            native = group[group['is_native'] & group['family'].eq(winner['family'])]
        else:
            native = group[group['is_native']]
        row = {**dict(zip(by, keys)),
               'best_inference': winner['inference'],
               'best_value': winner['mean_value'],
               'best_std': winner['std_value'],
               'seeds': winner['seeds'],
               'margin': (0.0 if runner_up is None
                          else float(winner['gap_to_best'] - runner_up['gap_to_best'])),
               'runner_up': None if runner_up is None else runner_up['inference']}
        if 'family' in group.columns:
            row['best_family'] = winner['family']
            row['runner_up_family'] = None if runner_up is None else runner_up['family']
        if not native.empty:
            row['native_inference'] = native.iloc[0]['inference']
            row['native_value'] = native.iloc[0]['mean_value']
        rows.append(row)
    front = [key for key in by]
    frame = pd.DataFrame(rows)
    ordered = front + [c for c in ('best_family', 'best_inference', 'best_value', 'best_std',
                                   'native_inference', 'native_value', 'margin', 'runner_up',
                                   'runner_up_family', 'seeds') if c in frame.columns]
    return frame[ordered]


# ==========================================================================
# FIGURES
# ==========================================================================
# Every figure is laid out in inches at the size it is printed at, so the font
# sizes below are the sizes the reader sees on the page: include them with
# \includegraphics[width=\textwidth]{...} and no rescaling. Each call saves a
# PDF for LaTeX plus a PNG for previews.
#
# These are module-level settings on purpose: a notebook overrides them with
# ``posthoc.THESIS_STYLE = False`` and every figure follows.
# --------------------------------------------------------------------------
# Width of the text block the figure is included in, read from the one place the
# thesis geometry is written down. main.tex uses a4paper with inner=3cm,
# outer=2.5cm and bindingoffset=0.5cm, which leaves 150 mm = 5.906 in.
from thesis_style import TEXT_WIDTH_IN, use_paper_style  # noqa: E402
# True: no in-figure title or explanatory paragraph, because in a thesis both
# belong to the LaTeX caption. False: keep them, for reading in the notebook.
THESIS_STYLE = True
# Keep notebook output focused on the analysis. Set this to True only when a
# ready-to-paste LaTeX figure environment is explicitly wanted.
PRINT_LATEX_BLOCKS = False

# What thesis mode leaves behind is an unlabelled figure on screen: these grids
# carry no title, and a notebook that renders five of them gives the reader
# nothing saying which comparison, which datasets or which decoder any one of
# them is. The title and a one-line scope note are therefore written above the
# figure as notebook text, which labels it while it is read and never reaches
# the exported PDF the document includes. Set to False for bare figures.
SHOW_FIGURE_CAPTIONS = True


def show_markdown(text):
    """Render a line of markdown in a notebook; plain text anywhere else."""
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display

        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ == 'ZMQInteractiveShell':
            display(Markdown(text))
            return
    except Exception:
        pass
    print(text)


def _caption(title, note=None):
    """Name the figure just above it, when the title is not on the canvas."""
    if not (THESIS_STYLE and SHOW_FIGURE_CAPTIONS):
        return
    show_markdown(f'**{title}**' + (f'  \n{note}' if note else ''))


def _scope_note(datasets, series_list, decoders=None, extra=None):
    """One line saying what the figure covers: datasets, rows, decoders."""
    parts = ['datasets: ' + ', '.join(DATASET_SHORT.get(d, d) for d in datasets)]
    if series_list is not None:
        labels = [series_label(s, short=True) for s in series_list]
        parts.append('rows: ' + (', '.join(labels) if len(labels) <= 6
                                 else f'{len(labels)} rows'))
    if decoders:
        parts.append(' and '.join(DECODER_SHORT.get(d, d) for d in decoders)
                     + ' decoding')
    if extra:
        parts.append(extra)
    return ' \u00b7 '.join(parts)

# Use the same serif typography, font sizes, grid treatment and vector-font
# settings as the model-analysis notebooks.  The inference figures keep their
# own task-specific layouts and colour encodings, but should not look like a
# separate document when placed beside the model-analysis figures.
use_paper_style()
mpl.rcParams.update({'svg.fonttype': 'none'})

# --- design tokens: dataviz reference palette ------------------------------
# Diverging pair (polarity: worse <-> better) with a neutral gray midpoint.
NEG_HUE, MID_HUE, POS_HUE = '#e34948', '#ececeb', '#2a78d6'
DIVERGING = LinearSegmentedColormap.from_list('gain', [NEG_HUE, MID_HUE, POS_HUE])
# Sequential blue ramp (one hue, light -> dark), the reference sequential scale.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'level', ['#eaf1fc', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
              '#256abf', '#184f95', '#0d366b'])
INK, INK_2, INK_3 = '#0b0b0b', '#52514e', '#8a8a85'
NATIVE_FILL = '#f2f2f0'
GRID_LINE = '#e9e9e7'
AXIS_LINE = '#d8d8d5'

# Point sizes, chosen for the printed width above: nothing here is below 6 pt.
FS_TILE, FS_TICK, FS_GROUP, FS_TITLE, FS_NOTE = 6.8, 6.8, 7.6, 8.6, 6.6
MIN_COL_W = 0.43     # inches; narrower than this and the two-line cell labels collide
EPS = 1e-6


def _text_on(rgb):
    r, g, b = to_rgb(rgb)
    return '#ffffff' if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else INK


def _nice_ceiling(value):
    """Round a colour-scale limit up to a readable number (40, not 33.0912)."""
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exp = np.floor(np.log10(value))
    for step in (1, 1.5, 2, 3, 4, 5, 7.5, 10):
        candidate = step * 10.0 ** exp
        if candidate >= value * 0.999:
            return float(candidate)
    return float(10.0 ** (exp + 1))


def seeds_phrase(table):
    """'3 seeds', or '2-3 seeds' when coverage is uneven, from the summary itself."""
    if table is None or 'seeds' not in getattr(table, 'columns', []) or table.empty:
        return 'the available seeds'
    lo, hi = int(table['seeds'].min()), int(table['seeds'].max())
    return f'{lo} seeds' if lo == hi else f'{lo}\u2013{hi} seeds'


def latex_block(path, caption, label, width=r'\textwidth'):
    """Optionally print a figure environment to paste into the thesis."""
    if not PRINT_LATEX_BLOCKS:
        return
    print('\n% ---- LaTeX ----------------------------------------------------')
    print('\\begin{figure}[tbp]\n  \\centering')
    print(f'  \\includegraphics[width={width}]{{figures/{Path(path).stem}}}')
    print(f'  \\caption{{{caption}}}')
    print(f'  \\label{{fig:{label}}}\n\\end{{figure}}')


def save_figure(fig, save_path):
    """Write the vector copy LaTeX includes plus a raster copy for slides/preview."""
    if not save_path:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.png'):
        out = Path(save_path).with_suffix(suffix)
        # These figures use explicit inch-based ``add_axes`` geometry.  The
        # shared paper style enables constrained layout for the model-analysis
        # GridSpec figures, so disable it again before every backend render.
        fig.set_layout_engine('none')
        fig.savefig(out, dpi=300, facecolor='white')
        print('[saved]', out)


def _row_geometry(series_list):
    """Left gutter and dataset-label offset, in inches and points.

    A cross-mechanism grid has row labels like 'H-CAST · HCC', which do not fit
    the gutter a model-only grid needs, so the space is measured rather than
    fixed.
    """
    longest = max((len(series_label(s, short=True)) for s in series_list), default=8)
    label_in = longest * (FS_TICK + 0.4) * 0.55 / 72.0
    dataset_offset_pt = (label_in + 0.14) * 72.0
    return max(1.0, label_in + 0.42), dataset_offset_pt


def _draw_row_labels(ax0, rows, datasets, series_list, dataset_offset_pt):
    """Series label on every row, dataset name once per block, leftmost panel only."""
    for i, (_, series) in enumerate(rows):
        ax0.annotate(series_label(series, short=True), xy=(0, i + .5),
                     xycoords=('axes fraction', 'data'), xytext=(-4, 0),
                     textcoords='offset points', ha='right', va='center',
                     fontsize=FS_TICK + 0.4, color=INK)
    for k, dataset in enumerate(datasets):
        ax0.annotate(DATASET_LABELS[dataset], xy=(0, (k + .5) * len(series_list)),
                     xycoords=('axes fraction', 'data'), xytext=(-dataset_offset_pt, 0),
                     textcoords='offset points', rotation=90, ha='center', va='center',
                     fontsize=FS_GROUP - 0.4, fontweight='bold', color=INK)


def _draw_block_rules(ax, datasets, series_list, n_cell, decoders):
    """Separators: solid between datasets, hairline between models inside a block."""
    for k, _ in enumerate(decoders):
        if k:
            ax.axvline(k * n_cell, color=INK_3, linewidth=1.0, zorder=4)
    for k in range(1, len(datasets)):
        ax.axhline(k * len(series_list), color=INK_3, linewidth=1.0, zorder=4)
    models = [series_model(s) for s in series_list]
    if len(set(models)) > 1:
        for k in range(1, len(series_list)):
            if models[k] != models[k - 1]:
                for block in range(len(datasets)):
                    ax.axhline(block * len(series_list) + k, color='#ffffff',
                               linewidth=1.6, zorder=4)
                    ax.axhline(block * len(series_list) + k, color=AXIS_LINE,
                               linewidth=0.7, zorder=5)


def _resolve_axes(table, datasets, series, metrics, decoders_wanted):
    """Concrete, ordered axis values that exist in ``table``."""
    datasets = [d for d in DATASET_ORDER if d in set(table['dataset'])] \
        if datasets is None else [d for d in datasets if d in set(table['dataset'])]
    series = order_series(set(table['series'])) if series is None \
        else [s for s in series if s in set(table['series'])]
    metrics = [m for m in metrics if m in set(table['metric_family'])]
    decoders = [d for d in decoders_wanted if d in set(table['decoder'])]
    return datasets, series, metrics, decoders


def effect_grid(gain_table, seed_table, absolute_table, metrics=('fpa', 'tice'),
                datasets=None, series=None, save_path=None, width_in=None,
                caption_note='', reference_map=None, reference_label='native',
                scale_label=None):
    """Annotated gain heatmap: rows = dataset x series, columns = cell x decoder.

    Laid out in inches rather than by tight_layout, so tiles keep the same size
    whatever is selected and the figure lands at exactly ``width_in`` on the page.
    The colour scale is symmetric-log: without it the +1 pp effects that matter
    for H-CAST and Hier-COS are invisible next to a ~-33 pp collapse.

    By default the zero tile of every row is that checkpoint's own native cell.
    ``reference_map`` overrides it with an explicit ``{(dataset, series): cell}``
    mapping, which is how the cross-mechanism notebook marks the single anchor
    row that every other row is measured against; ``reference_label`` names it in
    the tile and ``scale_label`` on the colour bar.
    """
    width_in = TEXT_WIDTH_IN if width_in is None else width_in
    if gain_table is None or gain_table.empty:
        print('Nothing to plot.')
        return
    datasets, series_list, metrics, decoders = _resolve_axes(
        gain_table, datasets, series, metrics, DECODERS)
    if not (datasets and series_list and metrics and decoders):
        print('Nothing to plot.')
        return

    rows = [(d, s) for d in datasets for s in series_list]
    native_of = (absolute_table[absolute_table['is_native']]
                 .groupby(['dataset', 'series'])['inference'].first().to_dict()
                 if reference_map is None else dict(reference_map))
    n_cell = len(CELL_ORDER)

    # ---- per metric: values, seed agreement, and which decoders carry a result
    panels = []
    for metric in metrics:
        title, short, unit, higher_better, linthresh = METRIC_SPECS[metric]
        sub = gain_table[gain_table['metric_family'].eq(metric)]
        lookup = sub.set_index(['dataset', 'series', 'decoder', 'inference'])['mean_gain'].to_dict()
        agree = {}
        if seed_table is not None and not seed_table.empty:
            s = seed_table[seed_table['metric_family'].eq(metric)]
            for key, grp in s.groupby(['dataset', 'series', 'decoder', 'inference']):
                v = grp['gain_display'].to_numpy()
                agree[key] = bool(np.all(v > 0) or np.all(v < 0)) or len(v) < 2
        # A decoder block that is identically zero is a structural fact, not a
        # result: state it in one line instead of printing 36 zeros.
        keep = []
        for dec in decoders:
            vals = [lookup.get((ds, sr, dec, c), np.nan) for ds, sr in rows
                    for c in CELL_ORDER if native_of.get((ds, sr)) != c]
            vals = [v for v in vals if np.isfinite(v)]
            if not (vals and all(abs(v) < EPS for v in vals)):
                keep.append(dec)
        dropped = [d for d in decoders if d not in keep]
        if not keep:                      # degenerate everywhere: keep one block
            keep, dropped = decoders[:1], list(decoders[1:])
        panels.append(dict(metric=metric, title=title, short=short, unit=unit,
                           higher_better=higher_better, linthresh=linthresh,
                           lookup=lookup, agree=agree, decoders=keep, dropped=dropped,
                           peak=float(np.nanmax(np.abs(sub['mean_gain']))) if len(sub) else 1.0))

    # ---- colour scale: one shared bar when the metrics share unit and threshold
    scales = {(p['unit'], p['linthresh']) for p in panels}
    shared = len(scales) == 1
    if shared:
        vmax = _nice_ceiling(max(max(p['peak'] for p in panels),
                                 2 * panels[0]['linthresh']))
        for p in panels:
            p['norm'] = SymLogNorm(linthresh=p['linthresh'], vmin=-vmax, vmax=vmax, base=10)
    else:
        for p in panels:
            v = _nice_ceiling(max(p['peak'], 2 * p['linthresh']))
            p['norm'] = SymLogNorm(linthresh=p['linthresh'], vmin=-v, vmax=v, base=10)

    # ---- layout, in inches ------------------------------------------------
    gutter, dataset_offset_pt = _row_geometry(series_list)
    tile_h = 0.30
    pad_r, gap = 0.04, 0.20
    head = 0.79                                   # title + direction + decoder + cell labels
    grid_h = tile_h * len(rows)
    note_h = 0.30 if any(p['dropped'] for p in panels) else 0.0
    cbar_h = 0.60 if shared else 0.46 * len(panels) + 0.14
    foot_h = 0.34 if not THESIS_STYLE else 0.0
    top_pad = 0.05 if THESIS_STYLE else 0.28
    height = top_pad + head + grid_h + note_h + cbar_h + foot_h + 0.05

    total_cols = sum(len(p['decoders']) * n_cell for p in panels)
    fixed = gutter + pad_r + gap * (len(panels) - 1)
    col_w = max(MIN_COL_W, (width_in - fixed) / total_cols)
    if fixed + col_w * total_cols > width_in + 1e-9:
        # too many columns for the text block: widen rather than crush the labels,
        # and say by how much LaTeX will then shrink it
        width_in = fixed + col_w * total_cols
        print(f'[note] {total_cols} columns need {width_in:.2f} in; at '
              f'width=\\textwidth LaTeX scales it to {TEXT_WIDTH_IN / width_in:.0%} '
              f'(tile labels land at {FS_TILE * TEXT_WIDTH_IN / width_in:.1f} pt). '
              'Show fewer metrics, or place it on a landscape page.')

    fig = plt.figure(figsize=(width_in, height), layout='none')
    x = gutter
    for p in panels:
        cols = [(dec, c) for dec in p['decoders'] for c in CELL_ORDER]
        w = col_w * len(cols)
        ax = fig.add_axes([x / width_in, 1 - (top_pad + head + grid_h) / height,
                           w / width_in, grid_h / height])
        p['ax'] = ax
        unit, linthresh, norm = p['unit'], p['linthresh'], p['norm']

        for i, (ds, sr) in enumerate(rows):
            for j, (dec, cell) in enumerate(cols):
                if native_of.get((ds, sr)) == cell:
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor=NATIVE_FILL,
                                           edgecolor='white', linewidth=1.4, zorder=1))
                    ax.text(j + .5, i + .5, reference_label, ha='center', va='center',
                            fontsize=FS_TILE - 0.6, style='italic', color=INK_3, zorder=3)
                    continue
                val = p['lookup'].get((ds, sr, dec, cell), np.nan)
                if not np.isfinite(val):
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor='#ffffff',
                                           edgecolor='white', linewidth=1.4, zorder=1))
                    continue
                face = DIVERGING(norm(val))
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face,
                                       edgecolor='white', linewidth=1.4, zorder=1))
                label = '0' if abs(val) < EPS else (
                    f'{val:+.2f}' if unit == 'raw' else f'{val:+.1f}')
                # flag only real effects whose sign disagrees across seeds
                if abs(val) >= EPS and not p['agree'].get((ds, sr, dec, cell), True):
                    label += '†'
                ax.text(j + .5, i + .5, label, ha='center', va='center',
                        fontsize=FS_TILE, color=_text_on(face), zorder=3,
                        fontweight='bold' if abs(val) >= 10 * linthresh else 'normal')

        ax.set_xlim(0, len(cols)); ax.set_ylim(len(rows), 0)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # column headers above the grid, table-style: metric, decoder, then cell
        for j, (_, cell) in enumerate(cols):
            ax.annotate(CELL_LABELS[cell], xy=(j + .5, 1), xycoords=('data', 'axes fraction'),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                        fontsize=FS_TICK, color=INK_2, linespacing=1.15)
        for k, dec in enumerate(p['decoders']):
            centre = (k + .5) * n_cell
            ax.annotate(DECODER_LABELS[dec], xy=(centre, 1), xycoords=('data', 'axes fraction'),
                        xytext=(0, 22), textcoords='offset points', ha='center', va='bottom',
                        fontsize=FS_GROUP, fontweight='bold', color=INK)
        _draw_block_rules(ax, datasets, series_list, n_cell, p['decoders'])

        direction = 'higher' if p['higher_better'] else 'lower'
        ax.annotate(f'raw metric: {direction} is better', xy=(0.5, 1), xycoords='axes fraction',
                    xytext=(0, 34), textcoords='offset points', ha='center', va='bottom',
                    fontsize=FS_NOTE, color=INK_3)
        full = f"{p['title']} ({p['short']})"
        title_txt = full if len(full) * FS_TITLE * 0.58 <= w * 72 else p['short']
        ax.annotate(title_txt, xy=(0.5, 1), xycoords='axes fraction',
                    xytext=(0, 44), textcoords='offset points', ha='center', va='bottom',
                    fontsize=FS_TITLE, fontweight='bold', color=INK)
        if p['dropped']:
            ax.annotate('\n'.join(f'{DECODER_SHORT[d]} decoding omitted:'
                                  for d in p['dropped'])
                        + f'\n{p["short"]} gain ≡ 0 for every cell, by construction',
                        xy=(1, 0), xycoords='axes fraction', xytext=(0, -7),
                        textcoords='offset points', ha='right', va='top', linespacing=1.25,
                        fontsize=FS_NOTE, style='italic', color=INK_3)
        x += w + gap

    _draw_row_labels(panels[0]['ax'], rows, datasets, series_list, dataset_offset_pt)

    # ---- colour bars ------------------------------------------------------
    def draw_bar(p, y_in, w_in, x_in):
        norm, unit, linthresh = p['norm'], p['unit'], p['linthresh']
        vmax_p = norm.vmax
        cax = fig.add_axes([x_in / width_in, 1 - y_in / height, w_in / width_in, 0.115 / height])
        fmt = '{:.2f}' if unit == 'raw' else '{:g}'
        wanted = [t for t in (-vmax_p, -10 * linthresh, -linthresh, 0,
                              linthresh, 10 * linthresh, vmax_p) if abs(t) <= vmax_p]
        # keep the outer ticks and drop any inner one whose label would touch its
        # neighbour: on a narrow bar +-10*linthresh can sit right under +-vmax
        bar_pt, ticks = w_in * 72, []
        for t in sorted(set(wanted)):
            half = 0.5 * len(fmt.format(t)) * FS_NOTE * 0.62
            if ticks:
                prev = ticks[-1]
                need = (half + 0.5 * len(fmt.format(prev)) * FS_NOTE * 0.62 + 2) / bar_pt
                if abs(float(norm(t)) - float(norm(prev))) < need:
                    if abs(t) >= vmax_p:          # never drop an end of the scale
                        ticks.pop()
                    else:
                        continue
            ticks.append(t)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax,
                          orientation='horizontal', ticks=ticks)
        cb.ax.set_xticklabels([fmt.format(t) for t in ticks],
                              fontsize=FS_NOTE, color=INK_2)
        cb.ax.tick_params(length=2, pad=1.5, color=INK_3)
        cb.outline.set_visible(False)
        unit_txt = 'pp' if unit == 'pp' else 'raw units'
        against = scale_label or 'the checkpoint’s own native cell'
        label = (f'gain against {against} ({unit_txt}, symmetric-log), '
                 'sign-adjusted: positive is better'
                 if shared else
                 f'{p["short"]} gain vs {against} ({unit_txt}, symlog) · positive is better')
        cb.set_label(label, fontsize=FS_NOTE, color=INK_2, labelpad=3)

    bar_top = top_pad + head + grid_h + note_h + 0.20
    bar_w = min(3.1, width_in - gutter - pad_r)
    bar_x = gutter + (width_in - gutter - pad_r - bar_w) / 2
    if shared:
        draw_bar(panels[0], bar_top, bar_w, bar_x)
    else:
        for n, p in enumerate(panels):
            draw_bar(p, bar_top + n * 0.46, bar_w, bar_x)

    _caption('Gain of every inference cell against the checkpoint’s own readout',
             _scope_note(datasets, series_list, decoders,
                         'metrics: ' + ', '.join(METRIC_SPECS[m][1] for m in metrics)))
    if not THESIS_STYLE:
        fig.suptitle('Does any inference cell beat the checkpoint’s own readout?',
                     fontsize=11, fontweight='bold', color=INK, y=1 - 0.06 / height)
        fig.text(0.5, 0.02,
                 'Mean gain against each checkpoint’s native cell, sign-adjusted so '
                 'positive is better.\n† marks an effect whose sign is not consistent across '
                 'all seeds.', ha='center', va='bottom', fontsize=FS_NOTE, color=INK_2)
    save_figure(fig, save_path)
    plt.show(); plt.close(fig)
    if save_path:
        against_txt = scale_label or "the same checkpoint's own native readout"
        latex_block(save_path,
                    'Post-hoc inference grid. Each tile is the mean gain over '
                    f'{seeds_phrase(gain_table)} of one '
                    f'readout$\\times$transform cell against {against_txt}'
                    ', sign-adjusted so that positive always means better, including for '
                    'the lower-is-better TICE. Colour is symmetric-log with a linear region of '
                    '$\\pm1$~pp, so a $1$~pp effect stays visible next to a $30$~pp collapse; '
                    'read the printed numbers for exact values. \\emph{'
                    + reference_label + '} marks the cell that is $0$ by definition, '
                    'and $\\dagger$ marks an '
                    'effect whose sign is not the same for every seed. Each decoder is read '
                    'from its own validation-selected checkpoint.' + caption_note,
                    'posthoc-effect-grid')


# Metrics whose values are heavy-tailed enough that a linear ramp shows nothing:
# shade them on the symmetric-log scale already used for the gains.
ABS_SYMLOG_METRICS = ('tice',)


def absolute_grid(absolute_table, metrics=('fpa', 'tice'), datasets=None, series=None,
                  show_spread=True, shade_scope='dataset', show_footer=True, save_path=None,
                  width_in=None, caption_note=''):
    """Raw test values on the effect grid's geometry.

    A gain hides the level it sits on: ``+2 pp`` on top of 75 pp and on top of
    1 pp look identical in the effect grid. This figure answers what that one
    cannot — where each row actually sits, whether a readout moves it past
    *another* row rather than only past its own native cell, and how large each
    effect is next to its seed spread.

    ``shade_scope='dataset'`` normalises the ramp inside each dataset block,
    because absolute levels are not comparable across datasets; ``'row'``
    rescales inside each series, which separates cells that a collapsed row
    would otherwise flatten; ``'panel'`` puts everything on one ramp. Darker is
    always better, including for the lower-is-better metrics. Set
    ``show_footer=False`` for a compact thesis footer: it retains the colour
    scale and any structurally omitted-decoder note, but removes the detailed
    ramp-range list and the seed/native-readout explanation.
    """
    width_in = TEXT_WIDTH_IN if width_in is None else width_in
    assert shade_scope in ('dataset', 'row', 'panel'), shade_scope
    if absolute_table is None or absolute_table.empty:
        print('Nothing to plot.')
        return
    A = absolute_table
    datasets, series_list, metrics, decoders = _resolve_axes(A, datasets, series, metrics, DECODERS)
    if not (datasets and series_list and metrics and decoders):
        print('Nothing to plot.')
        return

    rows = [(d, s) for d in datasets for s in series_list]
    native_of = (A[A['is_native']].groupby(['dataset', 'series'])['inference']
                 .first().to_dict())
    n_cell = len(CELL_ORDER)

    panels = []
    for metric in metrics:
        title, short, unit, higher_better, linthresh = METRIC_SPECS[metric]
        sub = A[A['metric_family'].eq(metric)]
        means = sub.set_index(['dataset', 'series', 'decoder', 'inference'])['mean_value'].to_dict()
        stds = sub.set_index(['dataset', 'series', 'decoder', 'inference'])['std_value'].to_dict()
        # A decoder block that is constant is a structural fact, not a result:
        # top-down TICE is 0 for every cell by construction.
        keep = []
        for dec in decoders:
            vals = [means.get((ds, sr, dec, c), np.nan) for ds, sr in rows for c in CELL_ORDER]
            vals = [v for v in vals if np.isfinite(v)]
            if not (vals and max(vals) - min(vals) < EPS):
                keep.append(dec)
        dropped = [d for d in decoders if d not in keep]
        if not keep:
            keep, dropped = decoders[:1], list(decoders[1:])
        # one scale per shading scope, over the columns actually drawn
        if shade_scope == 'dataset':
            groups = {ds: [(ds, sr) for sr in series_list] for ds in datasets}
        elif shade_scope == 'row':
            groups = {row: [row] for row in rows}
        else:
            groups = {'panel': list(rows)}
        norms, ranges = {}, {}
        for key, group_rows in groups.items():
            vals = [means.get((ds, sr, dec, c), np.nan)
                    for ds, sr in group_rows for dec in keep for c in CELL_ORDER]
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                continue
            lo, hi = float(min(vals)), float(max(vals))
            if hi - lo < EPS:
                hi = lo + 1.0
            norms[key] = (SymLogNorm(linthresh=linthresh, vmin=lo, vmax=hi, base=10)
                          if metric in ABS_SYMLOG_METRICS else Normalize(vmin=lo, vmax=hi))
            ranges[key] = (lo, hi)
        panels.append(dict(metric=metric, title=title, short=short, unit=unit,
                           higher_better=higher_better, means=means, stds=stds,
                           decoders=keep, dropped=dropped, norms=norms, ranges=ranges))

    # ---- layout, in inches ------------------------------------------------
    gutter, dataset_offset_pt = _row_geometry(series_list)
    tile_h = 0.36 if show_spread else 0.30
    pad_r, gap = 0.12, 0.20
    head = 0.79
    grid_h = tile_h * len(rows)
    note_lines = ({'dataset': 1 + len(datasets), 'row': 3, 'panel': 1}[shade_scope]
                  + 2 * max((len(p['dropped']) for p in panels), default=0))
    note_h = (0.115 * note_lines + 0.10 if show_footer else
              (0.30 if any(p['dropped'] for p in panels) else 0.0))
    legend_h = 0.60
    foot_h = 0.30 if show_footer and not THESIS_STYLE else 0.0
    top_pad = 0.05 if THESIS_STYLE else 0.28
    height = top_pad + head + grid_h + note_h + legend_h + foot_h + 0.05

    total_cols = sum(len(p['decoders']) * n_cell for p in panels)
    fixed = gutter + pad_r + gap * (len(panels) - 1)
    col_w = max(MIN_COL_W, (width_in - fixed) / total_cols)
    if fixed + col_w * total_cols > width_in + 1e-9:
        width_in = fixed + col_w * total_cols
        print(f'[note] {total_cols} columns need {width_in:.2f} in; at width=\\textwidth '
              f'LaTeX scales it to {TEXT_WIDTH_IN / width_in:.0%}.')

    fig = plt.figure(figsize=(width_in, height), layout='none')
    x = gutter
    for p in panels:
        cols = [(dec, c) for dec in p['decoders'] for c in CELL_ORDER]
        w = col_w * len(cols)
        ax = fig.add_axes([x / width_in, 1 - (top_pad + head + grid_h) / height,
                           w / width_in, grid_h / height])
        p['ax'] = ax
        fmt = '{:.2f}' if p['unit'] == 'raw' else '{:.1f}'

        for i, (ds, sr) in enumerate(rows):
            for j, (dec, cell) in enumerate(cols):
                val = p['means'].get((ds, sr, dec, cell), np.nan)
                if not np.isfinite(val):
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor='#ffffff',
                                           edgecolor='white', linewidth=1.4, zorder=1))
                    continue
                scope_key = ds if shade_scope == 'dataset' else (
                    (ds, sr) if shade_scope == 'row' else 'panel')
                norm = p['norms'].get(scope_key)
                t = float(norm(val)) if norm is not None else 0.5
                face = SEQUENTIAL(1.0 - t if not p['higher_better'] else t)
                native = native_of.get((ds, sr)) == cell
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face,
                                       edgecolor='white', linewidth=1.4, zorder=1))
                ink = _text_on(face)
                dy = -0.10 if show_spread else 0.0
                ax.text(j + .5, i + .5 + dy, fmt.format(val), ha='center', va='center',
                        fontsize=FS_TILE, color=ink, zorder=3,
                        fontweight='bold' if native else 'normal')
                sd = p['stds'].get((ds, sr, dec, cell), np.nan)
                if show_spread and np.isfinite(sd):
                    ax.text(j + .5, i + .78, '±' + fmt.format(sd), ha='center', va='center',
                            fontsize=FS_TILE - 1.6, color=ink, alpha=0.75, zorder=3)

        ax.set_xlim(0, len(cols)); ax.set_ylim(len(rows), 0)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        for j, (_, cell) in enumerate(cols):
            ax.annotate(CELL_LABELS[cell], xy=(j + .5, 1), xycoords=('data', 'axes fraction'),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                        fontsize=FS_TICK, color=INK_2, linespacing=1.15)
        for k, dec in enumerate(p['decoders']):
            centre = (k + .5) * n_cell
            ax.annotate(DECODER_LABELS[dec], xy=(centre, 1), xycoords=('data', 'axes fraction'),
                        xytext=(0, 22), textcoords='offset points', ha='center', va='bottom',
                        fontsize=FS_GROUP, fontweight='bold', color=INK)
        _draw_block_rules(ax, datasets, series_list, n_cell, p['decoders'])

        direction = 'higher' if p['higher_better'] else 'lower'
        ax.annotate(f'raw metric: {direction} is better', xy=(0.5, 1), xycoords='axes fraction',
                    xytext=(0, 34), textcoords='offset points', ha='center', va='bottom',
                    fontsize=FS_NOTE, color=INK_3)
        unit_txt = ', %' if p['unit'] == 'pp' else ''
        options = [f"{p['title']} ({p['short']}{unit_txt})",
                   f"{p['title']} ({p['short']})", f"{p['short']}{unit_txt}", p['short']]
        title_txt = next((t for t in options if len(t) * FS_TITLE * 0.58 <= w * 72), p['short'])
        ax.annotate(title_txt, xy=(0.5, 1), xycoords='axes fraction',
                    xytext=(0, 44), textcoords='offset points', ha='center', va='bottom',
                    fontsize=FS_TITLE, fontweight='bold', color=INK)

        # what the ramp spans in this panel, one line per dataset block
        if shade_scope == 'dataset':
            note = ['ramp spans, per dataset block:']
            note += [f'{DATASET_SHORT.get(ds, DATASET_LABELS[ds])} '
                     f'{fmt.format(p["ranges"][ds][0])}–{fmt.format(p["ranges"][ds][1])}'
                     for ds in datasets if ds in p['ranges']]
        elif shade_scope == 'row':
            note = ['each row has its own ramp:', 'compare rows by value,', 'not by colour']
        else:
            lo, hi = p['ranges'].get('panel', (np.nan, np.nan))
            note = [f'one ramp for the whole panel: {fmt.format(lo)}–{fmt.format(hi)}']
        note += [f'{DECODER_SHORT[d]} decoding omitted:'
                 f'\n{p["short"]} is constant, by construction' for d in p['dropped']]
        if show_footer:
            ax.annotate('\n'.join(note), xy=(0, 0), xycoords='axes fraction',
                        xytext=(0, -6), textcoords='offset points', ha='left', va='top',
                        linespacing=1.3, fontsize=FS_NOTE, style='italic', color=INK_3)
        elif p['dropped']:
            ax.annotate(
                '\n'.join(f'{DECODER_SHORT[d]} decoding omitted:' for d in p['dropped'])
                + f'\n{p["short"]} ≡ 0 for every cell, by construction',
                xy=(1, 0), xycoords='axes fraction', xytext=(0, -7),
                textcoords='offset points', ha='right', va='top', linespacing=1.25,
                fontsize=FS_NOTE, style='italic', color=INK_3)
        x += w + gap

    _draw_row_labels(panels[0]['ax'], rows, datasets, series_list, dataset_offset_pt)

    # One generic ramp, since each shading block has its own numerical limits.
    # Its structure mirrors the relative grid's footer: scale, endpoint labels,
    # and a concise statement of what the colour encodes.
    bar_w = min(3.1, width_in - gutter - pad_r)
    bar_x = gutter + (width_in - gutter - pad_r - bar_w) / 2
    bar_y = top_pad + head + grid_h + note_h + 0.20
    cax = fig.add_axes([bar_x / width_in, 1 - bar_y / height,
                        bar_w / width_in, 0.115 / height])
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect='auto', cmap=SEQUENTIAL,
               extent=(0, 1, 0, 1))
    cax.set_yticks([])
    scope_txt = {'dataset': 'its dataset block', 'row': 'its row',
                 'panel': 'the panel'}[shade_scope]
    cax.set_xticks([0, 1], [f'worst in {scope_txt}', 'best'])
    cax.tick_params(axis='x', labelsize=FS_NOTE, colors=INK_2, length=2, pad=1.5)
    cax.set_xlabel(f'raw metric shading, normalised within {scope_txt}; darker is better',
                   fontsize=FS_NOTE, color=INK_2, labelpad=3)
    for spine in cax.spines.values():
        spine.set_visible(False)
    if show_footer:
        spread_txt = ('mean ± sample SD over ' if show_spread else 'mean over ')
        fig.text(bar_x / width_in + bar_w / (2 * width_in), 1 - (bar_y + 0.46) / height,
                 f'{spread_txt}{seeds_phrase(A)}; bold type marks the '
                 'checkpoint’s own native readout',
                 ha='center', va='top', fontsize=FS_NOTE, color=INK_2)

    _caption('Raw test values of every inference cell',
             _scope_note(datasets, series_list, decoders,
                         'metrics: ' + ', '.join(METRIC_SPECS[m][1] for m in metrics)))
    if show_footer and not THESIS_STYLE:
        fig.suptitle('Where the inference cells actually land',
                     fontsize=11, fontweight='bold', color=INK, y=1 - 0.06 / height)
        fig.text(0.5, 0.02,
                 'Raw test values, not gains: the same grid as the effect figure, so a '
                 'gain there can be read here\nagainst the level it sits on and against '
                 'the other rows on the same dataset.',
                 ha='center', va='bottom', fontsize=FS_NOTE, color=INK_2)
    save_figure(fig, save_path)
    plt.show(); plt.close(fig)
    if save_path:
        latex_block(save_path,
                    'Absolute post-hoc test values: every readout$\\times$transform cell '
                    'for every row, mean '
                    + ('$\\pm$ sample standard deviation ' if show_spread else '')
                    + f'over {seeds_phrase(A)}, each decoder read from its own '
                    'validation-selected checkpoint. Shading is normalised separately '
                    + {'dataset': 'inside each dataset block, because absolute levels are '
                                  'not comparable across datasets',
                       'row': 'inside each row',
                       'panel': 'over the whole panel'}[shade_scope]
                    + ', and is direction-adjusted so that darker is always '
                    'better; ' + ('the span of every ramp is printed under the panel and '
                                  if show_footer else '')
                    + 'the exact value is printed in every tile. Bold type marks the cell that '
                    'reproduces the checkpoint\'s own inference.' + caption_note,
                    'posthoc-absolute-grid')


# Categorical colours for the mechanisms, used when a figure holds more than one.
FAMILY_COLORS = {
    'baseline': '#52514e',
    'lexmode': '#2a78d6',
    'hcc': '#eb6834',
    'lhprojection': '#9467bd',
    'subspace_supervision': '#1baf7a',
}
# Categorical colours for the four inference cells.
INFERENCE_COLORS = {
    'node_score': '#2563eb',
    'subspace_norm': '#16a34a',
    'hcc_node_score': '#d97706',
    'hcc_subspace_norm': '#7c3aed',
}
CELL_MARKERS = {'node_score': 'o', 'subspace_norm': 's',
                'hcc_node_score': '^', 'hcc_subspace_norm': 'D'}


def _colour_key(series_list):
    """Colour by mechanism when several are shown, otherwise by model."""
    families = {series_family(s) for s in series_list} - {None}
    if len(families) > 1:
        return 'family', {s: FAMILY_COLORS.get(series_family(s), INK_2) for s in series_list}
    return 'model', {s: series_color(s) for s in series_list}


def accuracy_consistency_map(absolute_table, datasets=None, series=None,
                             save_path=None, width_in=None, caption_note=''):
    """TICE against FPA under independent decoding, native cell -> every other cell.

    Consistency is the x axis and accuracy the y axis, so the reader scans the
    trade-off the way the thesis states it: does a readout buy consistency, and
    what does it cost in accuracy. Top-down decoding is excluded on purpose: it
    is consistent by construction, so its TICE is identically zero.
    """
    width_in = TEXT_WIDTH_IN if width_in is None else width_in
    if absolute_table is None or absolute_table.empty:
        print('Nothing to plot.')
        return
    A = absolute_table[absolute_table['decoder'].eq('independent')]
    if A.empty:
        print('This figure needs independent decoding rows.')
        return
    wide = A.pivot_table(index=['dataset', 'series', 'inference', 'is_native'],
                         columns='metric_family', values='mean_value').reset_index()
    if not {'fpa', 'tice'}.issubset(wide.columns):
        print('Need both fpa and tice selected for this figure.')
        return
    datasets = [d for d in DATASET_ORDER if d in set(wide['dataset'])] \
        if datasets is None else [d for d in datasets if d in set(wide['dataset'])]
    series_list = order_series(set(wide['series'])) if series is None \
        else [s for s in series if s in set(wide['series'])]
    if not (datasets and series_list):
        print('Nothing to plot.')
        return
    colour_by, colours = _colour_key(series_list)
    floor = max(min(wide['tice'][wide['tice'] > 0], default=0.3) * 0.6, 0.05)
    clamped = bool((wide['tice'] < floor).any())

    # shared axes: the panels are directly comparable, and only one set of tick
    # labels is needed, which leaves more room for the panels themselves
    x_lo, x_hi = floor * 0.75, wide['tice'].max() * 1.5
    y_lo, y_hi = wide['fpa'].min(), wide['fpa'].max()
    y_pad = 0.08 * (y_hi - y_lo)

    left, right, top, gap = 0.52, 0.04, 0.30, 0.13
    legend_h = 0.34 + 0.17 * (1 + int(np.ceil(len(series_list) / 4)))
    foot_h = 0.30 if not THESIS_STYLE else 0.0
    panel_w = (width_in - left - right - gap * (len(datasets) - 1)) / len(datasets)
    panel_h = panel_w * 1.02
    axis_lab = 0.44
    height = top + panel_h + axis_lab + legend_h + foot_h

    fig = plt.figure(figsize=(width_in, height), layout='none')
    axes = []
    for n, ds in enumerate(datasets):
        ax = fig.add_axes([(left + n * (panel_w + gap)) / width_in,
                           1 - (top + panel_h) / height, panel_w / width_in,
                           panel_h / height])
        axes.append(ax)
        d = wide[wide['dataset'].eq(ds)]
        for sr in series_list:
            dm = d[d['series'].eq(sr)]
            if dm.empty:
                continue
            colour = colours[sr]
            nat = dm[dm['is_native']]
            if nat.empty:
                continue
            nx, ny = max(float(nat['tice'].iloc[0]), floor), float(nat['fpa'].iloc[0])
            for _, r in dm[~dm['is_native']].iterrows():
                x, y = max(float(r['tice']), floor), float(r['fpa'])
                ax.annotate('', xy=(x, y), xytext=(nx, ny),
                            arrowprops=dict(arrowstyle='-|>', color=colour, alpha=.34,
                                            linewidth=1.1, shrinkA=5, shrinkB=5))
                ax.scatter(x, y, s=36, marker=CELL_MARKERS[r['inference']], facecolor=colour,
                           edgecolor='white', linewidth=1.0, alpha=.95, zorder=3)
            ax.scatter(nx, ny, s=62, marker=CELL_MARKERS[nat['inference'].iloc[0]],
                       facecolor='white', edgecolor=colour, linewidth=1.6, zorder=4)
        ax.set_xscale('log')
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.set_title(DATASET_LABELS[ds], fontsize=FS_GROUP + 0.4, fontweight='bold',
                     color=INK, pad=4)
        ax.grid(color=GRID_LINE, linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=FS_TICK, colors=INK_2, length=2.5, pad=1.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(AXIS_LINE)
        if n:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel('FPA (%)   ↑ better', fontsize=FS_GROUP, color=INK_2, labelpad=2)
    x_mid = (left + (width_in - left - right) / 2) / width_in
    fig.text(x_mid, (legend_h + foot_h + 0.17) / height,
             'Tree inconsistency, TICE (%, log scale)   ← better',
             ha='center', va='bottom', fontsize=FS_GROUP, color=INK_2)
    fig.text(x_mid, (legend_h + foot_h + 0.04) / height,
             f'independent decoding, mean of {seeds_phrase(absolute_table)}',
             ha='center', va='bottom', fontsize=FS_NOTE, color=INK_3)

    if colour_by == 'family':
        seen, series_keys = set(), []
        for sr in series_list:
            family = series_family(sr)
            if family in seen:
                continue
            seen.add(family)
            series_keys.append(Line2D([0], [0], marker='o', linewidth=0, markersize=5,
                                      markerfacecolor=colours[sr], markeredgecolor='white',
                                      label=FAMILY_LABELS.get(family, str(family))))
    else:
        seen, series_keys = set(), []
        for sr in series_list:
            model = series_model(sr)
            if model in seen:
                continue
            seen.add(model)
            series_keys.append(Line2D([0], [0], marker='o', linewidth=0, markersize=5,
                                      markerfacecolor=colours[sr], markeredgecolor='white',
                                      label=MODEL_LABELS.get(model, str(model))))
    cell_keys = [Line2D([0], [0], marker=CELL_MARKERS[c], linewidth=0, markersize=5,
                        markerfacecolor=INK_3, markeredgecolor='white',
                        label=CELL_LABELS[c].replace('\n', ' ')) for c in CELL_ORDER]
    cell_keys += [Line2D([0], [0], marker='o', linewidth=0, markersize=6,
                         markerfacecolor='white', markeredgecolor=INK_2,
                         markeredgewidth=1.4, label='hollow = native cell')]
    for row, keys in enumerate((cell_keys, series_keys)):
        legend = fig.legend(handles=keys, loc='lower center',
                            ncol=min(len(keys), 5), frameon=False,
                            fontsize=FS_NOTE + 0.4, handletextpad=.35, columnspacing=1.3,
                            bbox_to_anchor=(0.5, (foot_h + 0.04 + row * 0.17) / height))
        for text in legend.get_texts():
            text.set_color(INK_2)
        fig.add_artist(legend)
    _caption('What each inference cell trades: consistency against accuracy',
             _scope_note(datasets, series_list, ['independent']))
    if not THESIS_STYLE:
        fig.suptitle('What each inference cell trades: consistency against accuracy',
                     fontsize=11, fontweight='bold', color=INK, y=1 - 0.04 / height)
        fig.text(.5, .01,
                 'Arrows start at each checkpoint’s native readout. Up-and-left is a free '
                 'win; straight left buys consistency at no accuracy cost; down is a loss.',
                 ha='center', va='bottom', fontsize=FS_NOTE, color=INK_2)
    save_figure(fig, save_path)
    plt.show(); plt.close(fig)
    if save_path:
        latex_block(save_path,
                    'What each inference cell trades. Absolute test TICE (horizontal, log scale, '
                    'lower is better) against full-path accuracy (vertical, higher is better) '
                    f'under independent decoding, averaged over {seeds_phrase(absolute_table)}. '
                    'Each arrow starts at the checkpoint\'s own native readout (hollow marker) '
                    'and ends at one alternative cell, so up-and-left is a free win, straight '
                    'left buys consistency at no accuracy cost, and downward movement is an '
                    'accuracy loss. Top-down decoding is omitted because it is consistent by '
                    'construction (TICE $\\equiv 0$)'
                    + (f'; TICE values below {floor:.2g}\\% are clamped to the axis floor so that '
                       'they remain visible on the log scale' if clamped else '')
                    + '.' + caption_note,
                    'posthoc-accuracy-consistency')


# A relative change is a ratio, so it explodes when the independent-decoding
# baseline is near zero: HT-CapsNet on FGVC-Aircraft moves 0.8% -> 2.3% FPA and
# scores +220%. Rows below this baseline are still drawn and still labelled with
# their value, but they are drawn off-scale rather than allowed to set the axis.
MIN_BASELINE_FPA = 5.0


def relative_decoder_gains(frame, datasets=None, series=None):
    """Per-seed relative FPA change from independent to top-down decoding.

    For every (dataset, series, readout, seed) the paired quantity is

        rel = 100 * (FPA_topdown - FPA_independent) / FPA_independent,

    i.e. the relative improvement top-down decoding buys over independent
    decoding, not a percentage-point difference. It is formed per seed and only
    then averaged, so the reported spread is the spread of the paired effect.
    Each decoder is read from its own validation-selected checkpoint, so the
    pairing is by dataset, series, readout and seed, not by parameter state.
    """
    required = {'dataset', 'series', 'seed', 'checkpoint_mode', 'decoder',
                'inference', 'metric_family', 'value'}
    if frame.empty or not required.issubset(frame.columns):
        print('No compatible rows for the readout/decoder comparison.')
        return pd.DataFrame()

    selected = frame[
        frame['inference'].isin(tuple(READOUT_LABELS))
        & frame['metric_family'].eq('fpa')
        & frame['decoder'].isin(DECODERS)
        & frame['decoder'].eq(frame['checkpoint_mode'])
    ].copy()
    datasets = [d for d in DATASET_ORDER if d in set(selected['dataset'])] \
        if datasets is None else [d for d in datasets if d in set(selected['dataset'])]
    series_list = order_series(set(selected['series'])) if series is None \
        else [s for s in series if s in set(selected['series'])]
    selected = selected[selected['dataset'].isin(datasets) & selected['series'].isin(series_list)]
    selected['fpa_percent'] = 100.0 * selected['value']

    per_seed = (selected.pivot_table(
        index=['dataset', 'series', 'inference', 'seed'], columns='decoder',
        values='fpa_percent', aggfunc='first').reset_index())
    if not set(DECODERS).issubset(per_seed.columns):
        print('Need matched independent and top-down FPA rows.')
        return pd.DataFrame()
    per_seed = per_seed.dropna(subset=list(DECODERS))
    per_seed = per_seed[per_seed['independent'] > 0]
    per_seed['decoder_delta_pp'] = per_seed['topdown'] - per_seed['independent']
    per_seed['decoder_delta_rel'] = 100.0 * per_seed['decoder_delta_pp'] / per_seed['independent']

    rows = []
    for (dataset, sr, inference), group in per_seed.groupby(['dataset', 'series', 'inference']):
        ind_mean, ind_std, _ = sample_stats(group['independent'].tolist())
        top_mean, _, _ = sample_stats(group['topdown'].tolist())
        pp_mean, _, _ = sample_stats(group['decoder_delta_pp'].tolist())
        rel_mean, rel_std, seeds = sample_stats(group['decoder_delta_rel'].tolist())
        # same convention as the effect grid: flag an effect whose sign is not
        # the same for every seed, counting an exactly zero seed as no effect
        effects = group['decoder_delta_rel'].to_numpy()
        effects = effects[np.abs(effects) > 1e-9]
        rows.append({
            'dataset': dataset, 'series': sr, 'inference': inference,
            'independent_fpa': ind_mean, 'independent_std': ind_std,
            'topdown_fpa': top_mean, 'delta_pp': pp_mean,
            'relative_gain': rel_mean, 'relative_std': rel_std, 'seeds': seeds,
            'sign_agrees': len(effects) < 2 or bool(np.all(effects > 0) or np.all(effects < 0)),
            # a ratio against a collapsed baseline is arithmetic, not a result
            'weak_baseline': bool(ind_mean < MIN_BASELINE_FPA),
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        print('No paired seeds for the readout/decoder comparison.')
        return summary
    summary['_dataset_rank'] = summary['dataset'].map({v: i for i, v in enumerate(datasets)})
    summary['_series_rank'] = summary['series'].map({v: i for i, v in enumerate(series_list)})
    summary['_inference_rank'] = summary['inference'].map(
        {v: i for i, v in enumerate(READOUT_LABELS)})
    return (summary.sort_values(['_inference_rank', '_dataset_rank', '_series_rank'])
            .drop(columns=['_dataset_rank', '_series_rank', '_inference_rank'])
            .reset_index(drop=True))


def relative_gain_limits(table, pad=0.30):
    """An x range that holds every bar with a usable baseline, plus its error bar.

    Rows whose independent-decoding FPA is below MIN_BASELINE_FPA are ignored
    here: one collapsed run would otherwise stretch the axis by a factor of 20
    and flatten every real effect to a hairline.
    """
    usable = table[~table['weak_baseline']] if 'weak_baseline' in table else table
    if usable.empty:
        usable = table
    lo, hi = 0.0, 0.0
    for _, row in usable.iterrows():
        gain = float(row['relative_gain'])
        if not np.isfinite(gain):
            continue
        std = float(row['relative_std']) if np.isfinite(row['relative_std']) else 0.0
        lo = min(lo, gain - std)
        hi = max(hi, gain + std)
    span = max(hi - lo, 1.0)
    # the value label sits outside the bar, so leave room on whichever side
    # carries real bars; a side holding only hairlines gets none, since those
    # labels are placed across the zero line anyway
    minor = 0.10 * max(abs(lo), abs(hi))
    return (lo - (pad if lo < -minor else 0.04) * span,
            hi + (pad if hi > minor else 0.04) * span)


def decoder_gain_figure(summary, readout, datasets=None, series=None, save_path=None,
                        width_in=None, xlim=None, show_footer=True, caption_note=''):
    """Relative independent -> top-down FPA change for one readout.

    One panel per dataset, one row per series, a bar from the zero line to the
    mean relative gain with the sample SD of the paired per-seed effect. Colour
    carries the sign with the same polarity as the effect grid: blue is a gain
    for top-down decoding, red a loss. Set ``show_footer=False`` for a compact
    thesis figure without the explanatory text below the panels.
    """
    width_in = TEXT_WIDTH_IN if width_in is None else width_in
    data = summary[summary['inference'].eq(readout)] if not summary.empty else summary
    if data.empty:
        print(f'Nothing to plot for the {readout} readout.')
        return
    datasets = [d for d in DATASET_ORDER if d in set(data['dataset'])] \
        if datasets is None else [d for d in datasets if d in set(data['dataset'])]
    series_list = order_series(set(data['series'])) if series is None \
        else [s for s in series if s in set(data['series'])]
    if not (datasets and series_list):
        print(f'Nothing to plot for the {readout} readout.')
        return

    if xlim is None:
        xlim = relative_gain_limits(data)
    x_min, x_max = xlim

    offscale = [row for _, row in data.iterrows()
                if np.isfinite(row['relative_gain'])
                and (bool(row['weak_baseline'])
                     or not (x_min <= row['relative_gain'] <= x_max))]

    left = max(0.78, _row_geometry(series_list)[0] - 0.20)
    right, top, gap = 0.10, 0.50, 0.14
    bottom = (0.62 + (0.15 if offscale else 0.0)) if show_footer else 0.18
    panel_w = (width_in - left - right - gap * (len(datasets) - 1)) / len(datasets)
    panel_h = max(1.08, 0.36 * len(series_list))
    foot_h = 0.24 if show_footer and not THESIS_STYLE else 0.0
    height = top + panel_h + bottom + foot_h
    fig = plt.figure(figsize=(width_in, height), layout='none')

    bar_h = 0.42
    lookup = data.set_index(['dataset', 'series']).to_dict('index')
    for panel, dataset in enumerate(datasets):
        ax = fig.add_axes([(left + panel * (panel_w + gap)) / width_in,
                           1 - (top + panel_h) / height,
                           panel_w / width_in, panel_h / height])
        for y, sr in enumerate(series_list):
            row = lookup.get((dataset, sr))
            if row is None or not np.isfinite(row['relative_gain']):
                continue
            gain = float(row['relative_gain'])
            std = float(row['relative_std']) if np.isfinite(row['relative_std']) else 0.0
            colour = POS_HUE if gain >= 0 else NEG_HUE
            # a bar past the axis limit is drawn to the edge and marked, so the
            # run is still visible without the axis being stretched to reach it
            clipped = not (x_min <= gain <= x_max)
            weak = bool(row['weak_baseline'])
            drawn = float(np.clip(gain, x_min, x_max))
            ax.barh(y, drawn, height=bar_h, color=colour, alpha=0.88,
                    edgecolor='white', linewidth=0.6, zorder=2)
            if clipped:
                ax.plot(drawn, y, marker='>' if gain > 0 else '<', markersize=4.2,
                        color=colour, markeredgecolor='white', markeredgewidth=0.5,
                        clip_on=False, zorder=5)
            # a collapsed baseline also blows up the spread: a +-35% whisker would
            # be drawn straight across the panel, so it is listed instead of drawn
            show_error = std > 0 and not clipped and not weak
            if show_error:
                ax.errorbar(gain, y, xerr=std, fmt='none', ecolor=INK_2,
                            elinewidth=0.8, capsize=1.8, zorder=4)
            label = f'{gain:+.1f}%'
            if not bool(row['sign_agrees']):
                label += '†'
            tip = gain + (std if gain >= 0 else -std) if show_error else drawn
            label_w = len(label) * FS_TILE * 0.62 * (x_max - x_min) / (panel_w * 72)
            forward, back = ('left', 'right') if gain >= 0 else ('right', 'left')
            if (tip + label_w <= x_max) if gain >= 0 else (tip - label_w >= x_min):
                anchor, side, colour_txt = tip, forward, INK_2      # outside the bar
            elif abs(drawn) >= label_w:
                anchor, side, colour_txt = drawn, back, _text_on(colour)   # inside it
            else:
                # a hairline bar with no room on its own side: the only space left
                # is across the zero line, which is why the sign is printed too
                anchor, side, colour_txt = 0.0, back, INK_2
            # a clipped bar ends in an arrow head, so its label starts further in
            offset = 8 if clipped else 3
            ax.annotate(label, xy=(anchor, y),
                        xytext=(offset if side == 'left' else -offset, 0),
                        textcoords='offset points', va='center', ha=side,
                        fontsize=FS_TILE, color=colour_txt, zorder=6, clip_on=False)
        ax.axvline(0, color=INK_3, linewidth=0.9, zorder=3)
        ax.grid(axis='x', color=GRID_LINE, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(x_min, x_max); ax.set_ylim(len(series_list) - 0.5, -0.5)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda value, _pos: f'{value:g}'))
        ax.set_yticks(range(len(series_list)))
        ax.set_yticklabels([series_label(s, short=True) for s in series_list]
                           if panel == 0 else [])
        ax.set_title(DATASET_LABELS[dataset], fontsize=FS_GROUP + 0.4,
                     fontweight='bold', color=INK, pad=4)
        ax.tick_params(axis='x', labelsize=FS_TICK, colors=INK_2, length=2.5, pad=1.5)
        ax.tick_params(axis='y', labelsize=FS_TICK, colors=INK_2,
                       length=2.5 if panel == 0 else 0, pad=1.5)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color(AXIS_LINE)

    # the two figures are near-identical apart from the readout, so name the
    # readout inside the image and not only in the caption
    if THESIS_STYLE:
        fig.text(left / width_in, 1 - 0.16 / height,
                 f'Readout: {READOUT_LABELS[readout]}', ha='left', va='top',
                 fontsize=FS_TITLE, fontweight='bold', color=INK)

    if show_footer:
        base = foot_h + (0.24 if offscale else 0.09)  # inches from the figure bottom
        x_mid = (left + (width_in - left - right) / 2) / width_in
        fig.text(x_mid, (base + 0.17) / height,
                 'Top-down decoding relative to independent decoding, '
                 '$100\\,(\\mathrm{td}-\\mathrm{ind})/\\mathrm{ind}$ (%)',
                 ha='center', va='bottom', fontsize=FS_GROUP, color=INK_2)
        fig.text(x_mid, base / height,
                 f'paired per seed over {seeds_phrase(data)} · bars right of zero: '
                 'top-down wins · error bars: sample SD of the paired effect',
                 ha='center', va='bottom', fontsize=FS_NOTE, color=INK_3)
        if offscale:
            parts = [f"{series_label(r['series'], short=True)} on {DATASET_SHORT[r['dataset']]} "
                     f"{r['relative_gain']:+.1f}% from {r['independent_fpa']:.1f}%"
                     for r in offscale]
            # this line is the longest, so it is centred on the figure, not the panels
            fig.text(0.5, (foot_h + 0.07) / height,
                     f'collapsed baseline (< {MIN_BASELINE_FPA:g}% independent FPA), '
                     'error bar omitted: ' + '; '.join(parts),
                     ha='center', va='bottom', fontsize=FS_NOTE, color=INK_3)
    _caption(f'What top-down decoding buys under the {READOUT_LABELS[readout]} readout',
             _scope_note(datasets, series_list, extra='relative FPA change, paired per seed'))
    if not THESIS_STYLE:
        fig.suptitle(f'What top-down decoding buys under the {READOUT_LABELS[readout]} readout',
                     fontsize=11, fontweight='bold', color=INK, y=1 - 0.05 / height)
        if show_footer:
            fig.text(0.5, 0.01,
                     'Relative change, not percentage points: +5% means 5% more full paths '
                     'are correct, whatever the absolute level.',
                     ha='center', va='bottom', fontsize=FS_NOTE, color=INK_2)
    save_figure(fig, save_path)
    plt.show(); plt.close(fig)
    if save_path:
        latex_block(save_path,
                    'Relative gain of top-down over independent decoding under the '
                    f'untransformed {READOUT_LABELS[readout]} readout. Each bar is '
                    '$100\\,(\\mathrm{FPA}_{\\text{top-down}} - \\mathrm{FPA}_{'
                    '\\text{independent}}) / \\mathrm{FPA}_{\\text{independent}}$, formed '
                    f'per seed and then averaged over {seeds_phrase(data)}; error bars are '
                    'the sample standard deviation of that paired per-seed effect and '
                    '$\\dagger$ marks an effect whose sign is not the same for every seed. '
                    'The quantity is a relative change, not a percentage-point difference, '
                    'so the same bar length means the same fraction of full paths recovered '
                    'at every absolute accuracy level. The two readout figures share one '
                    'horizontal range and are directly comparable. Each decoder is read from '
                    'its own validation-selected checkpoint. TICE is not shown because '
                    'top-down decoding makes it identically zero by construction.'
                    + ('' if not offscale else
                       ' A run whose independent-decoding FPA is below '
                       f'{MIN_BASELINE_FPA:g}\\% neither sets the axis nor shows an error '
                       'bar, because the ratio and its spread then describe the collapsed '
                       'baseline rather than the decoder; such a bar is drawn to the axis '
                       'edge with an arrow and its value is printed inside it'
                       + (', with its baseline listed under the axis' if show_footer else '')
                       + '.') + caption_note,
                    f'posthoc-decoder-gain-{readout.replace("_", "-")}')


def best_inference_chart(absolute_table, metric='fpa', decoder='independent',
                         datasets=None, series=None, save_path=None, width_in=None,
                         caption_note=''):
    """Every inference cell of every mechanism on one axis, per dataset.

    This is the figure the per-mechanism notebooks cannot draw: it puts each
    (model, mechanism) row on its own line and marks all four cells on it, so
    the best readout *anywhere* — not only the best within one checkpoint — is
    read off directly, together with how far it sits from the row's own native
    cell and from the other mechanisms.

    Each dataset panel has its own x range, because absolute levels are not
    comparable across datasets. The dashed rule marks the best value in the
    panel, so a row that never reaches it is visibly behind.
    """
    width_in = TEXT_WIDTH_IN if width_in is None else width_in
    if absolute_table is None or absolute_table.empty:
        print('Nothing to plot.')
        return
    title, short, unit, higher_better, linthresh = METRIC_SPECS[metric]
    table = absolute_table[absolute_table['metric_family'].eq(metric)
                           & absolute_table['decoder'].eq(decoder)]
    if table.empty:
        print(f'No rows for {metric} under {decoder} decoding.')
        return
    datasets = [d for d in DATASET_ORDER if d in set(table['dataset'])] \
        if datasets is None else [d for d in datasets if d in set(table['dataset'])]
    series_list = order_series(set(table['series'])) if series is None \
        else [s for s in series if s in set(table['series'])]
    cells = [c for c in CELL_ORDER if c in set(table['inference'])]
    if not (datasets and series_list and cells):
        print('Nothing to plot.')
        return

    values = table.set_index(['dataset', 'series', 'inference'])['mean_value'].to_dict()
    native_of = (table[table['is_native']].groupby(['dataset', 'series'])['inference']
                 .first().to_dict())
    fmt = '{:.2f}' if unit == 'raw' else '{:.1f}'

    left = max(0.90, _row_geometry(series_list)[0] - 0.10)
    right, top, gap = 0.12, 0.44, 0.20
    legend_h, foot_h = 0.40, 0.26 if not THESIS_STYLE else 0.0
    bottom = 0.52 + legend_h + foot_h
    panel_w = (width_in - left - right - gap * (len(datasets) - 1)) / len(datasets)
    panel_h = max(1.2, 0.26 * len(series_list))
    height = top + panel_h + bottom

    fig = plt.figure(figsize=(width_in, height), layout='none')
    for panel, dataset in enumerate(datasets):
        ax = fig.add_axes([(left + panel * (panel_w + gap)) / width_in,
                           1 - (top + panel_h) / height,
                           panel_w / width_in, panel_h / height])
        panel_values = [values.get((dataset, sr, c)) for sr in series_list for c in cells]
        panel_values = [v for v in panel_values if v is not None and np.isfinite(v)]
        if not panel_values:
            ax.set_axis_off()
            continue
        lo, hi = min(panel_values), max(panel_values)
        span = max(hi - lo, 1e-9)
        # the right margin carries each row's best value as text, so it is
        # reserved rather than shared with the markers
        x_lo, x_hi = lo - 0.10 * span, hi + 0.32 * span
        # a heavy-tailed metric flattens on a linear axis: TICE spans 0.0 to 100,
        # so the cells that differ by tenths of a point pile up on the zero line
        if metric in ABS_SYMLOG_METRICS:
            ax.set_xscale('symlog', linthresh=linthresh)
            # headroom is multiplicative on a log axis: the right margin still has
            # to hold the row's printed value without touching the last marker
            x_lo, x_hi = min(0.0, lo), (hi * 8.0 if hi > 0 else 1.0)
        best = max(panel_values) if higher_better else min(panel_values)

        ax.axvline(best, color=INK_3, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        for y, sr in enumerate(series_list):
            row_values = [(c, values.get((dataset, sr, c))) for c in cells]
            row_values = [(c, v) for c, v in row_values if v is not None and np.isfinite(v)]
            if not row_values:
                continue
            xs = [v for _, v in row_values]
            ax.plot([min(xs), max(xs)], [y, y], color=AXIS_LINE, linewidth=1.0, zorder=2)
            for cell, value in row_values:
                is_native = native_of.get((dataset, sr)) == cell
                is_best = abs(value - best) < 1e-9
                ax.scatter(value, y, s=34 if not is_best else 52,
                           marker=CELL_MARKERS[cell],
                           facecolor='white' if is_native else INFERENCE_COLORS[cell],
                           edgecolor=INFERENCE_COLORS[cell] if is_native else 'white',
                           linewidth=1.5 if is_native else 0.8,
                           zorder=4 if is_best else 3)
            row_best = max(xs) if higher_better else min(xs)
            ax.annotate(fmt.format(row_best), xy=(1.0, y),
                        xycoords=('axes fraction', 'data'), xytext=(-2, 0),
                        textcoords='offset points', ha='right', va='center',
                        fontsize=FS_NOTE, color=INK_2, zorder=5)
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(len(series_list) - 0.5, -0.5)
        ax.set_yticks(range(len(series_list)))
        ax.set_yticklabels([series_label(s, short=True) for s in series_list]
                           if panel == 0 else [])
        ax.grid(axis='x', color=GRID_LINE, linewidth=0.6)
        ax.set_axisbelow(True)
        if metric not in ABS_SYMLOG_METRICS:
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        else:
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                lambda value, _pos: f'{value:g}'))
        ax.set_title(DATASET_LABELS[dataset], fontsize=FS_GROUP + 0.4, fontweight='bold',
                     color=INK, pad=4)
        ax.tick_params(axis='x', labelsize=FS_TICK, colors=INK_2, length=2.5, pad=1.5)
        ax.tick_params(axis='y', labelsize=FS_TICK, colors=INK_2,
                       length=2.5 if panel == 0 else 0, pad=1.5)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color(AXIS_LINE)
        # hairline between models, so a mechanism block is visible as one group
        models = [series_model(s) for s in series_list]
        for k in range(1, len(series_list)):
            if models[k] != models[k - 1]:
                ax.axhline(k - 0.5, color=AXIS_LINE, linewidth=0.7, zorder=1)

    unit_txt = ' (%)' if unit == 'pp' else ''
    direction = '↑ better' if higher_better else '← better'
    x_mid = (left + (width_in - left - right) / 2) / width_in
    scale_txt = ', symmetric-log' if metric in ABS_SYMLOG_METRICS else ''
    fig.text(x_mid, (legend_h + foot_h + 0.18) / height,
             f'{title} ({short}){unit_txt}{scale_txt}, {DECODER_SHORT[decoder]} decoding'
             f'   {direction}',
             ha='center', va='bottom', fontsize=FS_GROUP, color=INK_2)
    fig.text(x_mid, (legend_h + foot_h + 0.05) / height,
             f'mean of {seeds_phrase(table)} · the number at the right of each row is that '
             'row’s best cell · dashed rule: best in the panel',
             ha='center', va='bottom', fontsize=FS_NOTE, color=INK_3)
    keys = [Line2D([0], [0], marker=CELL_MARKERS[c], linewidth=0, markersize=5,
                   markerfacecolor=INFERENCE_COLORS[c], markeredgecolor='white',
                   label=CELL_LABELS[c].replace('\n', ' ')) for c in cells]
    keys += [Line2D([0], [0], marker='o', linewidth=0, markersize=6, markerfacecolor='white',
                    markeredgecolor=INK_2, markeredgewidth=1.4, label='hollow = native cell')]
    legend = fig.legend(handles=keys, loc='lower center', ncol=min(len(keys), 5),
                        frameon=False, fontsize=FS_NOTE + 0.4, handletextpad=.35,
                        columnspacing=1.3, bbox_to_anchor=(0.5, (foot_h + 0.06) / height))
    for text in legend.get_texts():
        text.set_color(INK_2)
    # the caption differs in kind between a one-mechanism and a cross-mechanism
    # figure, because only the latter compares separate training runs
    multi_family = len({series_family(s) for s in series_list} - {None}) > 1
    _caption(
        'Best inference cell across '
        + ('every mechanism' if multi_family else 'every model')
        + f' ({short})',
        _scope_note(datasets, series_list, [decoder]),
    )
    if not THESIS_STYLE:
        scope = 'every mechanism' if multi_family else 'every model'
        fig.suptitle(f'Best inference cell across {scope} ({short})',
                     fontsize=11, fontweight='bold', color=INK, y=1 - 0.04 / height)
        if multi_family:
            fig.text(0.5, 0.01,
                     'Rows are training mechanisms, not checkpoints of one model: a comparison '
                     'across rows mixes the mechanism with the readout.',
                     ha='center', va='bottom', fontsize=FS_NOTE, color=INK_2)
    save_figure(fig, save_path)
    plt.show(); plt.close(fig)
    if save_path:
        latex_block(save_path,
                    'Every inference cell of '
                    + ('every training mechanism' if multi_family else 'every model')
                    + f', {short} under {DECODER_SHORT[decoder]} decoding, mean over '
                    f'{seeds_phrase(table)}. Each row is one '
                    + ('(model, mechanism) pair' if multi_family else 'model')
                    + ' and carries all four readout$\\times$transform cells; the hollow '
                    'marker is the cell that reproduces that checkpoint\'s own inference and '
                    'the dashed rule marks the best value in the panel. Each dataset panel has '
                    'its own horizontal range, because absolute levels are not comparable '
                    'across datasets.'
                    + (' Rows come from different training runs, so a difference between rows '
                       'mixes the training mechanism with the inference rule and is matched by '
                       'seed only, unlike the within-checkpoint gains.' if multi_family else '')
                    + caption_note,
                    f'posthoc-best-inference-{metric.replace("_", "-")}')
