"""Second-stage Optuna grid for Hier-COS on CUB-200.

This search is based on the first CUB grid:

- advanced timm augmentation was consistently worse, so it is excluded;
- transform_lr_scale had weak signal, so it is fixed to 1.0;
- alpha controlled the FPA/TICE trade-off, so it is searched more densely;
- the original grid did not vary the global SGD/cosine learning rate, so this
  grid searches it directly.

The script reuses the original gridsearch helpers for frozen-config handling,
metric extraction, and Optuna bookkeeping.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gridsearch import hiercos_cub200_optuna as base


ALPHAS = [0.08, 0.1, 0.15, 0.2]
BASE_LRS = [0.05, 0.075, 0.1, 0.15]
SELECTION_MODES = base.SELECTION_MODES
OBJECTIVE_MODES = base.OBJECTIVE_MODES

DEFAULT_CONFIG = "configs/hiercos/hiercos_cub200.yaml"
DEFAULT_OUTPUT_ROOT = str(
    Path(os.environ.get("OUTPUTS_ROOT", "/scratch/g.saggini1/outputs"))
    / "gridsearch/hiercos_cub200_refined_optuna"
)
DEFAULT_STUDY_NAME = "hiercos_cub200_refined"


BASE_224_OVERRIDES = [
    "dataset.image_size=224",
    "dataset.transforms.use_timm=false",
    "dataset.transforms.manual.crop_mode=random_crop",
    "dataset.transforms.manual.resize_before_crop=true",
    "dataset.transforms.manual.resize_before_crop_size=224",
    "dataset.transforms.manual.random_crop_padding=4",
    "dataset.transforms.manual.random_crop_padding_mode=constant",
    "dataset.transforms.manual.interpolation=bilinear",
    "dataset.transforms.manual.random_horizontal_flip_prob=0.5",
    "dataset.transforms.eval.resize_mode=resize",
    "dataset.transforms.eval.resize_size=224",
    "dataset.transforms.eval.crop_ratio=1.0",
    "dataset.transforms.eval.interpolation=bilinear",
    "dataloader.batch_size=64",
    "dataloader.num_workers=8",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refined Optuna GridSampler search for CUB-200 Hier-COS.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Base training config.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Directory for trial artifacts.")
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME, help="Optuna study name.")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL. Defaults to sqlite:///<output-root>/study.db.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials to run. Defaults to the full refined grid size.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default="independent",
        help="Checkpoint-selection mode whose validation metric is optimized.",
    )
    parser.add_argument(
        "--objective-mode",
        choices=OBJECTIVE_MODES,
        default="fpa_tice_pareto",
        help="Use fpa_tice_pareto for maximize-FPA/minimize-TICE selection.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed passed to Optuna's GridSampler.")
    parser.add_argument(
        "--overwrite-frozen-config",
        action="store_true",
        help="Rewrite the study's frozen config snapshot if it already exists.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional train.train dotlist override. May be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without creating an Optuna study or launching training.",
    )
    return parser.parse_args()


def effective_study_name(args: argparse.Namespace) -> str:
    if args.objective_mode == "fpa_tice_pareto" and args.study_name == DEFAULT_STUDY_NAME:
        return f"{DEFAULT_STUDY_NAME}_fpa_tice_pareto"
    return str(args.study_name)


def search_space() -> Dict[str, List[Any]]:
    return {
        "model.alpha": ALPHAS,
        "base_lr": BASE_LRS,
    }


def all_grid_params() -> List[Dict[str, Any]]:
    keys = list(search_space().keys())
    values = [search_space()[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def resolve_n_trials(raw_n_trials: int | None) -> int:
    full_grid_size = len(all_grid_params())
    if raw_n_trials is None:
        return full_grid_size
    if raw_n_trials < 1:
        raise ValueError("--n-trials must be >= 1 when provided.")
    return int(raw_n_trials)


def trial_run_name(params: Mapping[str, Any]) -> str:
    alpha = base.float_tag(float(params["model.alpha"]))
    lr = base.float_tag(float(params["base_lr"]))
    return f"alpha_{alpha}__lr_{lr}__base224"


def build_overrides(params: Mapping[str, Any], output_root: Path, extra_overrides: Sequence[str]) -> List[str]:
    run_dir = output_root / trial_run_name(params)
    lr = float(params["base_lr"])
    return [
        f"model.alpha={params['model.alpha']}",
        f"optim.lr={lr}",
        f"scheduler.base_lr={lr}",
        *BASE_224_OVERRIDES,
        f"train.output_dir={run_dir}",
        *extra_overrides,
    ]


def print_dry_run(
    config: str,
    output_root: Path,
    study_name: str,
    extra_overrides: Sequence[str],
    n_trials: int,
) -> None:
    frozen_path = base.frozen_config_path(output_root, study_name)
    base.load_frozen_config_data(config)
    params_list = all_grid_params()[: min(int(n_trials), len(all_grid_params()))]
    for index, params in enumerate(params_list, start=1):
        run_dir = output_root / trial_run_name(params)
        overrides = build_overrides(params, output_root, extra_overrides)
        command = base.build_command(str(frozen_path), overrides)
        print(f"[dry-run {index:02d}/{len(params_list):02d}] {trial_run_name(params)}")
        print("  run_dir:", run_dir)
        print("  params:", params)
        print("  command:", " ".join(command))
    print("")
    print(f"[dry-run] listed_combinations={len(params_list)} total_grid_combinations={len(all_grid_params())}")
    print(f"[dry-run] output_root={output_root}")
    print(f"[dry-run] frozen_config={frozen_path}")
    print("[dry-run] fixed choices: augmentation=base, transform_lr_scale=1.0, image_size=224")


def objective_factory(args: argparse.Namespace, output_root: Path, frozen_config: Path):
    def objective(trial: Any):
        params = {
            "model.alpha": trial.suggest_categorical("model.alpha", ALPHAS),
            "base_lr": trial.suggest_categorical("base_lr", BASE_LRS),
        }
        run_dir = output_root / trial_run_name(params)
        overrides = build_overrides(params, output_root, args.override)
        command = base.build_command(str(frozen_config), overrides)
        base.set_trial_attrs(trial, run_dir=run_dir, overrides=overrides, command=command)

        run_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, cwd=base.REPO_ROOT, check=False)
        base.set_trial_attrs(
            trial,
            run_dir=run_dir,
            overrides=overrides,
            command=command,
            returncode=completed.returncode,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Training failed for trial {trial.number} with return code {completed.returncode}. "
                f"Run directory: {run_dir}"
            )

        test_metrics = base.load_test_metrics(run_dir)
        if args.objective_mode == "fpa_tice_pareto":
            objective_values = base.fpa_tice_objective_values(run_dir, test_metrics, args.selection_mode)
        else:
            objective_values = [base.selected_best_metric(test_metrics, args.selection_mode)]
        base.set_trial_attrs(
            trial,
            run_dir=run_dir,
            overrides=overrides,
            command=command,
            test_metrics=test_metrics,
            objective_values=objective_values,
            returncode=completed.returncode,
        )
        if args.objective_mode == "fpa_tice_pareto":
            return tuple(objective_values)
        return float(objective_values[0])

    return objective


def run_study(args: argparse.Namespace) -> None:
    import optuna

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    study_name = effective_study_name(args)
    frozen_config = base.prepare_frozen_config(args, output_root, study_name)
    storage = args.storage or base.default_storage(output_root)

    sampler = optuna.samplers.GridSampler(search_space(), seed=int(args.seed))
    study_kwargs: Dict[str, Any] = {
        "study_name": study_name,
        "storage": storage,
        "sampler": sampler,
        "pruner": optuna.pruners.NopPruner(),
        "load_if_exists": True,
    }
    if args.objective_mode == "fpa_tice_pareto":
        study_kwargs["directions"] = ["maximize", "minimize"]
    else:
        study_kwargs["direction"] = "maximize"
    study = optuna.create_study(**study_kwargs)
    study.optimize(objective_factory(args, output_root, frozen_config), n_trials=resolve_n_trials(args.n_trials))

    print("")
    print(f"[study] name={study.study_name}")
    print(f"[study] storage={storage}")
    print(f"[study] frozen_config={frozen_config}")
    print(f"[study] objective_mode={args.objective_mode}")
    print(f"[study] trials={len(study.trials)}")
    if args.objective_mode == "fpa_tice_pareto":
        print(f"[study] pareto_trials={len(study.best_trials)}")
        for trial in study.best_trials:
            print(f"[study] pareto trial={trial.number} values={trial.values} params={trial.params}")
    else:
        print(f"[study] best_value={study.best_value:.12g}")
        print(f"[study] best_params={study.best_params}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    n_trials = resolve_n_trials(args.n_trials)
    if args.dry_run:
        print_dry_run(args.config, output_root, effective_study_name(args), args.override, n_trials)
        return
    run_study(args)


if __name__ == "__main__":
    main()
