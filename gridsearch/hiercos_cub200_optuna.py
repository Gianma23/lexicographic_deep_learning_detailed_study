"""Optuna grid search for Hier-COS on CUB-200.

This module intentionally wraps the existing ``train.train`` CLI instead of
importing the training loop. Each trial runs in a fresh process, which keeps
CUDA state and training-side globals isolated across trials.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml


ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1]
TRANSFORM_LR_SCALES = [1.0, 0.1]
AUGMENTATIONS = ["base_aug", "adv_aug"]
SELECTION_MODES = ("topdown", "independent")
OBJECTIVE_MODES = ("selection_score", "fpa_tice_pareto")

DEFAULT_CONFIG = "configs/hiercos/hiercos_cub200.yaml"
DEFAULT_OUTPUT_ROOT = "/scratch/g.saggini1/outputs/gridsearch/hiercos_cub200_optuna"
DEFAULT_STUDY_NAME = "hiercos_cub200_baseline"
REPO_ROOT = Path(__file__).resolve().parents[1]


FROZEN_CONFIG_DEFAULTS: Dict[str, Any] = {
    "model.loss": "kl_reg",
    "model.weight_mode": "kl_leaf",
    "model.variant": "haframe_resnet50",
    "model.pretrained": True,
    "model.pool": "average",
    "model.backbone_lr_scale": 0.1,
    "optim.lr": 0.1,
    "scheduler.base_lr": 0.1,
    "dataset.transforms.mixup": 0.0,
    "dataset.transforms.cutmix": 0.0,
    "dataset.transforms.cutmix_minmax": None,
    "dataset.transforms.mixup_prob": 0.0,
    "dataset.transforms.mixup_switch_prob": 0.0,
}

BASE_AUG_OVERRIDES = [
    "dataset.image_size=224",
    "dataset.transforms.use_timm=false",
    "dataset.transforms.manual.crop_mode=random_crop",
    "dataset.transforms.manual.resize_before_crop=true",
    "dataset.transforms.manual.resize_before_crop_size=224",
    "dataset.transforms.manual.random_crop_padding=4",
    "dataset.transforms.manual.random_crop_padding_mode=constant",
    "dataset.transforms.manual.interpolation=bilinear",
    "dataset.transforms.manual.random_horizontal_flip_prob=0.5",
]

ADV_AUG_OVERRIDES = [
    "dataset.image_size=224",
    "dataset.transforms.use_timm=true",
    "dataset.transforms.timm.color_jitter=0.2",
    "dataset.transforms.timm.auto_augment=rand-m7-mstd0.5-inc1",
    "dataset.transforms.timm.train_interpolation=bicubic",
    "dataset.transforms.timm.random_erase.prob=0.1",
    "dataset.transforms.timm.random_erase.mode=pixel",
    "dataset.transforms.timm.random_erase.count=1",
]

EVAL_OVERRIDES = [
    "dataset.transforms.eval.resize_mode=resize",
    "dataset.transforms.eval.resize_size=224",
    "dataset.transforms.eval.crop_ratio=1.0",
    "dataset.transforms.eval.interpolation=bilinear",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna GridSampler search for CUB-200 Hier-COS.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Base training config.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Directory for trial artifacts.")
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME, help="Optuna study name.")
    parser.add_argument(
        "--storage",
        default=None,
        help=(
            "Optuna storage URL. Defaults to sqlite:///<output-root>/study.db. "
            "Use an absolute sqlite path as sqlite:////path/to/study.db."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials to run. Defaults to the full grid size.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default="independent",
        help="Checkpoint-selection mode whose validation best_metric is optimized.",
    )
    parser.add_argument(
        "--objective-mode",
        choices=OBJECTIVE_MODES,
        default="selection_score",
        help=(
            "selection_score maximizes train's validation best_metric. "
            "fpa_tice_pareto runs a multi-objective study: maximize validation FPA, minimize validation TICE."
        ),
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
        help="Print the planned grid and commands without creating an Optuna study or launching training.",
    )
    return parser.parse_args()


def effective_study_name(args: argparse.Namespace) -> str:
    if args.objective_mode == "fpa_tice_pareto" and args.study_name == DEFAULT_STUDY_NAME:
        return f"{DEFAULT_STUDY_NAME}_fpa_tice_pareto"
    return str(args.study_name)


def default_storage(output_root: Path) -> str:
    return f"sqlite:///{output_root / 'study.db'}"


def resolve_repo_path(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    return REPO_ROOT / raw


def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current: Dict[str, Any] = cfg
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            existing = {}
            current[part] = existing
        current = existing
    current[parts[-1]] = value


def load_frozen_config_data(config: str) -> Dict[str, Any]:
    config_path = resolve_repo_path(config)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config at {config_path}, got {type(data).__name__}.")
    for key, value in FROZEN_CONFIG_DEFAULTS.items():
        set_nested(data, key, value)
    return data


def frozen_config_path(output_root: Path, study_name: str) -> Path:
    return output_root / f"{study_name}_frozen_config.yaml"


def prepare_frozen_config(args: argparse.Namespace, output_root: Path, study_name: str) -> Path:
    path = frozen_config_path(output_root, study_name)
    if path.exists() and not bool(args.overwrite_frozen_config):
        print(f"[config] using existing frozen config: {path}")
        return path

    data = load_frozen_config_data(args.config)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    print(f"[config] wrote frozen config: {path}")
    return path


def search_space() -> Dict[str, List[Any]]:
    return {
        "model.alpha": ALPHAS,
        "model.transform_lr_scale": TRANSFORM_LR_SCALES,
        "augmentation": AUGMENTATIONS,
    }


def float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def trial_run_name(params: Mapping[str, Any]) -> str:
    alpha = float_tag(float(params["model.alpha"]))
    transform_lr_scale = float_tag(float(params["model.transform_lr_scale"]))
    augmentation = str(params["augmentation"])
    return f"alpha_{alpha}__tlr_{transform_lr_scale}__{augmentation}"


def augmentation_overrides(augmentation: str) -> List[str]:
    if augmentation == "base_aug":
        return list(BASE_AUG_OVERRIDES)
    if augmentation == "adv_aug":
        return list(ADV_AUG_OVERRIDES)
    raise ValueError(f"Unknown augmentation '{augmentation}'. Expected one of {AUGMENTATIONS}.")


def build_overrides(params: Mapping[str, Any], output_root: Path, extra_overrides: Sequence[str]) -> List[str]:
    run_dir = output_root / trial_run_name(params)
    overrides = [
        f"model.alpha={params['model.alpha']}",
        f"model.transform_lr_scale={params['model.transform_lr_scale']}",
        *augmentation_overrides(str(params["augmentation"])),
        *EVAL_OVERRIDES,
        f"train.output_dir={run_dir}",
        *extra_overrides,
    ]
    return overrides


def build_command(config: str, overrides: Sequence[str]) -> List[str]:
    return [sys.executable, "-m", "train.train", "--config", config, *overrides]


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


def print_dry_run(
    config: str,
    output_root: Path,
    study_name: str,
    extra_overrides: Sequence[str],
    n_trials: int,
) -> None:
    frozen_path = frozen_config_path(output_root, study_name)
    load_frozen_config_data(config)
    all_params = all_grid_params()
    params_list = all_params[: min(int(n_trials), len(all_params))]
    run_dirs = set()
    eval_override_sets = set()
    for index, params in enumerate(params_list, start=1):
        overrides = build_overrides(params, output_root, extra_overrides)
        run_dir = output_root / trial_run_name(params)
        run_dirs.add(str(run_dir))
        eval_override_sets.add(tuple(EVAL_OVERRIDES))
        command = build_command(str(frozen_path), overrides)
        print(f"[dry-run {index:02d}/{len(params_list):02d}] {trial_run_name(params)}")
        print("  run_dir:", run_dir)
        print("  params:", params)
        print("  command:", " ".join(command))
    print("")
    print(f"[dry-run] listed_combinations={len(params_list)} total_grid_combinations={len(all_params)}")
    print(f"[dry-run] unique_run_dirs={len(run_dirs)}")
    print(f"[dry-run] eval_override_variants={len(eval_override_sets)}")
    print(f"[dry-run] output_root={output_root}")
    print(f"[dry-run] frozen_config={frozen_path}")
    print("[dry-run] frozen_config_defaults:")
    for key, value in FROZEN_CONFIG_DEFAULTS.items():
        print(f"  {key}={value}")


def load_test_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "test_metrics.yaml"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected test metrics at {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {metrics_path}, got {type(data).__name__}.")
    return data


def selected_best_metric(test_metrics: Mapping[str, Any], selection_mode: str) -> float:
    selected = test_metrics.get(selection_mode)
    if not isinstance(selected, dict):
        raise KeyError(f"Missing '{selection_mode}' section in test_metrics.yaml.")
    if "best_metric" not in selected:
        raise KeyError(f"Missing '{selection_mode}.best_metric' in test_metrics.yaml.")
    return float(selected["best_metric"])


def selected_best_epoch(test_metrics: Mapping[str, Any], selection_mode: str) -> Any:
    selected = test_metrics.get(selection_mode)
    if isinstance(selected, dict):
        return selected.get("best_epoch")
    return None


def selected_final_metrics(test_metrics: Mapping[str, Any], selection_mode: str) -> Any:
    selected = test_metrics.get(selection_mode)
    if isinstance(selected, dict):
        return selected.get("test_metrics")
    return None


def load_val_metrics_at_epoch(run_dir: Path, epoch: Any) -> Dict[str, Any]:
    if epoch is None:
        raise ValueError("Cannot read validation metrics because best_epoch is missing.")
    target_epoch = int(epoch)
    run_log_path = run_dir / "run_log.jsonl"
    if not run_log_path.exists():
        raise FileNotFoundError(f"Expected run log at {run_log_path}")
    with run_log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != "epoch":
                continue
            if int(event.get("epoch", -1)) != target_epoch:
                continue
            val_metrics = event.get("val_metrics")
            if not isinstance(val_metrics, dict):
                raise ValueError(f"Missing val_metrics for epoch {target_epoch} in {run_log_path}")
            return val_metrics
    raise ValueError(f"Could not find epoch {target_epoch} in {run_log_path}")


def fpa_tice_objective_values(run_dir: Path, test_metrics: Mapping[str, Any], selection_mode: str) -> tuple[float, float]:
    best_epoch = selected_best_epoch(test_metrics, selection_mode)
    val_metrics = load_val_metrics_at_epoch(run_dir, best_epoch)
    fpa_key = f"fpa_{selection_mode}"
    tice_key = f"tice_{selection_mode}"
    if fpa_key not in val_metrics:
        raise KeyError(f"Missing validation metric '{fpa_key}' at best epoch {best_epoch}.")
    if tice_key not in val_metrics:
        raise KeyError(f"Missing validation metric '{tice_key}' at best epoch {best_epoch}.")
    return float(val_metrics[fpa_key]), float(val_metrics[tice_key])


def set_trial_attrs(
    trial: Any,
    *,
    run_dir: Path,
    overrides: Sequence[str],
    command: Sequence[str],
    test_metrics: Mapping[str, Any] | None = None,
    objective_values: Sequence[float] | None = None,
    returncode: int | None = None,
) -> None:
    trial.set_user_attr("run_dir", str(run_dir))
    trial.set_user_attr("overrides", list(overrides))
    trial.set_user_attr("command", list(command))
    if returncode is not None:
        trial.set_user_attr("returncode", int(returncode))
    if objective_values is not None:
        trial.set_user_attr("objective_values", list(objective_values))
        if len(objective_values) == 2:
            trial.set_user_attr("objective_val_fpa", float(objective_values[0]))
            trial.set_user_attr("objective_val_tice", float(objective_values[1]))
    if test_metrics is None:
        return
    for mode in SELECTION_MODES:
        if mode in test_metrics:
            trial.set_user_attr(f"{mode}_best_epoch", selected_best_epoch(test_metrics, mode))
            trial.set_user_attr(f"{mode}_best_metric", selected_best_metric(test_metrics, mode))
            trial.set_user_attr(f"{mode}_final_metrics", selected_final_metrics(test_metrics, mode))


def objective_factory(args: argparse.Namespace, output_root: Path, frozen_config: Path):
    def objective(trial: Any):
        params = {
            "model.alpha": trial.suggest_categorical("model.alpha", ALPHAS),
            "model.transform_lr_scale": trial.suggest_categorical(
                "model.transform_lr_scale",
                TRANSFORM_LR_SCALES,
            ),
            "augmentation": trial.suggest_categorical("augmentation", AUGMENTATIONS),
        }
        run_dir = output_root / trial_run_name(params)
        overrides = build_overrides(params, output_root, args.override)
        command = build_command(str(frozen_config), overrides)
        set_trial_attrs(trial, run_dir=run_dir, overrides=overrides, command=command)

        run_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        set_trial_attrs(
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

        test_metrics = load_test_metrics(run_dir)
        objective_values: Sequence[float]
        if args.objective_mode == "fpa_tice_pareto":
            objective_values = fpa_tice_objective_values(run_dir, test_metrics, args.selection_mode)
        else:
            objective_values = [selected_best_metric(test_metrics, args.selection_mode)]
        set_trial_attrs(
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
    frozen_config = prepare_frozen_config(args, output_root, study_name)
    storage = args.storage or default_storage(output_root)

    sampler = optuna.samplers.GridSampler(search_space(), seed=int(args.seed))
    pruner = optuna.pruners.NopPruner()
    study_kwargs: Dict[str, Any] = {
        "study_name": study_name,
        "storage": storage,
        "sampler": sampler,
        "pruner": pruner,
        "load_if_exists": True,
    }
    if args.objective_mode == "fpa_tice_pareto":
        study_kwargs["directions"] = ["maximize", "minimize"]
    else:
        study_kwargs["direction"] = "maximize"
    study = optuna.create_study(**study_kwargs)
    n_trials = resolve_n_trials(args.n_trials)
    study.optimize(objective_factory(args, output_root, frozen_config), n_trials=n_trials)
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
