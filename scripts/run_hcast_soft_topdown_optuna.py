#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

try:
    import optuna
except Exception as exc:  # pragma: no cover
    optuna = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _OPTUNA_IMPORT_ERROR = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna tuner for H-CAST soft-topdown (grid/TPE with optional halving-style pruning)."
    )
    parser.add_argument("--config", type=str, default="configs/hcast_soft_topdown_cifar100.yaml")
    parser.add_argument("--output-root", type=str, default="/scratch/g.saggini1/outputs")
    parser.add_argument("--study-tag", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--study-name", type=str, default="")
    parser.add_argument("--storage-url", type=str, default="")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--sampler", type=str, choices=["grid", "tpe"], default="grid")
    parser.add_argument("--pruner", type=str, choices=["none", "sha", "hyperband"], default="sha")
    parser.add_argument("--n-trials", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-if-exists", action="store_true")
    parser.add_argument("--no-load-if-exists", dest="load_if_exists", action="store_false")
    parser.add_argument("--n-startup-trials", type=int, default=8)

    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--gate-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=[0.3, 2.0])
    parser.add_argument("--kl-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=[0.1, 1.0])
    parser.add_argument("--tau-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=[0.0, 0.4])
    parser.add_argument(
        "--gate-values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit gate_strength values for grid search. Overrides linspace sampling for this parameter.",
    )
    parser.add_argument(
        "--kl-values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit kl_weight values for grid search. Overrides linspace sampling for this parameter.",
    )
    parser.add_argument(
        "--tau-values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit tau values for grid search. Overrides linspace sampling for this parameter.",
    )
    parser.add_argument("--include-config-values", action="store_true")
    parser.add_argument("--no-include-config-values", dest="include_config_values", action="store_false")

    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.set_defaults(load_if_exists=True, include_config_values=True)
    return parser.parse_args()


def _as_float_dict(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for key, value in metrics.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    out = _as_float_dict(metrics)
    if "weighted_ap_topdown" not in out:
        if "weighted_ap" in out:
            out["weighted_ap_topdown"] = out["weighted_ap"]
        elif "weighted_ap_independent" in out:
            out["weighted_ap_topdown"] = out["weighted_ap_independent"]
    if "weighted_ap_independent" not in out and "weighted_ap_topdown" in out:
        out["weighted_ap_independent"] = out["weighted_ap_topdown"]

    if "fpa_topdown" not in out:
        if "fpa" in out:
            out["fpa_topdown"] = out["fpa"]
        elif "fpa_independent" in out:
            out["fpa_topdown"] = out["fpa_independent"]
    if "fpa_independent" not in out and "fpa_topdown" in out:
        out["fpa_independent"] = out["fpa_topdown"]

    if "tice_topdown" not in out:
        if "tice" in out:
            out["tice_topdown"] = out["tice"]
        elif "tice_independent" in out:
            out["tice_topdown"] = out["tice_independent"]
    if "tice_independent" not in out and "tice_topdown" in out:
        out["tice_independent"] = out["tice_topdown"]
    return out


def _metric_for_best(eval_metrics: Dict[str, float]) -> float:
    has_fpa = "fpa_topdown" in eval_metrics
    has_tice = "tice_topdown" in eval_metrics
    has_wap = "weighted_ap_topdown" in eval_metrics
    if has_fpa or has_tice or has_wap:
        fpa = float(eval_metrics.get("fpa_topdown", 0.0))
        neg_tice = -float(eval_metrics.get("tice_topdown", 1.0))
        wap = float(eval_metrics.get("weighted_ap_topdown", 0.0))
        return float(fpa + 1e-3 * neg_tice + 1e-6 * wap)

    deepest_topdown = [
        k for k in eval_metrics if k.startswith("acc_level_topdown_") and k[len("acc_level_topdown_") :].isdigit()
    ]
    deepest_independent = [
        k
        for k in eval_metrics
        if k.startswith("acc_level_independent_") and k[len("acc_level_independent_") :].isdigit()
    ]
    deepest = deepest_topdown or deepest_independent
    if not deepest:
        return float(eval_metrics.get("fpa_topdown", 0.0))
    deepest_key = max(deepest, key=lambda key: int(key.rsplit("_", 1)[-1]))
    primary = float(eval_metrics.get(deepest_key, 0.0))
    tie = float(eval_metrics.get("fpa_topdown", 0.0))
    return float(primary + 1e-3 * tie)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _best_score_up_to_epoch(run_dir: Path, max_epoch: int) -> Dict[str, Any]:
    events = [e for e in _load_jsonl(run_dir / "run_log.jsonl") if str(e.get("event", "")) == "epoch"]
    events = [e for e in events if int(e.get("epoch", -1)) <= int(max_epoch)]
    if not events:
        return {
            "best_val_fpa_topdown": float("nan"),
            "best_rank_score": float("nan"),
            "best_epoch": None,
        }

    parsed = []
    max_fpa = -math.inf
    for event in events:
        val_norm = _normalize_metrics(event.get("val_metrics", {}))
        rank_score = _metric_for_best(val_norm)
        fpa = float(val_norm.get("fpa_topdown", float("nan")))
        parsed.append((event, fpa, rank_score))
        if math.isfinite(fpa):
            max_fpa = max(max_fpa, fpa)

    if not math.isfinite(max_fpa):
        best = max(parsed, key=lambda x: x[2])
    else:
        candidates = [x for x in parsed if math.isfinite(x[1]) and abs(x[1] - max_fpa) <= 1e-12]
        best = max(candidates, key=lambda x: x[2]) if candidates else max(parsed, key=lambda x: x[2])

    return {
        "best_val_fpa_topdown": float(best[1]) if math.isfinite(best[1]) else float("nan"),
        "best_rank_score": float(best[2]),
        "best_epoch": int(best[0].get("epoch")) if best[0].get("epoch") is not None else None,
    }


def _linspace_values(lo: float, hi: float, n: int) -> List[float]:
    if n < 1:
        raise ValueError(f"samples must be >= 1, got {n}")
    if lo > hi:
        raise ValueError(f"Invalid range [{lo}, {hi}]")
    if n == 1:
        return [round((lo + hi) / 2.0, 6)]
    step = (hi - lo) / float(n - 1)
    values = [lo + i * step for i in range(n)]
    values[0] = lo
    values[-1] = hi
    rounded = [round(float(v), 6) for v in values]
    if len(set(rounded)) != len(rounded):
        raise RuntimeError(f"Duplicate values after rounding: {rounded}")
    return rounded


def _sorted_unique_values(values: Sequence[float], name: str) -> List[float]:
    seen = set()
    out: List[float] = []
    for raw in values:
        try:
            value = round(float(raw), 6)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains a non-numeric value: {raw!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite value: {raw!r}")
        if value not in seen:
            seen.add(value)
            out.append(value)
    if not out:
        raise ValueError(f"{name} must contain at least one finite numeric value.")
    return sorted(out)


def _try_get_float(d: Dict[str, Any], key: str) -> Optional[float]:
    if key not in d:
        return None
    try:
        value = float(d.get(key))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _resource_schedule(min_epochs: int, max_epochs: int, factor: int) -> List[int]:
    if min_epochs < 1:
        raise ValueError("min_epochs must be >= 1")
    if max_epochs < min_epochs:
        raise ValueError(f"max_epochs ({max_epochs}) must be >= min_epochs ({min_epochs})")
    if factor <= 1:
        raise ValueError("factor must be > 1")

    resources = [int(min_epochs)]
    while resources[-1] < max_epochs:
        nxt = int(resources[-1] * factor)
        if nxt <= resources[-1]:
            nxt = resources[-1] + 1
        if nxt >= max_epochs:
            nxt = max_epochs
        if nxt == resources[-1]:
            break
        resources.append(nxt)
    if resources[-1] != max_epochs:
        resources.append(max_epochs)
    return sorted(set(resources))


def _sort_key(row: Dict[str, Any]) -> Tuple[float, float]:
    fpa = row.get("best_val_fpa_topdown", float("nan"))
    rank = row.get("best_rank_score", float("nan"))
    fpa_sort = float(fpa) if isinstance(fpa, float) and math.isfinite(fpa) else -math.inf
    rank_sort = float(rank) if isinstance(rank, float) and math.isfinite(rank) else -math.inf
    return (fpa_sort, rank_sort)


def _build_train_cmd(
    python_bin: str,
    base_config: str,
    run_dir: Path,
    target_epochs: int,
    gate_strength: float,
    kl_weight: float,
    tau: float,
) -> List[str]:
    cmd = [
        python_bin,
        "-m",
        "train.train",
        "--config",
        base_config,
        f"train.output_dir={run_dir}",
        f"train.epochs={int(target_epochs)}",
        "model.loss.globalkl=false",
        "model.soft_topdown.enabled=true",
        "model.soft_topdown.temperature=1.0",
        "model.soft_topdown.detach_upper_probs=false",
        "model.soft_topdown.eps=1.0e-12",
        f"model.soft_topdown.gate_strength={gate_strength}",
        f"model.soft_topdown.kl_weight={kl_weight}",
        f"model.soft_topdown.tau={tau}",
    ]
    resume_ckpt = run_dir / "latest.pt"
    if resume_ckpt.exists():
        cmd.append(f"train.resume={resume_ckpt}")
    return cmd


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if args.factor <= 1:
        raise SystemExit("--factor must be > 1")
    if args.min_epochs < 1:
        raise SystemExit("--min-epochs must be >= 1")

    with config_path.open("r", encoding="utf-8") as f:
        cfg_blob = yaml.safe_load(f) or {}
    cfg_epochs = int(((cfg_blob.get("train") or {}).get("epochs", 100)))
    max_epochs = int(args.max_epochs) if int(args.max_epochs) > 0 else cfg_epochs

    soft_topdown_cfg = ((cfg_blob.get("model") or {}).get("soft_topdown") or {})
    config_defaults = {
        "gate_strength": _try_get_float(soft_topdown_cfg, "gate_strength"),
        "kl_weight": _try_get_float(soft_topdown_cfg, "kl_weight"),
        "tau": _try_get_float(soft_topdown_cfg, "tau"),
    }

    gate_values: List[float]
    kl_values: List[float]
    tau_values: List[float]
    if args.gate_values is not None:
        gate_values = _sorted_unique_values(args.gate_values, "--gate-values")
    else:
        gate_values = _linspace_values(float(args.gate_range[0]), float(args.gate_range[1]), int(args.samples))
    if args.kl_values is not None:
        kl_values = _sorted_unique_values(args.kl_values, "--kl-values")
    else:
        kl_values = _linspace_values(float(args.kl_range[0]), float(args.kl_range[1]), int(args.samples))
    if args.tau_values is not None:
        tau_values = _sorted_unique_values(args.tau_values, "--tau-values")
    else:
        tau_values = _linspace_values(float(args.tau_range[0]), float(args.tau_range[1]), int(args.samples))

    if bool(args.include_config_values):
        if config_defaults["gate_strength"] is not None:
            gate_values = _sorted_unique_values([*gate_values, config_defaults["gate_strength"]], "gate_strength values")
        if config_defaults["kl_weight"] is not None:
            kl_values = _sorted_unique_values([*kl_values, config_defaults["kl_weight"]], "kl_weight values")
        if config_defaults["tau"] is not None:
            tau_values = _sorted_unique_values([*tau_values, config_defaults["tau"]], "tau values")

    sampled_values = {
        "gate_strength": gate_values,
        "kl_weight": kl_values,
        "tau": tau_values,
    }
    grid_combinations = list(
        itertools.product(sampled_values["gate_strength"], sampled_values["kl_weight"], sampled_values["tau"])
    )
    resource_schedule = _resource_schedule(int(args.min_epochs), int(max_epochs), int(args.factor))

    output_root = Path(args.output_root)
    artifact_dir = output_root / f"hcast_softtd_optuna_{args.study_tag}"
    trials_root = artifact_dir / "trials"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trials_root.mkdir(parents=True, exist_ok=True)

    space_yaml = artifact_dir / "optuna_search_space.yaml"
    trials_csv = artifact_dir / "optuna_trials.csv"
    results_yaml = artifact_dir / "optuna_results.yaml"

    study_name = args.study_name.strip() or f"hcast_softtd_optuna_{args.study_tag}"
    storage_url = args.storage_url.strip() or f"sqlite:///{artifact_dir / 'optuna_study.db'}"
    timeout_sec = int(args.timeout_sec) if int(args.timeout_sec) > 0 else None

    n_trials: Optional[int]
    if int(args.n_trials) > 0:
        n_trials = int(args.n_trials)
    else:
        n_trials = len(grid_combinations)
    if args.sampler == "grid":
        n_trials = min(n_trials, len(grid_combinations))

    space_payload = {
        "search_type": "optuna",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_name": study_name,
        "storage_url": storage_url,
        "base_config": str(config_path),
        "artifact_dir": str(artifact_dir),
        "sampler": args.sampler,
        "pruner": args.pruner,
        "seed": int(args.seed),
        "n_trials": int(n_trials),
        "timeout_sec": timeout_sec,
        "ranges": {
            "gate_strength": [float(args.gate_range[0]), float(args.gate_range[1])],
            "kl_weight": [float(args.kl_range[0]), float(args.kl_range[1])],
            "tau": [float(args.tau_range[0]), float(args.tau_range[1])],
        },
        "explicit_values_if_any": {
            "gate_strength": _sorted_unique_values(args.gate_values, "--gate-values")
            if args.gate_values is not None
            else None,
            "kl_weight": _sorted_unique_values(args.kl_values, "--kl-values") if args.kl_values is not None else None,
            "tau": _sorted_unique_values(args.tau_values, "--tau-values") if args.tau_values is not None else None,
        },
        "include_config_values": bool(args.include_config_values),
        "config_soft_topdown_defaults": config_defaults,
        "samples_per_param_for_grid": int(args.samples),
        "sampled_values_for_grid": sampled_values,
        "grid_num_combinations": len(grid_combinations),
        "resource_schedule_epochs": resource_schedule,
        "factor": int(args.factor),
        "min_epochs": int(args.min_epochs),
        "max_epochs": int(max_epochs),
    }
    with space_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(space_payload, f, sort_keys=False, allow_unicode=False)

    print(f"[optuna] study_name={study_name}")
    print(f"[optuna] storage={storage_url}")
    print(f"[optuna] sampler={args.sampler} pruner={args.pruner}")
    print(f"[optuna] n_trials={n_trials} timeout_sec={timeout_sec}")
    print(f"[optuna] resource schedule (epochs)={resource_schedule}")
    if args.sampler != "grid" and any(v is not None for v in [args.gate_values, args.kl_values, args.tau_values]):
        print("[optuna][warning] Explicit --*-values are ignored when sampler=tpe (TPE uses continuous ranges).")
    if args.sampler == "grid" and args.pruner != "none":
        print(
            "[optuna][warning] Grid + pruning can drop late-improving configs before max_epochs. "
            "Use --pruner none for exhaustive grid evaluation."
        )
    print("[optuna] sampled values (grid baseline):")
    for key in ["gate_strength", "kl_weight", "tau"]:
        print(f"  - {key}: {sampled_values[key]}")
    print(f"[optuna] wrote: {space_yaml}")
    print(f"[optuna] artifact dir: {artifact_dir}")

    if args.dry_run:
        print("[optuna] dry-run enabled: no trials executed.")
        return

    if optuna is None:
        raise SystemExit(f"Optuna is not installed. Install dependencies first. Import error: {_OPTUNA_IMPORT_ERROR}")

    if args.sampler == "grid":
        sampler = optuna.samplers.GridSampler(
            search_space={
                "gate_strength": sampled_values["gate_strength"],
                "kl_weight": sampled_values["kl_weight"],
                "tau": sampled_values["tau"],
            }
        )
    else:
        sampler = optuna.samplers.TPESampler(seed=int(args.seed), n_startup_trials=int(args.n_startup_trials))

    if args.pruner == "none":
        pruner = optuna.pruners.NopPruner()
    elif args.pruner == "sha":
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=int(args.min_epochs),
            reduction_factor=int(args.factor),
            min_early_stopping_rate=0,
        )
    else:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=int(args.min_epochs),
            max_resource=int(max_epochs),
            reduction_factor=int(args.factor),
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=bool(args.load_if_exists),
    )

    root_dir = Path.cwd()
    python_bin = args.python
    base_config = str(config_path)

    def objective(trial: "optuna.Trial") -> float:
        if args.sampler == "grid":
            gate_strength = float(trial.suggest_categorical("gate_strength", sampled_values["gate_strength"]))
            kl_weight = float(trial.suggest_categorical("kl_weight", sampled_values["kl_weight"]))
            tau = float(trial.suggest_categorical("tau", sampled_values["tau"]))
        else:
            gate_strength = float(
                trial.suggest_float("gate_strength", float(args.gate_range[0]), float(args.gate_range[1]))
            )
            kl_weight = float(trial.suggest_float("kl_weight", float(args.kl_range[0]), float(args.kl_range[1])))
            tau = float(trial.suggest_float("tau", float(args.tau_range[0]), float(args.tau_range[1])))

        run_dir = trials_root / f"trial_{trial.number:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        trial.set_user_attr("run_dir", str(run_dir))

        stage_history: List[Dict[str, Any]] = []
        best_value = -math.inf

        for target_epochs in resource_schedule:
            cmd = _build_train_cmd(
                python_bin=python_bin,
                base_config=base_config,
                run_dir=run_dir,
                target_epochs=int(target_epochs),
                gate_strength=gate_strength,
                kl_weight=kl_weight,
                tau=tau,
            )
            stage_log = run_dir / f"stage_{int(target_epochs):04d}.log"
            cmd_text = shlex.join(cmd)

            try:
                env = os.environ.copy()
                env.setdefault("PYTHONUNBUFFERED", "1")
                with stage_log.open("a", encoding="utf-8") as f:
                    f.write(
                        f"\n=== [{datetime.now(timezone.utc).isoformat()}] "
                        f"stage_start epoch={int(target_epochs)} ===\n"
                    )
                    f.write(f"command: {cmd_text}\n")
                    f.flush()
                    subprocess.run(
                        cmd,
                        cwd=str(root_dir),
                        check=True,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                    )
                    f.write(
                        f"\n=== [{datetime.now(timezone.utc).isoformat()}] "
                        f"stage_end epoch={int(target_epochs)} status=ok ===\n"
                    )
            except subprocess.CalledProcessError as exc:
                with stage_log.open("a", encoding="utf-8") as f:
                    f.write(
                        f"\n=== [{datetime.now(timezone.utc).isoformat()}] "
                        f"stage_end epoch={int(target_epochs)} "
                        f"status=fail returncode={int(exc.returncode)} ===\n"
                    )
                trial.set_user_attr("failed_stage_epoch", int(target_epochs))
                trial.set_user_attr("failed_returncode", int(exc.returncode))
                trial.set_user_attr("failed_stage_log", str(stage_log))
                trial.set_user_attr("failed_command", cmd_text)
                raise

            score_blob = _best_score_up_to_epoch(run_dir, max_epoch=int(target_epochs))
            value = score_blob["best_val_fpa_topdown"]
            report_value = float(value) if isinstance(value, float) and math.isfinite(value) else -1.0
            trial.report(report_value, step=int(target_epochs))

            stage_record = {
                "target_epochs": int(target_epochs),
                "best_val_fpa_topdown": float(score_blob["best_val_fpa_topdown"])
                if isinstance(score_blob["best_val_fpa_topdown"], float)
                and math.isfinite(score_blob["best_val_fpa_topdown"])
                else float("nan"),
                "best_rank_score": float(score_blob["best_rank_score"])
                if isinstance(score_blob["best_rank_score"], float) and math.isfinite(score_blob["best_rank_score"])
                else float("nan"),
                "best_epoch": score_blob["best_epoch"],
            }
            stage_history.append(stage_record)

            if math.isfinite(report_value):
                best_value = max(best_value, report_value)
            if trial.should_prune():
                trial.set_user_attr("stage_history", stage_history)
                trial.set_user_attr("best_val_fpa_topdown", float(best_value) if math.isfinite(best_value) else float("nan"))
                raise optuna.TrialPruned(f"Pruned at epoch={target_epochs}")

        final_blob = _best_score_up_to_epoch(run_dir, max_epoch=int(max_epochs))
        trial.set_user_attr("stage_history", stage_history)
        trial.set_user_attr("best_val_fpa_topdown", float(final_blob["best_val_fpa_topdown"]))
        trial.set_user_attr("best_rank_score", float(final_blob["best_rank_score"]))
        trial.set_user_attr("best_epoch", final_blob["best_epoch"])

        if math.isfinite(final_blob["best_val_fpa_topdown"]):
            return float(final_blob["best_val_fpa_topdown"])
        return float(best_value) if math.isfinite(best_value) else -1.0

    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)

    rows: List[Dict[str, Any]] = []
    for trial in study.trials:
        attrs = trial.user_attrs or {}
        value = trial.value if trial.value is not None else float("nan")
        row = {
            "trial_number": int(trial.number),
            "state": str(trial.state.name),
            "objective_value": float(value) if isinstance(value, float) and math.isfinite(value) else float("nan"),
            "best_val_fpa_topdown": float(attrs.get("best_val_fpa_topdown", float("nan"))),
            "best_rank_score": float(attrs.get("best_rank_score", float("nan"))),
            "best_epoch": attrs.get("best_epoch"),
            "gate_strength": float(trial.params.get("gate_strength", float("nan"))),
            "kl_weight": float(trial.params.get("kl_weight", float("nan"))),
            "tau": float(trial.params.get("tau", float("nan"))),
            "run_dir": str(attrs.get("run_dir", "")),
        }
        rows.append(row)

    with trials_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_number",
                "state",
                "objective_value",
                "best_val_fpa_topdown",
                "best_rank_score",
                "best_epoch",
                "gate_strength",
                "kl_weight",
                "tau",
                "run_dir",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    completed_rows = [row for row in rows if row["state"] == "COMPLETE"]
    ranked_rows = sorted(completed_rows, key=_sort_key, reverse=True)
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = int(rank)

    best_row = ranked_rows[0] if ranked_rows else None
    top_k = [
        {
            "rank": row["rank"],
            "trial_number": row["trial_number"],
            "best_val_fpa_topdown": row["best_val_fpa_topdown"],
            "best_epoch": row["best_epoch"],
            "gate_strength": row["gate_strength"],
            "kl_weight": row["kl_weight"],
            "tau": row["tau"],
            "run_dir": row["run_dir"],
        }
        for row in ranked_rows[:10]
    ]

    results_payload = {
        "search_type": "optuna",
        "study_name": study.study_name,
        "storage_url": storage_url,
        "sampler": args.sampler,
        "pruner": args.pruner,
        "n_trials_requested": n_trials,
        "n_trials_total": len(rows),
        "n_trials_complete": sum(1 for row in rows if row["state"] == "COMPLETE"),
        "n_trials_pruned": sum(1 for row in rows if row["state"] == "PRUNED"),
        "n_trials_failed": sum(1 for row in rows if row["state"] == "FAIL"),
        "resource_schedule_epochs": resource_schedule,
        "ranking_metric": "best_val_fpa_topdown",
        "tie_breaker": "metric_for_best-compatible rank score",
        "best_run_path": best_row["run_dir"] if best_row else None,
        "best_trial": best_row,
        "top_k_preview": top_k,
    }
    with results_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(results_payload, f, sort_keys=False, allow_unicode=False)

    print(f"[optuna] wrote: {trials_csv}")
    print(f"[optuna] wrote: {results_yaml}")
    if best_row is not None:
        print(
            "[optuna] best trial: "
            f"#{best_row['trial_number']} fpa_td={best_row['best_val_fpa_topdown']:.6f} "
            f"epoch={best_row['best_epoch']} path={best_row['run_dir']}"
        )


if __name__ == "__main__":
    main()
