"""Evaluate checkpoints with native and/or post-hoc inference rules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
import yaml

from datasets import build_dataloader
from models import build_model
from train.config_loader import load_resolved_run_config
from train.evaluation import evaluate_batch
from train.metric_formatting import pretty_metrics
from train.metrics import merge_metric_batches
from train.runtime.finetune import load_trusted_checkpoint
from train.runtime.optimization import seed_everything
from train.runtime.common import section_to_dict
from train.runtime.selection import BEST_SELECTION_MODES

from .posthoc_inference import (
    HCC_PREFIX,
    INFERENCE_RULES,
    NODE_SCORE,
    SUBSPACE_NORM,
    PosthocInferenceRule,
)


# Superseded flat mode names, kept so existing commands and scripts keep working.
# `normal` and `hcc` named the checkpoint's own readout, which differs by model,
# so both resolve through the model family.
LEGACY_MODE_ALIASES = {
    "normal": {"hiercos": SUBSPACE_NORM, None: NODE_SCORE},
    "hiercos": {None: SUBSPACE_NORM},
    "node_softmax": {None: NODE_SCORE},
    "hcc": {
        "hiercos": f"{HCC_PREFIX}{SUBSPACE_NORM}",
        None: f"{HCC_PREFIX}{NODE_SCORE}",
    },
}
INFERENCE_MODES = (*INFERENCE_RULES, "all", "both", *LEGACY_MODE_ALIASES)
CHECKPOINT_MODES = (*BEST_SELECTION_MODES, "both")
OUTCOME_PREFIXES = (
    "acc_level_independent_",
    "acc_level_topdown_",
    "weighted_ap_",
    "fpa_",
    "ahd_",
    "tice_",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run test-only evaluation from existing best checkpoints. Inference "
            "rules are one grid of readout (node_score, subspace_norm) times "
            "transform (none, hcc). `all` evaluates all four; `both` evaluates "
            "the two untransformed readouts. The checkpoint's own inference is "
            "one of the four cells and is used as the paired reference."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing config_resolved.yaml and best checkpoints.",
    )
    parser.add_argument(
        "--inference-mode",
        choices=INFERENCE_MODES,
        default="both",
        help="Inference rule to evaluate (default: both).",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=CHECKPOINT_MODES,
        default="both",
        help="Validation-selected checkpoint(s) to evaluate (default: both).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config path; defaults to RUN_DIR/config_resolved.yaml.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override such as cuda, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output YAML; defaults to RUN_DIR/posthoc_inference_test_metrics.yaml.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output YAML.",
    )
    return parser.parse_args()


def _selected_values(choice: str, both_values: Sequence[str]) -> Sequence[str]:
    return tuple(both_values) if choice == "both" else (choice,)


def native_inference_rule(model_name: str, hcc_trained: bool) -> str:
    """Return the grid cell that reproduces this checkpoint's own inference.

    Hier-COS ends its forward pass in taxonomy-subspace norms; classifier-head
    models rank each class by its own score. A run trained with HCC applies the
    projection inside its own forward pass, so its native cell is the
    `hcc_`-prefixed one.
    """
    readout = SUBSPACE_NORM if str(model_name) == "hiercos" else NODE_SCORE
    return f"{HCC_PREFIX}{readout}" if hcc_trained else readout


def canonical_inference_rule(requested: str, model_name: str) -> str:
    """Map one requested mode, legacy or canonical, to a grid cell name."""
    alias = LEGACY_MODE_ALIASES.get(requested)
    if alias is None:
        return requested
    return alias.get(str(model_name), alias[None])


def legacy_mode_name(
    rule: str,
    model_name: str,
    hcc_trained: bool = False,
) -> Optional[str]:
    """Return the row name a pre-rename YAML used for this cell, if any.

    The inverse of `canonical_inference_rule` is ambiguous — `normal` and
    `node_softmax` both resolve to `node_score` — so the direction that matters
    for reading old results is spelled out here. `normal` always named whatever
    the checkpoint did natively, which is why it follows the native cell.
    """
    if rule == native_inference_rule(model_name, hcc_trained):
        return "normal"
    if str(model_name) == "hiercos":
        return {
            NODE_SCORE: "node_softmax",
            f"{HCC_PREFIX}{SUBSPACE_NORM}": "hcc",
        }.get(rule)
    return {
        SUBSPACE_NORM: "hiercos",
        f"{HCC_PREFIX}{NODE_SCORE}": "hcc",
    }.get(rule)


def _resolve_inference_modes(
    requested: str,
    model_name: str,
    hcc_trained: bool = False,
) -> Sequence[str]:
    """Resolve a requested mode into grid cells, native cell first.

    Every cell is defined for every model, so no combination is rejected. The
    native cell leads so that it prints first and anchors the paired deltas.
    """
    native = native_inference_rule(model_name, hcc_trained)
    if requested in {"both", "all"}:
        candidates = (
            INFERENCE_RULES
            if requested == "all"
            else (NODE_SCORE, SUBSPACE_NORM)
        )
        ordered = [rule for rule in candidates if rule == native]
        ordered.extend(rule for rule in candidates if rule != native)
        return tuple(ordered)
    return (canonical_inference_rule(requested, model_name),)


def _run_trained_with_hcc(cfg: Any) -> bool:
    """Report whether the saved run enabled HCC, without validating the config.

    This CLI treats `config_resolved.yaml` as an immutable artifact and does not
    apply the current policy validator, so the flag is read directly instead of
    through `build_hcc_config`. Both the canonical top-level `hcc` section and a
    legacy `model.hcc` section count, so an unexpected layout cannot silently
    report an HCC run as HCC-free.
    """
    candidates = [
        getattr(cfg, "hcc", None),
        section_to_dict(getattr(cfg, "model", None)).get("hcc"),
    ]
    return any(
        bool(section_to_dict(candidate).get("enabled", False))
        for candidate in candidates
    )


def _set_model_epoch(model: torch.nn.Module, epoch: int) -> None:
    if hasattr(model, "set_epoch"):
        model.set_epoch(epoch)
        return
    wrapped_model = getattr(model, "module", None)
    if wrapped_model is not None and hasattr(wrapped_model, "set_epoch"):
        wrapped_model.set_epoch(epoch)


def _outcome_metrics(metrics: Mapping[str, float]) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if key.startswith(OUTCOME_PREFIXES)
    }


@torch.no_grad()
def _evaluate_inference_modes(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    cfg: Any,
    taxonomy: Mapping[str, Any],
    inference_modes: Sequence[str],
    inference_rules: Mapping[str, Any],
) -> tuple:
    """Score every requested grid cell from one shared forward pass.

    Also returns the model's own HCC diagnostics from the first batch. For a run
    trained with HCC these record the alpha its forward pass actually applied,
    which the `hcc_` cells do not replicate: they always project at `alpha=1`.
    """
    model.eval()
    metric_batches: Dict[str, list] = {mode: [] for mode in inference_modes}
    batch_weights = []
    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"
    taxonomy_dict = dict(taxonomy)
    model_hcc_diagnostics: Dict[str, float] = {}

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)

        if not model_hcc_diagnostics:
            diagnostics = output.get("hcc_diagnostics")
            if isinstance(diagnostics, Mapping):
                model_hcc_diagnostics = {
                    key: float(value)
                    for key, value in diagnostics.items()
                    if isinstance(value, (int, float))
                }

        for inference_mode in inference_modes:
            rule = inference_rules.get(inference_mode)
            if rule is None:
                raise RuntimeError(
                    f"Inference rule `{inference_mode}` was not initialized."
                )
            metric_batches[inference_mode].append(
                # `_outcome_metrics` keeps only the headline metrics, so the
                # level-3 diagnostics would be computed and then dropped.
                evaluate_batch(
                    rule.transform_output(output),
                    labels,
                    taxonomy=taxonomy_dict,
                    include_diagnostics=False,
                )
            )
        batch_weights.append(int(labels.size(0)))

    metrics_by_inference = {
        mode: _outcome_metrics(
            merge_metric_batches(batches, batch_weights=batch_weights)
        )
        for mode, batches in metric_batches.items()
    }
    return metrics_by_inference, model_hcc_diagnostics


def _metric_deltas(
    baseline: Mapping[str, float],
    alternative: Mapping[str, float],
) -> Dict[str, float]:
    return {
        key: float(alternative[key]) - float(baseline[key])
        for key in sorted(set(baseline) & set(alternative))
    }


def _save_yaml(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False, allow_unicode=False)


def _prepare_test_config(cfg: Any) -> list[Dict[str, Any]]:
    """Apply explicit, inference-only compatibility adjustments in memory."""
    adjustments: list[Dict[str, Any]] = []

    # Some resolved legacy configs retain placeholder JSON paths even though
    # the run used the dataset adapter's official-data fallback. For the test
    # split only, restore that fallback when the configured artifact is absent.
    annotations = cfg.dataset.get("annotations")
    configured_test = annotations.get("test") if annotations is not None else None
    if configured_test:
        annotation_path = Path(str(configured_test)).expanduser()
        if not annotation_path.is_absolute():
            annotation_path = Path(str(cfg.dataset.root)).expanduser() / annotation_path
        if not annotation_path.is_file():
            del annotations["test"]
            adjustments.append(
                {
                    "field": "dataset.annotations.test",
                    "saved_value": str(configured_test),
                    "evaluation_value": None,
                    "reason": "configured annotation is absent; use official dataset adapter",
                }
            )

    # Test evaluation should cover the complete split for every model. This is
    # especially important when comparing old Hier-COS/HRN configs whose
    # training-era loader setting dropped the final partial evaluation batch.
    if bool(cfg.dataloader.get("drop_last_eval", False)):
        cfg.dataloader.drop_last_eval = False
        adjustments.append(
            {
                "field": "dataloader.drop_last_eval",
                "saved_value": True,
                "evaluation_value": False,
                "reason": "evaluate the complete test split",
            }
        )

    return adjustments


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else run_dir / "config_resolved.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Resolved config does not exist: {config_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_dir / "posthoc_inference_test_metrics.yaml"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    cfg, _ = load_resolved_run_config(str(config_path))
    model_name = str(cfg.model.name)
    # A run trained with HCC applies the projection inside its own forward pass
    # at final test, so its native readout is the `hcc_`-prefixed grid cell.
    hcc_trained = _run_trained_with_hcc(cfg)
    inference_modes = _resolve_inference_modes(
        args.inference_mode,
        model_name,
        hcc_trained,
    )
    native_mode = native_inference_rule(model_name, hcc_trained)
    reference_mode = native_mode if native_mode in inference_modes else inference_modes[0]
    checkpoint_modes = _selected_values(args.checkpoint_mode, BEST_SELECTION_MODES)

    checkpoint_paths = {
        mode: run_dir / f"best_{mode}.pt"
        for mode in checkpoint_modes
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))

    seed_everything(
        int(cfg.train.seed),
        bool(cfg.runtime.get("deterministic", True)),
    )
    device = torch.device(
        args.device
        if args.device is not None
        else str(cfg.train.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    )

    # Full checkpoints replace every learned parameter. Avoid downloading or
    # initializing external pretrained weights while reconstructing the model.
    if "pretrained" in cfg.model:
        cfg.model.pretrained = False

    evaluation_config_adjustments = _prepare_test_config(cfg)
    annotation_fallback = any(
        adjustment["field"] == "dataset.annotations.test"
        for adjustment in evaluation_config_adjustments
    )
    if annotation_fallback:
        # The fallback can rebuild a different label space than the run trained
        # on, which shows up as plausible-looking but meaningless parent-level
        # metrics rather than as an error.
        print(
            "[warning] the configured test manifest is missing, so this "
            "evaluation falls back to the official dataset adapter. Its label "
            "space may differ from the one this run trained on; compare the "
            "native row against the run's own test_metrics.yaml before using "
            "these numbers.",
            file=sys.stderr,
        )
    test_loader, num_classes_per_level, taxonomy = build_dataloader(cfg, "test")
    if not isinstance(taxonomy, Mapping):
        raise ValueError("Post-hoc evaluation requires dataset taxonomy mappings.")

    model = build_model(cfg, num_classes_per_level, taxonomy).to(device)
    inference_rules: Dict[str, PosthocInferenceRule] = {
        mode: PosthocInferenceRule(
            rule=mode,
            num_classes_per_level=num_classes_per_level,
            taxonomy=taxonomy,
            model_name=model_name,
        )
        for mode in inference_modes
    }
    level_names = [str(name) for name in cfg.dataset.get("levels", [])]

    results: Dict[str, Any] = {}
    for checkpoint_mode, checkpoint_path in checkpoint_paths.items():
        checkpoint = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        checkpoint_epoch = int(checkpoint.get("epoch", int(cfg.train.epochs) - 1))
        _set_model_epoch(model, checkpoint_epoch)

        print(
            f"[checkpoint:{checkpoint_mode}] {checkpoint_path} "
            f"(epoch {checkpoint_epoch + 1})"
        )
        metrics_by_inference, model_hcc_diagnostics = _evaluate_inference_modes(
            model=model,
            loader=test_loader,
            device=device,
            cfg=cfg,
            taxonomy=taxonomy,
            inference_modes=inference_modes,
            inference_rules=inference_rules,
        )
        for inference_mode, metrics in metrics_by_inference.items():
            native_tag = " (native)" if inference_mode == native_mode else ""
            print(
                f"[test:{checkpoint_mode}:{inference_mode}]{native_tag} "
                f"{pretty_metrics(metrics, level_names=level_names)}"
            )

        checkpoint_result: Dict[str, Any] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch + 1,
            "inference": metrics_by_inference,
        }
        if model_hcc_diagnostics:
            checkpoint_result["model_hcc_diagnostics"] = model_hcc_diagnostics
        for alternative_mode in inference_modes:
            if alternative_mode == reference_mode:
                continue
            checkpoint_result[
                f"{alternative_mode}_minus_{reference_mode}"
            ] = _metric_deltas(
                metrics_by_inference[reference_mode],
                metrics_by_inference[alternative_mode],
            )
        results[checkpoint_mode] = checkpoint_result

    payload = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "model": model_name,
        "dataset": str(cfg.dataset.name),
        "seed": int(cfg.train.seed),
        "split": "test",
        "requested_inference_mode": args.inference_mode,
        "resolved_inference_modes": list(inference_modes),
        "native_inference_mode": native_mode,
        "paired_reference_mode": reference_mode,
        "hcc_trained_run": hcc_trained,
        "legacy_mode_names": {
            mode: legacy_mode_name(mode, model_name, hcc_trained)
            for mode in inference_modes
            if legacy_mode_name(mode, model_name, hcc_trained) is not None
        },
        "checkpoint_mode": args.checkpoint_mode,
        "test_split_source": (
            "official_dataset_adapter_fallback"
            if annotation_fallback
            else "run_configured"
        ),
        "evaluation_config_adjustments": evaluation_config_adjustments,
        "inference_rules": {
            mode: rule.describe() for mode, rule in inference_rules.items()
        },
        "inference_rule_notes": {
            "grid": (
                "readout (node_score, subspace_norm) x transform (none, hcc); "
                "every cell is defined for every model and all cells are scored "
                "from one shared forward pass"
            ),
            "native": (
                f"`{native_mode}` reproduces this checkpoint's own inference. "
                "Monotone per-level normalizations such as sigmoid or softmax "
                "are not applied, which leaves every reported metric unchanged "
                "because all of them are computed from argmax decoding."
            ),
            "hcc_alpha": (
                "`hcc_` cells always project, whether or not the run itself "
                "trained with HCC; check "
                "`model_hcc_diagnostics.proj_constraint_alpha` to see whether "
                "the model's own forward pass applied HCC at test time."
            ),
            "hcc_node_score_invariance": (
                "The projection subtracts one constant per parent from that "
                "parent's children, so sibling rankings survive whenever the "
                "readout is monotone in the coordinate. For signed classifier "
                "logits that makes every top-down metric identical to the "
                "untransformed readout and only independent decoding can move. "
                "It does not hold for Hier-COS, where the readout takes the "
                "magnitude."
            ),
            "subspace_norm_sign": (
                "An L2 norm squares its inputs, so subspace_norm discards the "
                "sign of a signed classifier logit. That is the substantive "
                "content of the identity-frame assumption."
            ),
        },
        "checkpoints": results,
    }
    _save_yaml(output_path, payload, overwrite=bool(args.overwrite))
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
