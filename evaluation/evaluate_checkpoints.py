"""Evaluate checkpoints with native and/or post-hoc inference rules."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

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
from train.runtime.selection import BEST_SELECTION_MODES

from .posthoc_inference import (
    IdentityFrameHierCosInference,
    NativeHierCosNodeSoftmaxInference,
)


INFERENCE_MODES = ("normal", "hiercos", "node_softmax", "both")
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
            "Run test-only evaluation from existing best checkpoints. For "
            "non-Hier-COS models, `both` means normal+hiercos. For native "
            "Hier-COS, `both` means normal+node_softmax."
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


def _resolve_inference_modes(requested: str, model_name: str) -> Sequence[str]:
    if requested == "both":
        if model_name == "hiercos":
            return "normal", "node_softmax"
        return "normal", "hiercos"
    if requested == "hiercos" and model_name == "hiercos":
        raise ValueError(
            "Native Hier-COS normal inference already uses its distance scores; "
            "select `node_softmax` to remove distance-based inference."
        )
    if requested == "node_softmax" and model_name != "hiercos":
        raise ValueError("`node_softmax` is defined only for native Hier-COS checkpoints.")
    return (requested,)


def _set_model_epoch(model: torch.nn.Module, epoch: int) -> None:
    if hasattr(model, "set_epoch"):
        model.set_epoch(epoch)
        return
    wrapped_model = getattr(model, "module", None)
    if wrapped_model is not None and hasattr(wrapped_model, "set_epoch"):
        wrapped_model.set_epoch(epoch)


def _set_hcc_final_test_active(model: torch.nn.Module, active: bool) -> None:
    if hasattr(model, "set_hcc_final_test_active"):
        model.set_hcc_final_test_active(active)
        return
    wrapped_model = getattr(model, "module", None)
    if wrapped_model is not None and hasattr(wrapped_model, "set_hcc_final_test_active"):
        wrapped_model.set_hcc_final_test_active(active)


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
    hiercos_inference: IdentityFrameHierCosInference | None,
    node_softmax_inference: NativeHierCosNodeSoftmaxInference | None,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    metric_batches: Dict[str, list] = {mode: [] for mode in inference_modes}
    batch_weights = []
    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"
    taxonomy_dict = dict(taxonomy)

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)

        for inference_mode in inference_modes:
            if inference_mode == "normal":
                inference_output = output
            elif inference_mode == "hiercos":
                if hiercos_inference is None:
                    raise RuntimeError("Hier-COS inference was not initialized.")
                inference_output = hiercos_inference.transform_output(output)
            elif inference_mode == "node_softmax":
                if node_softmax_inference is None:
                    raise RuntimeError("Node-softmax inference was not initialized.")
                inference_output = node_softmax_inference.transform_output(output)
            else:
                raise RuntimeError(f"Unexpected inference mode: {inference_mode}")
            metric_batches[inference_mode].append(
                evaluate_batch(inference_output, labels, taxonomy=taxonomy_dict)
            )
        batch_weights.append(int(labels.size(0)))

    return {
        mode: _outcome_metrics(
            merge_metric_batches(batches, batch_weights=batch_weights)
        )
        for mode, batches in metric_batches.items()
    }


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
    inference_modes = _resolve_inference_modes(args.inference_mode, model_name)
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
    test_loader, num_classes_per_level, taxonomy = build_dataloader(cfg, "test")
    if not isinstance(taxonomy, Mapping):
        raise ValueError("Post-hoc evaluation requires dataset taxonomy mappings.")

    model = build_model(cfg, num_classes_per_level, taxonomy).to(device)
    hiercos_inference = (
        IdentityFrameHierCosInference(num_classes_per_level, taxonomy)
        if "hiercos" in inference_modes
        else None
    )
    node_softmax_inference = (
        NativeHierCosNodeSoftmaxInference()
        if "node_softmax" in inference_modes
        else None
    )
    level_names = [str(name) for name in cfg.dataset.get("levels", [])]

    results: Dict[str, Any] = {}
    _set_hcc_final_test_active(model, True)
    try:
        for checkpoint_mode, checkpoint_path in checkpoint_paths.items():
            checkpoint = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state"])
            checkpoint_epoch = int(checkpoint.get("epoch", int(cfg.train.epochs) - 1))
            _set_model_epoch(model, checkpoint_epoch)

            print(
                f"[checkpoint:{checkpoint_mode}] {checkpoint_path} "
                f"(epoch {checkpoint_epoch + 1})"
            )
            metrics_by_inference = _evaluate_inference_modes(
                model=model,
                loader=test_loader,
                device=device,
                cfg=cfg,
                taxonomy=taxonomy,
                inference_modes=inference_modes,
                hiercos_inference=hiercos_inference,
                node_softmax_inference=node_softmax_inference,
            )
            for inference_mode, metrics in metrics_by_inference.items():
                print(
                    f"[test:{checkpoint_mode}:{inference_mode}] "
                    f"{pretty_metrics(metrics, level_names=level_names)}"
                )

            checkpoint_result: Dict[str, Any] = {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint_epoch + 1,
                "inference": metrics_by_inference,
            }
            if len(inference_modes) == 2:
                baseline_mode, alternative_mode = inference_modes
                checkpoint_result[
                    f"{alternative_mode}_minus_{baseline_mode}"
                ] = _metric_deltas(
                    metrics_by_inference[baseline_mode],
                    metrics_by_inference[alternative_mode],
                )
            results[checkpoint_mode] = checkpoint_result
    finally:
        _set_hcc_final_test_active(model, False)

    payload = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "model": model_name,
        "dataset": str(cfg.dataset.name),
        "seed": int(cfg.train.seed),
        "split": "test",
        "requested_inference_mode": args.inference_mode,
        "resolved_inference_modes": list(inference_modes),
        "checkpoint_mode": args.checkpoint_mode,
        "evaluation_config_adjustments": evaluation_config_adjustments,
        "inference_rules": {
            "normal": "model-native evaluation output",
            "hiercos": {
                "input": "native_raw_logits_per_level",
                "fixed_frame": "identity",
                "subspace": "ancestors+self+descendants",
                "score": "l2_projection_norm",
                "prediction": "raw_score_argmax",
                "training": False,
            },
            "node_softmax": {
                "input": "abs(native_hiercos_node_logits)",
                "normalization": "one_global_softmax_over_taxonomy_nodes",
                "score": "per-level selected node probability",
                "prediction": "raw_probability_argmax",
                "training": False,
            },
        },
        "checkpoints": results,
    }
    _save_yaml(output_path, payload, overwrite=bool(args.overwrite))
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
