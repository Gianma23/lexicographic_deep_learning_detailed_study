from typing import Any, Dict, List, Optional

import torch

from models import compute_loss
from .eval import evaluate_batch
from .metrics import merge_metric_batches
from .mixup import build_mixup_fn
from .trunk_metrics import (
    _capture_trainable_param_snapshot,
    _extract_level_losses,
    _prepare_lexicographic_update,
    _prepare_trunk_grad_metrics,
    _trainable_named_params,
    _trunk_param_norm_metrics,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _set_model_epoch(model: torch.nn.Module, epoch: int) -> None:
    if hasattr(model, "set_epoch"):
        model.set_epoch(epoch)
        return
    wrapped_model = getattr(model, "module", None)
    if wrapped_model is not None and hasattr(wrapped_model, "set_epoch"):
        wrapped_model.set_epoch(epoch)


def _section_to_dict(section: Any) -> Dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if hasattr(section, "items"):
        return {k: v for k, v in section.items()}
    return {}


def _resolve_lexicographic_cfg(cfg: Any) -> Dict[str, Any]:
    train_cfg = _section_to_dict(getattr(cfg, "train", None))
    raw_lex_cfg = _section_to_dict(train_cfg.get("lexicographic", None))
    enabled = bool(raw_lex_cfg.get("enabled", False))
    eps = float(raw_lex_cfg.get("eps", 1e-12))
    if eps <= 0.0:
        eps = 1e-12
    log_metrics = bool(raw_lex_cfg.get("log_metrics", True))
    return {"enabled": enabled, "eps": eps, "log_metrics": log_metrics}


def _validate_lexicographic_enabled(cfg: Any, level_losses: List[torch.Tensor]) -> None:
    model_cfg = _section_to_dict(getattr(cfg, "model", None))
    model_name = str(model_cfg.get("name", "")).strip().lower()
    if model_name != "hcast":
        raise ValueError(
            "train.lexicographic.enabled=true is currently supported only for model.name='hcast'."
        )

    loss_cfg = _section_to_dict(model_cfg.get("loss", None))
    if bool(loss_cfg.get("globalkl", False)):
        raise ValueError(
            "train.lexicographic.enabled=true requires model.loss.globalkl=false "
            "(pure level-loss lexicographic mode)."
        )

    if len(level_losses) != 3:
        raise ValueError(
            "train.lexicographic.enabled=true requires exactly 3 differentiable level losses "
            "(coarse, mid, fine)."
        )


def _assign_grads_to_params(
    params: List[torch.nn.Parameter],
    grads: List[Optional[torch.Tensor]],
) -> None:
    for param, grad in zip(params, grads):
        if grad is None:
            param.grad = None
            continue
        param.grad = grad.detach()


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    cfg: Any,
    epoch: int = 0,
    num_classes_per_level: Optional[List[int]] = None,
    taxonomy: Optional[Dict] = None,
) -> Dict[str, Any]:
    model.train()
    _set_model_epoch(model, epoch)
    loss_vals = []
    batch_metrics = []
    grad_metric_vals = []
    mixup_fn = build_mixup_fn(cfg, num_classes_per_level=num_classes_per_level)
    trainable_named_params = _trainable_named_params(model)
    trainable_params = [param for _, param in trainable_named_params]
    start_param_snapshot = _capture_trainable_param_snapshot(trainable_params)
    resolved_trunk_masks: Optional[Dict[str, List[bool]]] = None
    lex_cfg = _resolve_lexicographic_cfg(cfg)
    lex_validated = False

    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"
    use_pbar = bool(cfg.train.get("progress_bar", True)) and tqdm is not None
    iterator = loader
    if use_pbar:
        iterator = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)

    for images, labels, _ in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        targets_for_loss = labels
        if mixup_fn is not None:
            target_levels = [labels[:, level] for level in range(labels.size(1))]
            mixup_out = mixup_fn(images, target_levels)
            images = mixup_out[0]
            soft_targets_per_level = list(mixup_out[1:])
            targets_for_loss = {
                "soft_targets_per_level": soft_targets_per_level,
                "hard_targets": labels,
            }

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)
            loss, loss_dict, loss_aux = compute_loss(
                cfg,
                output,
                targets_for_loss,
                taxonomy,
                return_aux=True,
            )

        level_losses = _extract_level_losses(loss_aux)
        grad_state, grad_metrics = _prepare_trunk_grad_metrics(trainable_named_params, level_losses)
        if resolved_trunk_masks is None and grad_state is not None:
            trunk_masks = grad_state.get("trunk_masks")
            if isinstance(trunk_masks, dict):
                resolved_trunk_masks = trunk_masks

        batch_grad_metrics: Dict[str, float] = dict(grad_metrics)

        if lex_cfg["enabled"] and not lex_validated:
            _validate_lexicographic_enabled(cfg, level_losses)
            lex_validated = True

        if lex_cfg["enabled"]:
            level_losses_for_lex = list(level_losses[:3])
            lex_grad_scale = 1.0
            if scaler is not None and use_amp:
                lex_grad_scale = float(scaler.get_scale())
                level_losses_for_lex = [scaler.scale(level_loss) for level_loss in level_losses_for_lex]

            lex_state, lex_metrics = _prepare_lexicographic_update(
                trainable_named_params=trainable_named_params,
                level_losses=level_losses_for_lex,
                eps=float(lex_cfg["eps"]),
                include_metrics=bool(lex_cfg["log_metrics"]),
                grad_scale=lex_grad_scale,
            )
            if lex_state is None:
                raise ValueError(
                    "train.lexicographic.enabled=true requires trainable parameters and 3 level losses."
                )

            projected_grads = lex_state.get("projected_grads", {})
            total_grads = projected_grads.get("total")
            if not isinstance(total_grads, tuple):
                raise RuntimeError("Lexicographic update failed to produce total gradients.")

            _assign_grads_to_params(trainable_params, list(total_grads))

            if lex_metrics:
                batch_grad_metrics.update(lex_metrics)
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        else:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        loss_vals.append(loss_dict)
        if batch_grad_metrics:
            grad_metric_vals.append(batch_grad_metrics)
        batch_metric = evaluate_batch(output, labels, taxonomy)
        batch_metrics.append(batch_metric)

    loss_metrics = merge_metric_batches(loss_vals)
    metrics = merge_metric_batches(batch_metrics)
    grad_metrics = merge_metric_batches(grad_metric_vals)
    trunk_param_metrics = _trunk_param_norm_metrics(
        params=trainable_params,
        start_snapshot=start_param_snapshot,
        trunk_masks=resolved_trunk_masks,
    )
    metrics.update(grad_metrics)
    metrics.update(trunk_param_metrics)

    train_outputs: Dict[str, Any] = dict(loss_metrics)
    train_outputs.update(metrics)

    # Expose exact loss keys so downstream loggers do not need name-based heuristics.
    train_outputs["__loss_keys__"] = sorted(loss_metrics.keys())
    return train_outputs


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    cfg: Any,
    epoch: int = 0,
    taxonomy: Optional[Dict] = None,
) -> Dict[str, float]:
    model.eval()
    _set_model_epoch(model, epoch)

    batch_metrics = []
    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)

        batch_metrics.append(evaluate_batch(output, labels, taxonomy))

    metrics = merge_metric_batches(batch_metrics)
    return metrics
