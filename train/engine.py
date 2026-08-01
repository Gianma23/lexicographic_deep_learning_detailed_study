from typing import Any, Dict, List, Optional

import torch

from models import compute_loss
from .evaluation import evaluate_batch
from .lexicographic.config import resolve_lexicographic_config, validate_lexicographic_requirements
from .lexicographic.gradients import (
    assign_grads_to_params,
    capture_trainable_param_snapshot,
    compute_trunk_grad_metrics,
    compute_trunk_param_norm_metrics,
    extract_level_losses,
    get_trainable_named_params,
    prepare_lexicographic_update,
)
from .metrics import merge_metric_batches
from .mixup import build_mixup_fn

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


def _post_optimizer_step(model: torch.nn.Module, loss_aux: Any) -> None:
    """Run an optional model callback after an optimizer batch update."""
    target = getattr(model, "module", model)
    if hasattr(target, "post_optimizer_step"):
        target.post_optimizer_step(loss_aux)
        return
    base_model = getattr(target, "base_model", None)
    if base_model is not None and hasattr(base_model, "post_optimizer_step"):
        base_model.post_optimizer_step(loss_aux)


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
    batch_weights = []
    grad_metric_vals = []
    mixup_fn = build_mixup_fn(cfg, num_classes_per_level=num_classes_per_level)
    trainable_named_params = get_trainable_named_params(model)
    trainable_params = [param for _, param in trainable_named_params]
    start_param_snapshot = capture_trainable_param_snapshot(trainable_params)
    resolved_trunk_masks: Optional[Dict[str, List[bool]]] = None
    lex_cfg = resolve_lexicographic_config(cfg)
    lex_projection_active = lex_cfg.projection_active(epoch)
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

        level_losses = extract_level_losses(loss_aux)
        retain_graph_for_metrics = not lex_projection_active
        grad_state, grad_metrics = compute_trunk_grad_metrics(
            trainable_named_params,
            level_losses,
            retain_graph=retain_graph_for_metrics,
        )
        if resolved_trunk_masks is None and grad_state is not None:
            resolved_trunk_masks = dict(grad_state.trunk_masks)

        batch_grad_metrics: Dict[str, float] = dict(grad_metrics)

        if lex_cfg.enabled and not lex_validated:
            validate_lexicographic_requirements(cfg, level_losses)
            lex_validated = True

        if lex_projection_active:
            level_losses_for_lex = list(level_losses[:3])
            lex_grad_scale = 1.0
            if scaler is not None and use_amp:
                # GradScaler.step requires scale() to run in the iteration even
                # when custom gradients are assigned manually.
                scaler.scale(torch.ones((), device=device))
                lex_grad_scale = float(scaler.get_scale())

            precomputed_level_grad_map = None
            if grad_state is not None:
                precomputed_level_grad_map = grad_state.level_grad_map

            lex_state, lex_metrics = prepare_lexicographic_update(
                trainable_named_params=trainable_named_params,
                level_losses=level_losses_for_lex,
                eps=float(lex_cfg.eps),
                include_metrics=bool(lex_cfg.log_metrics),
                grad_scale=lex_grad_scale,
                projection_mode=str(lex_cfg.projection_mode),
                projection_rule=str(lex_cfg.projection_rule),
                precomputed_level_grad_map=precomputed_level_grad_map,
            )
            if lex_state is None:
                raise ValueError(
                    "train.lexicographic.enabled=true requires trainable parameters and 3 level losses."
                )

            projected_grads = lex_state.projected_grads
            total_grads = projected_grads.get("total")
            if not isinstance(total_grads, tuple):
                raise RuntimeError("Lexicographic update failed to produce total gradients.")

            assign_grads_to_params(trainable_params, list(total_grads))

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

        _post_optimizer_step(model, loss_aux)

        batch_weight = int(labels.size(0))
        loss_vals.append(loss_dict)
        batch_weights.append(batch_weight)
        if batch_grad_metrics:
            grad_metric_vals.append(batch_grad_metrics)
        batch_metric = evaluate_batch(output, labels, taxonomy)
        batch_metrics.append(batch_metric)

    loss_metrics = merge_metric_batches(loss_vals, batch_weights=batch_weights)
    metrics = merge_metric_batches(batch_metrics, batch_weights=batch_weights)
    # Gradient norms/cosines describe optimizer steps, not per-sample outcomes.
    grad_metrics = merge_metric_batches(grad_metric_vals)
    trunk_param_metrics = compute_trunk_param_norm_metrics(
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
    include_losses: bool = False,
) -> Dict[str, Any]:
    model.eval()
    _set_model_epoch(model, epoch)

    loss_vals = []
    batch_metrics = []
    batch_weights = []
    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)
            if include_losses:
                _, loss_dict = compute_loss(cfg, output, labels, taxonomy)

        if include_losses:
            loss_vals.append(loss_dict)
        batch_metrics.append(evaluate_batch(output, labels, taxonomy))
        batch_weights.append(int(labels.size(0)))

    metrics = merge_metric_batches(batch_metrics, batch_weights=batch_weights)
    if not include_losses:
        return metrics

    loss_metrics = merge_metric_batches(loss_vals, batch_weights=batch_weights)
    outputs: Dict[str, Any] = dict(loss_metrics)
    outputs.update(metrics)
    outputs["__loss_keys__"] = sorted(loss_metrics.keys())
    return outputs
