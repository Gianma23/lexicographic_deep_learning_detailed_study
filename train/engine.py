from typing import Any, Dict, List, Optional

import torch

from models import compute_loss
from .eval import evaluate_batch
from .lte_adam import LTEAdam
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


def _ordered_level_losses_for_lte(
    model: torch.nn.Module,
    cfg: Any,
    images: torch.Tensor,
    targets_for_loss: Any,
    taxonomy: Optional[Dict],
) -> List[float]:
    """Evaluate ordered per-level losses for LTE admissibility checks."""
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            output = model(images)
            _, _, aux_payload = compute_loss(
                cfg,
                output,
                targets_for_loss,
                taxonomy,
                return_aux=True,
            )
        level_losses = aux_payload.get("level_losses")
        if not isinstance(level_losses, (list, tuple)):
            raise ValueError(
                "compute_loss(..., return_aux=True) must provide `level_losses` list for LTE checks."
            )
        ordered: List[float] = []
        for level, level_loss in enumerate(level_losses):
            if not torch.is_tensor(level_loss):
                raise TypeError(
                    f"`level_losses[{level}]` must be a torch.Tensor, got {type(level_loss).__name__}."
                )
            if level_loss.numel() != 1:
                raise ValueError(
                    f"`level_losses[{level}]` must be scalar, got shape {tuple(level_loss.shape)}."
                )
            ordered.append(float(level_loss.detach().item()))
        return ordered
    finally:
        if was_training:
            model.train()


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
    mixup_fn = build_mixup_fn(cfg, num_classes_per_level=num_classes_per_level)
    lte_step_infos: List[Dict[str, Any]] = []

    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"
    use_lte_adam = isinstance(optimizer, LTEAdam)
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

        if use_lte_adam:
            losses_fn = lambda: _ordered_level_losses_for_lte(  # noqa: E731
                model=model,
                cfg=cfg,
                images=images,
                targets_for_loss=targets_for_loss,
                taxonomy=taxonomy,
            )

            if scaler is not None and use_amp:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    output = model(images)
                    loss, loss_dict = compute_loss(cfg, output, targets_for_loss, taxonomy)
                scaler.scale(loss).backward()
                lte_step_info = scaler.step(
                    optimizer,
                    losses_fn=losses_fn,
                    train_loss=float(loss.detach().item()),
                )
                scaler.update()
            else:
                closure_cache: Dict[str, Any] = {}

                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    output_closure = model(images)
                    loss_closure, loss_dict_closure = compute_loss(
                        cfg,
                        output_closure,
                        targets_for_loss,
                        taxonomy,
                    )
                    loss_closure.backward()
                    closure_cache["output"] = output_closure
                    closure_cache["loss_dict"] = loss_dict_closure
                    return loss_closure

                lte_step_info = optimizer.step(closure=closure, losses_fn=losses_fn)
                if "output" not in closure_cache or "loss_dict" not in closure_cache:
                    raise RuntimeError("LTEAdam closure did not populate output/loss_dict.")
                output = closure_cache["output"]
                loss_dict = closure_cache["loss_dict"]

            if isinstance(lte_step_info, dict):
                lte_step_infos.append(lte_step_info)
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(images)
                loss, loss_dict = compute_loss(cfg, output, targets_for_loss, taxonomy)

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        loss_vals.append(loss_dict)
        batch_metric = evaluate_batch(output, labels, taxonomy)
        batch_metrics.append(batch_metric)

    loss_metrics = merge_metric_batches(loss_vals)
    metrics = merge_metric_batches(batch_metrics)

    train_outputs: Dict[str, Any] = dict(loss_metrics)
    train_outputs.update(metrics)
    if use_lte_adam and lte_step_infos:
        attempted = float(len(lte_step_infos))
        accepted = float(sum(1.0 for info in lte_step_infos if bool(info.get("accepted", False))))
        rejected = attempted - accepted
        mean_backtracks = sum(float(info.get("num_backtracks", 0.0)) for info in lte_step_infos) / attempted
        mean_step_scale = sum(float(info.get("step_scale", 0.0)) for info in lte_step_infos) / attempted
        fallback_used = float(
            sum(
                1.0
                for info in lte_step_infos
                if str(info.get("inadmissible_reason", "")).startswith("fallback_step_applied_after_")
            )
        )
        train_outputs["lte_batches"] = attempted
        train_outputs["lte_accept_rate"] = accepted / attempted
        train_outputs["lte_reject_rate"] = rejected / attempted
        train_outputs["lte_mean_backtracks"] = mean_backtracks
        train_outputs["lte_mean_step_scale"] = mean_step_scale
        train_outputs["lte_fallback_rate"] = fallback_used / attempted

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
