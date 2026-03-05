from typing import Any, Dict, Optional

import torch

from models import compute_loss
from .eval import evaluate_batch
from .metrics import merge_metric_batches


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    cfg: Any,
    taxonomy: Optional[Dict] = None,
) -> Dict[str, float]:
    model.train()
    loss_vals = []
    batch_metrics = []

    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images, targets=labels)
            loss, loss_dict = compute_loss(cfg, output, labels, taxonomy)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_vals.append(loss_dict)
        batch_metrics.append(evaluate_batch(output, labels, taxonomy))

    losses = merge_metric_batches(loss_vals)
    metrics = merge_metric_batches(batch_metrics)
    losses.update(metrics)
    return losses


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    cfg: Any,
    taxonomy: Optional[Dict] = None,
) -> Dict[str, float]:
    model.eval()

    loss_vals = []
    batch_metrics = []
    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images, targets=labels)
            _, loss_dict = compute_loss(cfg, output, labels, taxonomy)

        loss_vals.append(loss_dict)
        batch_metrics.append(evaluate_batch(output, labels, taxonomy))

    losses = merge_metric_batches(loss_vals)
    metrics = merge_metric_batches(batch_metrics)
    losses.update(metrics)
    return losses
