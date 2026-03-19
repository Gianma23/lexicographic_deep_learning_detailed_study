from typing import Any, Dict, List, Optional

import torch

from models import compute_loss
from .eval import evaluate_batch
from .metrics import merge_metric_batches
from .mixup import build_mixup_fn

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    cfg: Any,
    num_classes_per_level: Optional[List[int]] = None,
    taxonomy: Optional[Dict] = None,
) -> Dict[str, float]:
    model.train()
    loss_vals = []
    batch_metrics = []
    running_loss_total = 0.0
    running_loss_count = 0
    mixup_fn = build_mixup_fn(cfg, num_classes_per_level=num_classes_per_level)

    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"
    use_pbar = bool(cfg.train.get("progress_bar", True)) and tqdm is not None
    iterator = loader
    if use_pbar:
        iterator = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)

    for images, labels, _ in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        targets_for_loss = labels
        if mixup_fn is not None:
            target_levels = [labels[:, level] for level in range(labels.size(1))]
            mixup_out = mixup_fn(images, target_levels)
            images = mixup_out[0]
            soft_targets_per_level = list(mixup_out[1:])
            targets_for_loss = {"soft_targets_per_level": soft_targets_per_level}

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

        total_key = "loss_total" if "loss_total" in loss_dict else "total"
        if total_key in loss_dict:
            running_loss_total += float(loss_dict[total_key])
            running_loss_count += 1

        if use_pbar:
            avg_total_loss = running_loss_total / max(1, running_loss_count)
            postfix = {"loss": f"{avg_total_loss:.4f}"}
            full_path = batch_metric.get("acc_path")
            if full_path is not None:
                postfix["acc_path"] = f"{full_path:.4f}"
            iterator.set_postfix(postfix)

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
            output = model(images)
            _, loss_dict = compute_loss(cfg, output, labels, taxonomy)

        #loss_vals.append(loss_dict)
        batch_metrics.append(evaluate_batch(output, labels, taxonomy))

    #losses = merge_metric_batches(loss_vals)
    metrics = merge_metric_batches(batch_metrics)
    #losses.update(metrics)
    return metrics
