from typing import Any, Dict, Optional

import torch

from models import compute_loss
from .eval import evaluate_batch
from .metrics import merge_metric_batches
from .mixup import apply_mixup, blend_metrics

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
    taxonomy: Optional[Dict] = None,
) -> Dict[str, float]:
    model.train()
    loss_vals = []
    batch_metrics = []
    running_loss_total = 0.0
    running_loss_count = 0
    model_name = str(cfg.model.name).lower()
    mixup_label_smoothing = float(cfg.train.get("smoothing", 0.1))

    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"
    use_pbar = bool(cfg.train.get("progress_bar", True)) and tqdm is not None
    iterator = loader
    if use_pbar:
        iterator = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)

    for images, labels, _ in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images_mixed, labels_a, labels_b, lam, mixup_applied = apply_mixup(images, labels, cfg)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images_mixed, targets=labels_a)
            if mixup_applied:
                if model_name == "hcast":
                    mixup_targets = {
                        "labels_a": labels_a,
                        "labels_b": labels_b,
                        "lam": float(lam),
                        "label_smoothing": mixup_label_smoothing,
                    }
                    loss, loss_dict = compute_loss(cfg, output, mixup_targets, taxonomy)
                else:
                    loss_a, loss_dict_a = compute_loss(cfg, output, labels_a, taxonomy)
                    loss_b, loss_dict_b = compute_loss(cfg, output, labels_b, taxonomy)
                    loss = lam * loss_a + (1.0 - lam) * loss_b
                    loss_dict = blend_metrics(loss_dict_a, loss_dict_b, lam=lam)
            else:
                loss, loss_dict = compute_loss(cfg, output, labels_a, taxonomy)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_dict["mixup_lam"] = float(lam)
        loss_dict["mixup_applied"] = 1.0 if mixup_applied else 0.0
        loss_vals.append(loss_dict)
        if mixup_applied:
            metric_a = evaluate_batch(output, labels_a, taxonomy)
            metric_b = evaluate_batch(output, labels_b, taxonomy)
            batch_metric = blend_metrics(metric_a, metric_b, lam=lam)
        else:
            batch_metric = evaluate_batch(output, labels_a, taxonomy)
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
            output = model(images, targets=labels)
            _, loss_dict = compute_loss(cfg, output, labels, taxonomy)

        #loss_vals.append(loss_dict)
        batch_metrics.append(evaluate_batch(output, labels, taxonomy))

    #losses = merge_metric_batches(loss_vals)
    metrics = merge_metric_batches(batch_metrics)
    #losses.update(metrics)
    return metrics
