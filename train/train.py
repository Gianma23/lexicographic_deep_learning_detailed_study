import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from datasets import build_dataloader
from models import build_model
from .artifacts import (
    append_epoch_metrics,
    as_float_dict,
    initialize_epoch_rows,
    save_train_level_losses_plot,
    save_val_level_accuracies_plot,
    save_yaml,
    update_level_history,
)
from .config_loader import load_config
from .engine import evaluate, train_one_epoch
from .eval import pretty_metrics
from .utils import (
    build_optimizer,
    build_scheduler,
    load_finetune_checkpoint,
    metric_for_best,
    resume_if_available,
    save_checkpoint,
    seed_everything,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Unified hierarchical image classification training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*", default=[])
    return parser.parse_args()


def _print_loader_sizes(train_loader, val_loader, test_loader):
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        sample_count = len(getattr(loader, "dataset", None))
        batch_count = len(loader)
        samples_txt = str(sample_count) if sample_count is not None else "unknown"
        batches_txt = str(batch_count) if batch_count is not None else "unknown"
        print(f"[data] {split_name:<5} samples={samples_txt} batches={batches_txt}")


def _print_model_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] parameters total={total_params:,} trainable={trainable_params:,}")


# ======================================================================== #
#                    M A I N   T R A I N I N G   L O O P                   #
# ======================================================================== #
def main():
    args = _parse_args()
    cfg, cfg_resolved = load_config(args.config, args.overrides)

    seed_everything(int(cfg.train.seed), bool(cfg.runtime.get("deterministic", True)))

    device = torch.device(str(cfg.train.get("device", "cuda" if torch.cuda.is_available() else "cpu")))

    train_loader, num_classes_per_level, taxonomy = build_dataloader(cfg, split="train")
    val_loader, _, _ = build_dataloader(cfg, split="val")
    test_loader, _, _ = build_dataloader(cfg, split="test")
    _print_loader_sizes(train_loader, val_loader, test_loader)

    model = build_model(cfg, num_classes_per_level, taxonomy)
    _print_model_parameter_count(model)
    load_finetune_checkpoint(cfg, model)
    model = model.to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    # Enable AMP only on CUDA; GradScaler is a no-op when disabled.
    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    out_dir = Path(str(cfg.train.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = out_dir / "latest.pt"
    best_ckpt = out_dir / "best.pt"
    resolved_cfg_path = out_dir / "config_resolved.yaml"
    epoch_metrics_jsonl_path = out_dir / "epoch_metrics.jsonl"
    epoch_metrics_csv_path = out_dir / "epoch_metrics.csv"
    test_metrics_path = out_dir / "test_metrics.yaml"

    save_yaml(resolved_cfg_path, cfg_resolved)
    print(f"[artifact] saved resolved config: {resolved_cfg_path}")

    start_epoch, best_metric = resume_if_available(str(cfg.train.get("resume", "")), model, optimizer, scheduler, scaler)
    epoch_rows: List[Dict[str, Any]] = initialize_epoch_rows(
        start_epoch=start_epoch,
        jsonl_path=epoch_metrics_jsonl_path,
        csv_path=epoch_metrics_csv_path,
    )

    epochs = int(cfg.train.epochs)
    level_names = [str(name) for name in cfg.dataset.get("levels", [])]
    epoch_ids = []
    level_loss_history = {}
    level_val_acc_history = {}
    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, taxonomy)
        val_metrics = evaluate(model, val_loader, device, cfg, taxonomy)

        # Collect per-level training losses for plotting.
        epoch_ids.append(epoch + 1)
        update_level_history(
            history=level_loss_history,
            epoch_ids=epoch_ids,
            metrics=train_metrics,
            metric_prefix="loss_level_",
        )

        # Collect per-level validation accuracies for plotting.
        update_level_history(
            history=level_val_acc_history,
            epoch_ids=epoch_ids,
            metrics=val_metrics,
            metric_prefix="acc_level_",
        )

        score = metric_for_best(val_metrics)
        if scheduler is not None:
            scheduler.step(epoch + 1, score)

        if score > best_metric:
            best_metric = score
            # Keep the best checkpoint according to validation metric
            save_checkpoint(
                str(best_ckpt),
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                cfg_resolved,
            )

        # Always refresh latest checkpoint for resumable training
        save_checkpoint(
            str(latest_ckpt),
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metric,
            cfg_resolved,
        )

        append_epoch_metrics(
            rows=epoch_rows,
            jsonl_path=epoch_metrics_jsonl_path,
            csv_path=epoch_metrics_csv_path,
            epoch=epoch + 1,
            lr=float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float("nan"),
            best_metric=best_metric,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        epoch_tag = f"[epoch {epoch + 1:03d}/{epochs:03d}]"
        print(f"{epoch_tag} train | {pretty_metrics(train_metrics, level_names=level_names)}")
        print(f"{epoch_tag} val   | {pretty_metrics(val_metrics, level_names=level_names)}")
        print("")

    loss_plot_path = save_train_level_losses_plot(
        out_dir=str(out_dir),
        epoch_ids=epoch_ids,
        level_loss_history=level_loss_history,
        level_names=level_names,
        model_name=str(cfg.model.name),
    )
    if loss_plot_path:
        print(f"saved_train_loss_plot: {loss_plot_path}")
    elif level_loss_history:
        print("saved_train_loss_plot: skipped (matplotlib not installed)")

    val_plot_path = save_val_level_accuracies_plot(
        out_dir=str(out_dir),
        epoch_ids=epoch_ids,
        level_acc_history=level_val_acc_history,
        level_names=level_names,
        model_name=str(cfg.model.name),
    )
    if val_plot_path:
        print(f"saved_val_accuracy_plot: {val_plot_path}")
    elif level_val_acc_history:
        print("saved_val_accuracy_plot: skipped (matplotlib not installed)")

    # Evaluate the best validation checkpoint on the test set.
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"[test] loaded best checkpoint from: {best_ckpt}")
    test_metrics = evaluate(model, test_loader, device, cfg, taxonomy)
    print(f"[test] {pretty_metrics(test_metrics, level_names=level_names)}")
    test_payload = {
        "best_checkpoint": str(best_ckpt),
        "best_metric": float(best_metric),
        "test_metrics": as_float_dict(test_metrics),
    }
    save_yaml(test_metrics_path, test_payload)
    print(f"[artifact] saved test metrics: {test_metrics_path}")


if __name__ == "__main__":
    main()

