import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from datasets import build_dataloader
from models import build_model
from .config_loader import load_config
from .engine import evaluate, train_one_epoch
from .metric_formatting import pretty_metrics
from .runtime.checkpointing import resume_if_available, save_checkpoint
from .runtime.finetune import load_finetune_checkpoint, load_trusted_checkpoint
from .runtime.optimization import build_optimizer, build_scheduler, seed_everything
from .runtime.selection import BEST_SELECTION_MODES, metric_for_best
from .training_logger import TrainingLogger


def _parse_args():
    parser = argparse.ArgumentParser(description="Unified hierarchical image classification training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*", default=[])
    return parser.parse_args()


def _print_loader_sizes(train_loader, val_loader, test_loader) -> None:
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        sample_count = len(getattr(loader, "dataset", None))
        batch_count = len(loader)
        samples_txt = str(sample_count) if sample_count is not None else "unknown"
        batches_txt = str(batch_count) if batch_count is not None else "unknown"
        print(f"[data] {split_name:<5} samples={samples_txt} batches={batches_txt}")


def _print_model_parameter_count(model) -> None:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"[model] parameters total={total_params:,} trainable={trainable_params:,}")


def _apply_lr_scaling_if_enabled(cfg: Any, cfg_resolved: Dict[str, Any]) -> None:
    if not bool(cfg.train.get("scale_lr", False)):
        return

    base_lr = float(cfg.optim.lr)
    reference_batch = float(cfg.train.get("scale_lr_reference_batch_size", 512.0))
    if reference_batch <= 0:
        raise ValueError("train.scale_lr_reference_batch_size must be > 0.")

    batch_size = float(cfg.dataloader.batch_size)
    # Simplified to single-process scaling (world_size assumed 1).
    scaled_lr = base_lr * batch_size / reference_batch

    cfg.optim.lr = float(scaled_lr)
    if isinstance(cfg_resolved, dict):
        cfg_resolved.setdefault("optim", {})
        cfg_resolved.setdefault("train", {})
        cfg_resolved["optim"]["lr_base"] = float(base_lr)
        cfg_resolved["optim"]["lr"] = float(scaled_lr)
        cfg_resolved["train"]["scale_lr"] = True
        cfg_resolved["train"]["scale_lr_reference_batch_size"] = float(reference_batch)

    print(
        "[optim] scale_lr enabled: "
        f"base_lr={base_lr:.8g} batch_size={int(batch_size)} "
        f"reference_batch={reference_batch:g} -> scaled_lr={scaled_lr:.8g}"
    )


def run_training(cfg: Any, cfg_resolved: Dict[str, Any]) -> None:
    _apply_lr_scaling_if_enabled(cfg, cfg_resolved)
    seed_everything(int(cfg.train.seed), bool(cfg.runtime.get("deterministic", True)))

    device = torch.device(str(cfg.train.get("device", "cuda" if torch.cuda.is_available() else "cpu")))

    train_loader, num_classes_per_level, taxonomy = build_dataloader(cfg, split="train")
    val_loader, _, _ = build_dataloader(cfg, split="val")
    test_loader, _, _ = build_dataloader(cfg, split="test")
    loaders_by_split = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
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
    best_ckpts = {mode: out_dir / f"best_{mode}.pt" for mode in BEST_SELECTION_MODES}

    level_names = [str(name) for name in cfg.dataset.get("levels", [])]
    start_epoch, best_metrics, resume_info = resume_if_available(
        str(cfg.train.get("resume", "")),
        model,
        optimizer,
        scheduler,
        scaler,
        cfg_resolved=cfg_resolved,
        loaders=loaders_by_split,
    )
    logger = TrainingLogger(
        output_dir=out_dir,
        start_epoch=start_epoch,
        level_names=level_names,
    )
    resolved_cfg_path = logger.save_resolved_config(cfg_resolved)
    print(f"[LOGGER] saved resolved config: {resolved_cfg_path}")
    print(f"[LOGGER] logging run events to: {logger.run_log_path}")
    if str(cfg.train.get("resume", "")).strip():
        logger.log_resume(resume_info.to_dict())
        print(
            "[resume] "
            f"path={resume_info.resume_path} "
            f"found={resume_info.checkpoint_found} "
            f"start_epoch={resume_info.start_epoch} "
            f"config_ok={resume_info.config_check_passed} "
            f"rng_restored={resume_info.rng_state_restored} "
            f"loader_rng_restored={resume_info.loader_rng_state_restored} "
            f"full_replay={resume_info.full_reproducibility_restored}"
        )

    epochs = int(cfg.train.epochs)
    stop_epoch = int(cfg.train.get("stop_epoch", epochs))
    if stop_epoch < 1:
        raise ValueError("train.stop_epoch must be >= 1.")
    if stop_epoch > epochs:
        raise ValueError("train.stop_epoch must be <= train.epochs.")
    if stop_epoch < epochs:
        print(f"[train] stop_epoch active: {stop_epoch} (scheduler horizon: {epochs})")
    if start_epoch >= stop_epoch:
        print(f"[train] resume start_epoch={start_epoch} >= stop_epoch={stop_epoch}; skipping train loop.")

    for epoch in range(start_epoch, stop_epoch):
        train_outputs = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
            epoch=epoch,
            num_classes_per_level=num_classes_per_level,
            taxonomy=taxonomy,
        )
        val_metrics = evaluate(model, val_loader, device, cfg, epoch=epoch, taxonomy=taxonomy)

        scores = {mode: metric_for_best(val_metrics, mode=mode) for mode in BEST_SELECTION_MODES}
        if scheduler is not None:
            scheduler.step(epoch + 1, scores["topdown"])

        improved_modes = []
        for mode, score in scores.items():
            if score > best_metrics[mode]:
                best_metrics[mode] = score
                improved_modes.append(mode)

        # Keep the best checkpoint for each validation-ranking mode.
        for mode in improved_modes:
            save_checkpoint(
                str(best_ckpts[mode]),
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metrics,
                cfg_resolved,
                loaders=loaders_by_split,
            )

        # Always refresh latest checkpoint for resumable training.
        save_checkpoint(
            str(latest_ckpt),
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metrics,
            cfg_resolved,
            loaders=loaders_by_split,
        )

        logger.log_epoch(
            epoch=epoch + 1,
            lr=float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float("nan"),
            best_metrics=best_metrics,
            train_outputs=train_outputs,
            val_metrics=val_metrics,
        )

        epoch_tag = f"[epoch {epoch + 1:03d}/{stop_epoch:03d}]"
        print(f"{epoch_tag} train | {pretty_metrics(train_outputs, level_names=level_names)}")
        print(f"{epoch_tag} val   | {pretty_metrics(val_metrics, level_names=level_names)}")
        print("")

    # Evaluate each best validation checkpoint on the test set.
    test_results = {}
    for mode in BEST_SELECTION_MODES:
        best_ckpt = best_ckpts[mode]
        if not best_ckpt.exists():
            raise RuntimeError(
                f"Best {mode} checkpoint not found at {best_ckpt}. "
                "Training likely ran zero epochs; check train.resume and train.stop_epoch."
            )

        checkpoint = load_trusted_checkpoint(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        checkpoint_epoch = int(checkpoint.get("epoch", epochs - 1))
        checkpoint_best_metrics = checkpoint["best_metrics"]
        best_metric = float(checkpoint_best_metrics[mode])

        print(f"[test:{mode}] loaded best checkpoint from: {best_ckpt} (epoch {checkpoint_epoch + 1})")
        test_metrics = evaluate(model, test_loader, device, cfg, epoch=checkpoint_epoch, taxonomy=taxonomy)
        print(f"[test:{mode}] {pretty_metrics(test_metrics, level_names=level_names)}")
        test_results[mode] = {
            "best_checkpoint": str(best_ckpt),
            "best_epoch": checkpoint_epoch + 1,
            "best_metric": best_metric,
            "test_metrics": test_metrics,
        }

    logger.log_test(test_results)


def main():
    args = _parse_args()
    cfg, cfg_resolved = load_config(args.config, args.overrides)
    run_training(cfg, cfg_resolved)


if __name__ == "__main__":
    main()
