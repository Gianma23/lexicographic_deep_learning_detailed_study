import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

try:
    from omegaconf import OmegaConf  # type: ignore
except ImportError:  # pragma: no cover
    OmegaConf = None

from datasets import build_dataloader
from models import build_model
from .engine import evaluate, train_one_epoch
from .eval import pretty_metrics
from .utils import (
    build_optimizer,
    build_scheduler,
    load_finetune_checkpoint,
    metric_for_best,
    resume_if_available,
    save_checkpoint,
    save_val_level_accuracies_plot,
    save_train_level_losses_plot,
    seed_everything,
    step_scheduler,
)


class AttrDict(dict):
    # Lightweight dict wrapper to support attribute-style access (`cfg.train.seed`).
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _to_attr(value: Any):
    if isinstance(value, dict):
        return AttrDict({k: _to_attr(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr(v) for v in value]
    return value


def _coerce_scalar(raw: str):
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _apply_dotlist(cfg: Dict[str, Any], dotlist):
    # Apply CLI overrides like `train.lr=1e-3` into nested config dictionaries
    for item in dotlist:
        if "=" not in item:
            continue
        key, raw_val = item.split("=", 1)
        value = _coerce_scalar(raw_val)
        parts = key.split(".")
        cur = cfg
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
    return cfg


def _load_config(path: str, overrides):
    # Prefer OmegaConf when available; otherwise use a minimal YAML + dotlist fallback
    if OmegaConf is not None:
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cfg, OmegaConf.to_container(cfg, resolve=True)

    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict = _apply_dotlist(cfg_dict, overrides)
    return _to_attr(cfg_dict), cfg_dict


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
    cfg, cfg_resolved = _load_config(args.config, args.overrides)

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

    start_epoch, best_metric = resume_if_available(str(cfg.train.get("resume", "")), model, optimizer, scheduler, scaler)

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
        observed_level_losses = {}
        for key, value in train_metrics.items():
            if not key.startswith("loss_level_"):
                continue
            suffix = key[len("loss_level_") :]
            if suffix.isdigit():
                observed_level_losses[int(suffix)] = float(value)
        for level_idx in observed_level_losses:
            if level_idx not in level_loss_history:
                level_loss_history[level_idx] = [float("nan")] * (len(epoch_ids) - 1)
        for level_idx, series in level_loss_history.items():
            series.append(observed_level_losses.get(level_idx, float("nan")))

        # Collect per-level validation accuracies for plotting.
        observed_level_accs = {}
        for key, value in val_metrics.items():
            if not key.startswith("acc_level_"):
                continue
            suffix = key[len("acc_level_") :]
            if suffix.isdigit():
                observed_level_accs[int(suffix)] = float(value)
        for level_idx in observed_level_accs:
            if level_idx not in level_val_acc_history:
                level_val_acc_history[level_idx] = [float("nan")] * (len(epoch_ids) - 1)
        for level_idx, series in level_val_acc_history.items():
            series.append(observed_level_accs.get(level_idx, float("nan")))

        score = metric_for_best(val_metrics)
        if scheduler is not None:
            step_scheduler(scheduler, epoch=epoch, metric=score)

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
    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device)

        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise KeyError(
                f"Could not find model weights inside checkpoint keys: {list(checkpoint.keys())}"
            )

        print(f"[test] loaded best checkpoint from: {best_ckpt}")
    else:
        print(f"[test] warning: best checkpoint not found at {best_ckpt}, using last model in memory")

    test_metrics = evaluate(model, test_loader, device, cfg, taxonomy)
    print(f"[test] {pretty_metrics(test_metrics, level_names=level_names)}")


if __name__ == "__main__":
    main()
