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
    metric_for_best,
    resume_if_available,
    save_checkpoint,
    seed_everything,
)


class AttrDict(dict):
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
    if OmegaConf is not None:
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cfg, OmegaConf.to_container(cfg, resolve=True)

    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict = _apply_dotlist(cfg_dict, overrides)
    return _to_attr(cfg_dict), cfg_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Unified hierarchical image classification training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg, cfg_resolved = _load_config(args.config, args.overrides)

    seed_everything(int(cfg.train.seed), bool(cfg.runtime.get("deterministic", True)))

    device = torch.device(str(cfg.train.get("device", "cuda" if torch.cuda.is_available() else "cpu")))

    train_loader, num_classes_per_level, taxonomy = build_dataloader(cfg, split="train")
    val_loader, _, _ = build_dataloader(cfg, split="val")

    model = build_model(cfg, num_classes_per_level, taxonomy).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    out_dir = Path(str(cfg.train.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = out_dir / "latest.pt"
    best_ckpt = out_dir / "best.pt"

    start_epoch, best_metric = resume_if_available(str(cfg.train.get("resume", "")), model, optimizer, scheduler, scaler)

    epochs = int(cfg.train.epochs)
    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, taxonomy)
        val_metrics = evaluate(model, val_loader, device, cfg, taxonomy)

        if scheduler is not None:
            scheduler.step()

        score = metric_for_best(val_metrics)
        if score > best_metric:
            best_metric = score
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

        print(f"epoch={epoch:03d} train: {pretty_metrics(train_metrics)}")
        print(f"epoch={epoch:03d} val:   {pretty_metrics(val_metrics)}")

    test_loader, _, _ = build_dataloader(cfg, split="test")
    test_metrics = evaluate(model, test_loader, device, cfg, taxonomy)
    print(f"test: {pretty_metrics(test_metrics)}")


if __name__ == "__main__":
    main()
