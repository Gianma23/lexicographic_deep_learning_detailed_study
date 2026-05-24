import inspect
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import torch

from .common import section_to_dict


def load_trusted_checkpoint(path: Any, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a full local training checkpoint produced by this repository."""
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def _normalize_finetune_url(url: str) -> str:
    # Hugging Face "blob" URLs are HTML pages; convert them to direct file URLs.
    if "huggingface.co" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    return url


def _load_external_checkpoint(path_or_url: str) -> Tuple[Any, str]:
    parsed = urlparse(path_or_url)
    is_url = parsed.scheme in {"http", "https"}
    source = _normalize_finetune_url(path_or_url) if is_url else path_or_url
    if is_url:
        return torch.hub.load_state_dict_from_url(source, map_location="cpu", check_hash=False), source
    return torch.load(source, map_location="cpu"), source


def _extract_checkpoint_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return dict(checkpoint["model"])
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return dict(checkpoint["state_dict"])

    if isinstance(checkpoint, dict) and checkpoint:
        sample = next(iter(checkpoint.values()))
        if torch.is_tensor(sample):
            return dict(checkpoint)

    raise ValueError("Unsupported checkpoint format. Expected keys `model` or `state_dict`.")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not all(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key[len("module.") :]: value for key, value in state_dict.items()}


def _interpolate_pos_embed_if_needed(model: torch.nn.Module, checkpoint_model: Dict[str, torch.Tensor]) -> None:
    if "pos_embed" not in checkpoint_model:
        return
    if not hasattr(model, "pos_embed") or not hasattr(model, "patch_embed"):
        return

    ckpt_pos = checkpoint_model["pos_embed"]
    model_pos = getattr(model, "pos_embed")
    if not torch.is_tensor(ckpt_pos) or not torch.is_tensor(model_pos):
        return
    if ckpt_pos.shape == model_pos.shape:
        return

    try:
        num_patches = int(model.patch_embed.num_patches)
    except Exception:
        return

    num_extra_tokens = int(model_pos.shape[-2] - num_patches)
    if num_extra_tokens < 0:
        return

    orig_tokens = int(ckpt_pos.shape[-2] - num_extra_tokens)
    new_tokens = int(num_patches)
    if orig_tokens <= 0 or new_tokens <= 0:
        return

    orig_size = int(orig_tokens**0.5)
    new_size = int(new_tokens**0.5)
    if orig_size * orig_size != orig_tokens or new_size * new_size != new_tokens:
        return

    embedding_size = ckpt_pos.shape[-1]
    extra_tokens = ckpt_pos[:, :num_extra_tokens]
    pos_tokens = ckpt_pos[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)


def load_finetune_checkpoint(cfg: Any, model: torch.nn.Module) -> bool:
    """Load a finetune checkpoint using upstream H-CAST adaptation logic."""
    model_cfg = section_to_dict(getattr(cfg, "model", None))
    train_cfg = section_to_dict(getattr(cfg, "train", None))
    finetune_path = str(model_cfg.get("finetune", train_cfg.get("finetune", "")) or "").strip()
    if not finetune_path:
        return False

    model_name = model_cfg.get("name", "")
    if not isinstance(model_name, str):
        raise ValueError("model.name must be a string.")
    if model_name == "hcast":
        inner = getattr(model, "model", None)
        if not isinstance(inner, torch.nn.Module):
            print("finetune: skipped (H-CAST timm backend unavailable)")
            return False
        target_model = inner
    else:
        target_model = model

    checkpoint, resolved_source = _load_external_checkpoint(finetune_path)
    checkpoint_model = _strip_module_prefix(_extract_checkpoint_state_dict(checkpoint))
    state_dict = target_model.state_dict()

    # Mirror upstream H-CAST key filtering for checkpoint adaptation.
    for key in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias", "cls_token"]:
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            del checkpoint_model[key]

    _interpolate_pos_embed_if_needed(target_model, checkpoint_model)
    incompat = target_model.load_state_dict(checkpoint_model, strict=False)
    missing = len(getattr(incompat, "missing_keys", []))
    unexpected = len(getattr(incompat, "unexpected_keys", []))
    print(f"finetune: loaded from {resolved_source} (missing={missing}, unexpected={unexpected})")
    return True
