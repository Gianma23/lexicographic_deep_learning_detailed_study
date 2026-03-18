from typing import Any, Dict, Iterable, Tuple

import yaml

try:
    from omegaconf import OmegaConf  # type: ignore
except ImportError:  # pragma: no cover
    OmegaConf = None


class AttrDict(dict):
    """Lightweight dict wrapper for attribute-style access."""

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


def _apply_dotlist(cfg: Dict[str, Any], dotlist: Iterable[str]):
    """Apply CLI overrides like `train.lr=1e-3` into nested config dictionaries."""
    for item in dotlist:
        if "=" not in item:
            continue
        key, raw_val = item.split("=", 1)
        value = _coerce_scalar(raw_val)
        parts = key.split(".")
        cur = cfg
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value
    return cfg


def load_config(path: str, overrides: Iterable[str]) -> Tuple[Any, Dict[str, Any]]:
    """Load config with optional dotlist overrides, using OmegaConf when available."""
    if OmegaConf is not None:
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
        return cfg, OmegaConf.to_container(cfg, resolve=True)

    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict = _apply_dotlist(cfg_dict, overrides)
    return _to_attr(cfg_dict), cfg_dict

