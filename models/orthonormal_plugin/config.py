from typing import Any, Dict


SECTION_KEY = "orthonormal_plugin"
INPUT_KEY = "orthonormal_plugin_scores_per_level"


def section_to_dict(section: Any) -> Dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if hasattr(section, "items"):
        return {k: v for k, v in section.items()}
    return {}


def get_section(cfg: Any, key: str) -> Dict[str, Any]:
    section = section_to_dict(getattr(cfg, key, None))
    if not section and hasattr(cfg, "get"):
        section = section_to_dict(cfg.get(key, None))
    return section


def plugin_section(cfg: Any) -> Dict[str, Any]:
    return get_section(cfg, SECTION_KEY)


def is_enabled(cfg: Any) -> bool:
    return bool(plugin_section(cfg).get("enabled", False))


def validate_disabled_mixup(cfg: Any) -> None:
    dataset_cfg = get_section(cfg, "dataset")
    transforms_cfg = section_to_dict(dataset_cfg.get("transforms", None))
    if float(transforms_cfg.get("mixup", 0.0) or 0.0) != 0.0:
        raise ValueError("orthonormal_plugin.enabled=true requires dataset.transforms.mixup=0.0.")
    if float(transforms_cfg.get("cutmix", 0.0) or 0.0) != 0.0:
        raise ValueError("orthonormal_plugin.enabled=true requires dataset.transforms.cutmix=0.0.")
