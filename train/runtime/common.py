from pathlib import Path
from typing import Any, Dict, Mapping


def section_to_dict(section: Any) -> Dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if hasattr(section, "items"):
        return {k: v for k, v in section.items()}
    return {}


def to_plain_data(value: Any) -> Any:
    """Convert nested mappings/sequences into plain Python containers."""
    if isinstance(value, Mapping):
        return {str(key): to_plain_data(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
