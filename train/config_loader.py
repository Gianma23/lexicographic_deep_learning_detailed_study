import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from dotenv import load_dotenv
from omegaconf import OmegaConf

from .config_validation import validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_project_environment() -> None:
    env_path = Path(os.environ.get("PROJECT_ENV_FILE", REPO_ROOT / ".env")).expanduser()
    load_dotenv(env_path, override=False)


def load_config(path: str, overrides: Iterable[str]) -> Tuple[Any, Dict[str, Any]]:
    """Load and validate a training config plus CLI overrides."""
    _load_project_environment()
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    validate_config(cfg, source=str(path))
    return cfg, OmegaConf.to_container(cfg, resolve=True)


def load_resolved_run_config(path: str) -> Tuple[Any, Dict[str, Any]]:
    """Load an immutable resolved run artifact without current-policy validation.

    Historical ``config_resolved.yaml`` files can contain runtime provenance
    fields or protocol settings that the current training entrypoint no longer
    accepts. Checkpoint-only evaluation must reproduce those saved settings,
    rather than reject or silently rewrite them.
    """
    _load_project_environment()
    cfg = OmegaConf.load(path)
    return cfg, OmegaConf.to_container(cfg, resolve=True)
