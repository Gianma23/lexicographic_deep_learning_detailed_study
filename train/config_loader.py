import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from dotenv import load_dotenv
from omegaconf import OmegaConf

from .config_validation import validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str, overrides: Iterable[str]) -> Tuple[Any, Dict[str, Any]]:
    """Load the project environment and resolve a YAML config plus CLI overrides."""
    env_path = Path(os.environ.get("PROJECT_ENV_FILE", REPO_ROOT / ".env")).expanduser()
    load_dotenv(env_path, override=False)

    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    validate_config(cfg, source=str(path))
    return cfg, OmegaConf.to_container(cfg, resolve=True)
