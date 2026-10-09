"""Central config loading so no module hardcodes paths or parameters."""
from pathlib import Path
from typing import Any, Dict, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """Load the YAML config. Relative paths are resolved from the project root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative: Union[str, Path]) -> Path:
    """Resolve a config path against the project root and create it if missing."""
    p = Path(relative)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p
