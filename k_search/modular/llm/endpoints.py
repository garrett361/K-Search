"""LLM endpoint configuration."""

import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

DEFAULT_FILENAME = "model_endpoints.json"
ENV_VAR = "MODEL_ENDPOINTS_FILE"


def _find_config_file() -> Path:
    """Find config file or raise FileNotFoundError with helpful message."""
    if env_path := os.environ.get(ENV_VAR):
        path = Path(env_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{ENV_VAR} set to '{env_path}' but file does not exist"
            )
        return path

    home = Path.home()
    current = Path.cwd()

    while True:
        candidate = current / DEFAULT_FILENAME
        if candidate.is_file():
            return candidate
        if current == home or current == current.parent:
            break
        current = current.parent

    raise FileNotFoundError(
        f"No {DEFAULT_FILENAME} found.\n"
        f"Search order:\n"
        f"  1. {ENV_VAR} env var (not set)\n"
        f"  2. {DEFAULT_FILENAME} in any directory from cwd to home\n"
        f"     cwd:  {Path.cwd()}\n"
        f"     home: {home}\n"
        f"Create {DEFAULT_FILENAME} in your project root or set {ENV_VAR}."
    )


_endpoints: dict[str, str] | None = None
_config_path: Path | None = None


def get_all_endpoints() -> Mapping[str, str]:
    """Get all configured endpoints as a read-only mapping.

    Raises:
        FileNotFoundError: If no config file found.
    """
    global _endpoints, _config_path
    if _endpoints is None:
        _config_path = _find_config_file()
        _endpoints = json.loads(_config_path.read_text())
    # MappingProxyType prevents accidental mutation of the shared cache
    return MappingProxyType(_endpoints)


def get_endpoint(model_name: str) -> str:
    """Get base_url for a specific model.

    Raises:
        FileNotFoundError: If no config file found.
        KeyError: If model not in config.
    """
    endpoints = get_all_endpoints()
    if model_name not in endpoints:
        available = ", ".join(endpoints.keys()) or "(file is empty)"
        raise KeyError(f"Model '{model_name}' not found. Available: {available}")
    return endpoints[model_name]
