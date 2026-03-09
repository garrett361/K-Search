# Model Endpoints Configuration

JSON-based configuration for LLM endpoint URLs, loaded at import time as a shared read-only mapping.

## File Format

`model_endpoints.json` - simple model name to base URL mapping:

```json
{
  "gpt-oss-120b": "https://rits.example.com/v1",
  "gpt-5": "https://api.openai.com/v1"
}
```

## File Discovery

1. Check `MODEL_ENDPOINTS_FILE` env var - if set, use that path directly
2. Otherwise, search for `model_endpoints.json` starting from cwd, walking up parent directories
3. Stop at home directory (`~/`)
4. If not found, raise `FileNotFoundError` with search locations listed

## Module Structure

```
k_search/modular/llm/
├── __init__.py          # Re-exports ENDPOINTS, get_endpoint
└── endpoints.py         # Discovery, loading, get_endpoint()
```

### `k_search/modular/llm/endpoints.py`

```python
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


# Private state - lazy loaded on first access
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
```

### `k_search/modular/llm/__init__.py`

```python
"""LLM configuration package."""

from k_search.modular.llm.endpoints import get_all_endpoints, get_endpoint

__all__ = ["get_all_endpoints", "get_endpoint"]
```

## Git Ignore

Add to `.gitignore`:

```
model_endpoints.json
```

## Usage

```python
from k_search.modular.llm import get_endpoint, get_all_endpoints

# Single model lookup
base_url = get_endpoint("meta-llama/Meta-Llama-3.1-70B-Instruct")

# Check available models
endpoints = get_all_endpoints()
if "gpt-4o" in endpoints:
    ...

# Iterate over all
for model, url in get_all_endpoints().items():
    ...
```

## Unit Tests

Test the discovery and loading logic in `tests/modular/test_llm_endpoints.py`:

```python
"""Tests for LLM endpoint configuration."""

import json
import pytest
from pathlib import Path

from k_search.modular.llm.endpoints import (
    _find_config_file,
    get_all_endpoints,
    get_endpoint,
)


@pytest.fixture(autouse=True)
def reset_endpoints_cache(monkeypatch: pytest.MonkeyPatch):
    """Reset the lazy-loaded cache before each test."""
    import k_search.modular.llm.endpoints as mod
    monkeypatch.setattr(mod, "_endpoints", None)
    monkeypatch.setattr(mod, "_config_path", None)


class TestFindConfigFile:
    def test_from_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config_file = tmp_path / "custom.json"
        config_file.write_text("{}")
        monkeypatch.setenv("MODEL_ENDPOINTS_FILE", str(config_file))

        assert _find_config_file() == config_file

    def test_from_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config_file = tmp_path / "model_endpoints.json"
        config_file.write_text("{}")
        monkeypatch.delenv("MODEL_ENDPOINTS_FILE", raising=False)
        monkeypatch.chdir(tmp_path)

        assert _find_config_file() == config_file

    def test_raises_when_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MODEL_ENDPOINTS_FILE", raising=False)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(FileNotFoundError, match="No model_endpoints.json found"):
            _find_config_file()

    def test_raises_when_env_var_path_missing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MODEL_ENDPOINTS_FILE", "/nonexistent/path.json")

        with pytest.raises(FileNotFoundError, match="but file does not exist"):
            _find_config_file()


class TestGetEndpoints:
    def test_returns_mapping(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config_file = tmp_path / "model_endpoints.json"
        config_file.write_text(json.dumps({"model-a": "https://a.com/v1"}))
        monkeypatch.setenv("MODEL_ENDPOINTS_FILE", str(config_file))

        endpoints = get_all_endpoints()
        assert endpoints["model-a"] == "https://a.com/v1"

    def test_get_endpoint_single(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config_file = tmp_path / "model_endpoints.json"
        config_file.write_text(json.dumps({"model-b": "https://b.com/v1"}))
        monkeypatch.setenv("MODEL_ENDPOINTS_FILE", str(config_file))

        assert get_endpoint("model-b") == "https://b.com/v1"

    def test_get_endpoint_missing_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config_file = tmp_path / "model_endpoints.json"
        config_file.write_text(json.dumps({"model-c": "https://c.com/v1"}))
        monkeypatch.setenv("MODEL_ENDPOINTS_FILE", str(config_file))

        with pytest.raises(KeyError, match="not found"):
            get_endpoint("nonexistent")
```

## Test Migration

Update tests to use endpoints config instead of hardcoded RITS env vars:

```python
import os
import pytest

def _endpoints_available() -> bool:
    """Check if endpoints config exists without raising."""
    try:
        from k_search.modular.llm import get_all_endpoints
        return bool(get_all_endpoints())
    except FileNotFoundError:
        return False

@pytest.mark.skipif(
    not _endpoints_available() or not os.getenv("RITS_API_KEY"),
    reason="No endpoints configured or RITS_API_KEY not set",
)
class TestLLMAPIIntegration:
    @pytest.fixture
    def model_name(self) -> str:
        from k_search.modular.llm import get_all_endpoints
        return next(iter(get_all_endpoints().keys()))

    @pytest.fixture
    def client(self, model_name: str) -> openai.OpenAI:
        from k_search.modular.llm import get_endpoint
        rits_key = os.environ["RITS_API_KEY"]
        return openai.OpenAI(
            base_url=get_endpoint(model_name),
            api_key=rits_key,
            default_headers={"RITS_API_KEY": rits_key},
        )
```

Files to update:
- `tests/modular/test_tool_calling_api.py`
- `tests/kernel_generators/test_generator.py`
- `tests/kernel_generators/test_world_model.py`

## Extension Story (Future)

When headers or other fields are needed, the JSON format could accept either strings (simple) or objects (extended):

```json
{
  "gpt-5": "https://api.openai.com/v1",

  "gpt-oss-120b": {
    "base_url": "https://rits.example.com/v1",
    "api_key_env": "RITS_API_KEY",
    "headers_from_env": {
      "RITS_API_KEY": "RITS_API_KEY"
    }
  }
}
```

Internal representation would normalize both forms to an `EndpointConfig` dataclass. Not implemented now - simple string format sufficient for current needs.
