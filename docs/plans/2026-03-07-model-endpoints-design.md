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
4. If not found, `ENDPOINTS = {}` (empty dict)

## Module: `k_search/modular/llm.py`

```python
"""LLM endpoint configuration."""

import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

DEFAULT_FILENAME = "model_endpoints.json"
ENV_VAR = "MODEL_ENDPOINTS_FILE"


def _find_config_file() -> Path | None:
    if env_path := os.environ.get(ENV_VAR):
        return Path(env_path)

    home = Path.home()
    current = Path.cwd()

    while True:
        candidate = current / DEFAULT_FILENAME
        if candidate.is_file():
            return candidate
        if current == home or current == current.parent:
            break
        current = current.parent

    return None


def _load_endpoints() -> dict[str, str]:
    config_path = _find_config_file()
    if config_path is None:
        return {}
    return json.loads(config_path.read_text())


# MappingProxyType prevents accidental mutation of shared config
ENDPOINTS: Mapping[str, str] = MappingProxyType(_load_endpoints())

# Extension: could support {"model": {"base_url": ..., "headers_from_env": ...}}
# for complex configs. For now, simple model -> base_url strings only.


def get_endpoint(model_name: str) -> str:
    """Get base_url for model. Raises KeyError with helpful message."""
    try:
        return ENDPOINTS[model_name]
    except KeyError:
        raise KeyError(
            f"Model '{model_name}' not configured. "
            f"Add it to {DEFAULT_FILENAME} or set {ENV_VAR} env var."
        ) from None
```

## Git Ignore

Add to `.gitignore`:

```
model_endpoints.json
```

## Usage

```python
from k_search.modular.llm import get_endpoint, ENDPOINTS

# Recommended - helpful error messages
base_url = get_endpoint("gpt-5")

# Raw access for iteration/membership checks
if "gemini-3-pro" in ENDPOINTS:
    ...
```

## Unit Tests

Test the discovery and loading logic in `tests/modular/test_llm.py`:

```python
"""Tests for LLM endpoint configuration."""

import json
import os
import pytest
from pathlib import Path


class TestEndpointDiscovery:
    def test_load_from_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config_file = tmp_path / "custom_endpoints.json"
        config_file.write_text(json.dumps({"test-model": "https://test.example.com/v1"}))

        monkeypatch.setenv("MODEL_ENDPOINTS_FILE", str(config_file))
        monkeypatch.chdir(tmp_path)

        from importlib import reload
        import k_search.modular.llm as llm_module
        reload(llm_module)

        assert llm_module.ENDPOINTS["test-model"] == "https://test.example.com/v1"

    def test_load_from_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config_file = tmp_path / "model_endpoints.json"
        config_file.write_text(json.dumps({"local-model": "https://local.example.com/v1"}))

        monkeypatch.delenv("MODEL_ENDPOINTS_FILE", raising=False)
        monkeypatch.chdir(tmp_path)

        from importlib import reload
        import k_search.modular.llm as llm_module
        reload(llm_module)

        assert llm_module.ENDPOINTS["local-model"] == "https://local.example.com/v1"

    def test_empty_when_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MODEL_ENDPOINTS_FILE", raising=False)
        monkeypatch.chdir(tmp_path)

        from importlib import reload
        import k_search.modular.llm as llm_module
        reload(llm_module)

        assert llm_module.ENDPOINTS == {}
```

## Test Migration

Update tests to use endpoints config instead of hardcoded RITS env vars:

```python
from k_search.modular.llm import ENDPOINTS, get_endpoint

@pytest.mark.skipif(
    not ENDPOINTS or not os.getenv("RITS_API_KEY"),
    reason="No endpoints configured or RITS_API_KEY not set",
)
class TestLLMAPIIntegration:
    @pytest.fixture
    def model_name(self) -> str:
        return next(iter(ENDPOINTS.keys()))

    @pytest.fixture
    def client(self, model_name: str) -> openai.OpenAI:
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
