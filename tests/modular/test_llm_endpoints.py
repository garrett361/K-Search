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
