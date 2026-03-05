"""Tests for ArtifactStore implementations."""

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from k_search.modular.config import ArtifactConfig
from k_search.modular.artifacts.noop import NoOpArtifactStore
from k_search.modular import Round


@contextmanager
def mock_artifact_dir(files: dict[str, str]) -> Iterator[Path | None]:
    """Helper to create mock artifact_dir context manager."""
    if not files:
        yield None
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        for name, content in files.items():
            path = tmpdir_p / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        yield tmpdir_p


def make_round_mock(
    is_success: bool = True,
    name: str = "test_impl",
    metrics: dict | None = None,
    files: dict[str, str] | None = None,
) -> Round:
    impl = Mock()
    impl.name = name
    impl.artifact_dir = lambda: mock_artifact_dir(files or {"kernel.py": "# code"})

    result = Mock()
    result.is_success.return_value = is_success
    result.get_metrics.return_value = metrics or {"latency_ms": 10.0}

    return Round(
        impl=impl,
        result=result,
        prompt="test",
        llm_response="test",
        prompt_tokens=0,
        completion_tokens=0,
        duration_secs=0.0,
        score=1.0 if is_success else 0.0,
    )


class TestArtifactConfig:
    def test_string_to_path_coercion(self):
        config = ArtifactConfig(output_dir="/tmp/test")
        assert isinstance(config.output_dir, Path)


class TestNoOpArtifactStore:
    def test_store_does_not_raise(self):
        store = NoOpArtifactStore()
        store.store(make_round_mock(), round_idx=0)


class TestLocalArtifactStore:
    def test_copies_files_to_code_dir(self, tmp_path):
        from k_search.modular.artifacts.local import LocalArtifactStore

        config = ArtifactConfig(output_dir=tmp_path, only_store_successes=False)
        store = LocalArtifactStore(config)
        round_ = make_round_mock(files={"kernel.py": "def kernel(): pass"})

        store.store(round_, round_idx=0)

        assert (tmp_path / "round_0" / "code" / "kernel.py").read_text() == "def kernel(): pass"

    def test_writes_metadata_json(self, tmp_path):
        from k_search.modular.artifacts.local import LocalArtifactStore

        config = ArtifactConfig(output_dir=tmp_path, only_store_successes=False)
        store = LocalArtifactStore(config)
        round_ = make_round_mock(
            name="my_impl",
            metrics={"latency_ms": 5.5, "speedup_factor": 2.0},
        )

        store.store(round_, round_idx=3)

        metadata = json.loads((tmp_path / "round_3" / "metadata.json").read_text())
        assert metadata["name"] == "my_impl"
        assert metadata["latency_ms"] == 5.5

    def test_only_store_successes_skips_failures(self, tmp_path):
        from k_search.modular.artifacts.local import LocalArtifactStore

        config = ArtifactConfig(output_dir=tmp_path, only_store_successes=True)
        store = LocalArtifactStore(config)

        store.store(make_round_mock(is_success=False), round_idx=0)
        assert not (tmp_path / "round_0").exists()

        store.store(make_round_mock(is_success=True), round_idx=1)
        assert (tmp_path / "round_1").is_dir()

    def test_handles_nested_file_paths(self, tmp_path):
        from k_search.modular.artifacts.local import LocalArtifactStore

        config = ArtifactConfig(output_dir=tmp_path, only_store_successes=False)
        store = LocalArtifactStore(config)
        round_ = make_round_mock(files={"src/utils/helper.py": "# helper"})

        store.store(round_, round_idx=0)

        assert (tmp_path / "round_0" / "code" / "src" / "utils" / "helper.py").read_text() == "# helper"


class TestWandbArtifactStore:
    def test_creates_artifact_with_code_files(self):
        from k_search.modular.artifacts.wandb import WandbArtifactStore

        with patch("k_search.modular.artifacts.wandb.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            mock_wandb.run.id = "test_run_123"
            mock_artifact = MagicMock()
            mock_wandb.Artifact.return_value = mock_artifact

            config = ArtifactConfig(wandb=True, only_store_successes=False)
            store = WandbArtifactStore(config)
            round_ = make_round_mock(files={"kernel.py": "# code"})

            store.store(round_, round_idx=3)

            mock_wandb.Artifact.assert_called_once()
            assert "test_run_123" in mock_wandb.Artifact.call_args[1]["name"]
            assert mock_artifact.add_file.called
            mock_wandb.log_artifact.assert_called_once()

    def test_only_store_successes_skips_failures(self):
        from k_search.modular.artifacts.wandb import WandbArtifactStore

        with patch("k_search.modular.artifacts.wandb.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            mock_wandb.run.id = "test"

            store = WandbArtifactStore(ArtifactConfig(wandb=True, only_store_successes=True))

            store.store(make_round_mock(is_success=False), round_idx=0)
            mock_wandb.Artifact.assert_not_called()


class TestCreateArtifactStores:
    def test_returns_local_when_output_dir_set(self, tmp_path):
        from k_search.modular.artifacts import create_artifact_stores
        from k_search.modular.artifacts.local import LocalArtifactStore

        stores = create_artifact_stores(ArtifactConfig(output_dir=tmp_path))
        assert len(stores) == 1
        assert isinstance(stores[0], LocalArtifactStore)

    def test_returns_both_when_both_configured(self, tmp_path):
        from k_search.modular.artifacts import create_artifact_stores
        from k_search.modular.artifacts.local import LocalArtifactStore
        from k_search.modular.artifacts.wandb import WandbArtifactStore

        with patch("k_search.modular.artifacts.wandb.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            mock_wandb.run.id = "test"
            stores = create_artifact_stores(ArtifactConfig(output_dir=tmp_path, wandb=True))
            assert len(stores) == 2
            assert isinstance(stores[0], LocalArtifactStore)
            assert isinstance(stores[1], WandbArtifactStore)
