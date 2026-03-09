# Wandb Artifacts Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add artifact persistence to V2 search loop via ArtifactStore protocol.

**Architecture:** Protocol injection with no-op default. `Implementation.artifact_dir()` context manager materializes files; ArtifactStore copies/uploads them.

**Tech Stack:** Python dataclasses, typing.Protocol, pathlib, shutil, wandb (optional).

**Design doc:** `docs/plans/2026-03-04-wandb-integration-design.md`

**Depends on:** PR1 (04a-wandb-metrics-integration.md)

---

## Task 1: Add artifact_dir to Implementation protocol and GpuModeImplementation

**Files:**
- Modify: `k_search/modular/protocols/results.py`
- Modify: `k_search/modular/adapters/wrappers.py`
- Test: `tests/modular/test_wrappers.py`

**Step 1: Write the failing test**

Add to `tests/modular/test_wrappers.py`:

```python
from pathlib import Path


class TestGpuModeImplementationArtifactDir:
    def test_yields_directory_with_source_files(self):
        from k_search.modular.adapters.wrappers import GpuModeImplementation
        from k_search.tasks.task_base import Solution, BuildSpec, SourceFile, SupportedLanguages

        solution = Solution(
            name="test",
            definition="test_task",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=[],
                entry_point="kernel.py::custom_kernel",
            ),
            sources=[
                SourceFile(path="kernel.py", content="def custom_kernel(): pass"),
                SourceFile(path="utils.py", content="# utils"),
            ],
        )
        impl = GpuModeImplementation(solution)

        with impl.artifact_dir() as src_dir:
            assert src_dir is not None
            assert (src_dir / "kernel.py").read_text() == "def custom_kernel(): pass"
            assert (src_dir / "utils.py").read_text() == "# utils"

    def test_yields_none_when_no_sources(self):
        from k_search.modular.adapters.wrappers import GpuModeImplementation
        from k_search.tasks.task_base import Solution, BuildSpec, SupportedLanguages

        solution = Solution(
            name="test",
            definition="test_task",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=[],
                entry_point="kernel.py::custom_kernel",
            ),
            sources=[],
        )
        impl = GpuModeImplementation(solution)

        with impl.artifact_dir() as src_dir:
            assert src_dir is None

    def test_cleans_up_temp_dir_after_context(self):
        from k_search.modular.adapters.wrappers import GpuModeImplementation
        from k_search.tasks.task_base import Solution, BuildSpec, SourceFile, SupportedLanguages

        solution = Solution(
            name="test",
            definition="test_task",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=[],
                entry_point="kernel.py::custom_kernel",
            ),
            sources=[SourceFile(path="kernel.py", content="code")],
        )
        impl = GpuModeImplementation(solution)

        with impl.artifact_dir() as src_dir:
            dir_path = src_dir

        assert not dir_path.exists()
```

**Step 2: Run test to verify it fails**

Run: `cd K-Search && python -m pytest tests/modular/test_wrappers.py::TestGpuModeImplementationArtifactDir -v`
Expected: FAIL with "no attribute 'artifact_dir'"

**Step 3: Write implementation**

Update `k_search/modular/protocols/results.py` - add to Implementation protocol:

```python
from collections.abc import Iterator
from contextlib import contextmanager


class Implementation(Protocol):
    name: str
    content: Any

    @contextmanager
    def artifact_dir(self) -> Iterator[Path | None]:
        """Yield directory containing files for artifact storage, or None.

        Files may exist only in memory (e.g., from LLM output). This context
        manager materializes them to a temp directory for the duration of the
        context, allowing artifact stores to copy/upload without knowing
        whether files were in-memory or already on disk.

        Yields:
            Path to directory containing files, or None if no files.
        """
        yield None
```

Update `k_search/modular/adapters/wrappers.py` - add to GpuModeImplementation:

```python
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class GpuModeImplementation:
    def __init__(self, inner: Solution) -> None:
        self.inner = inner
        self.name = inner.name
        self.content = inner

    @contextmanager
    def artifact_dir(self) -> Iterator[Path | None]:
        """Materialize Solution sources to temp directory."""
        sources = {sf.path: sf.content for sf in self.inner.sources}
        if not sources:
            yield None
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = Path(tmpdir)
            for rel_path, content in sources.items():
                path = tmpdir_p / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            yield tmpdir_p
```

**Step 4: Run test to verify it passes**

Run: `cd K-Search && python -m pytest tests/modular/test_wrappers.py::TestGpuModeImplementationArtifactDir -v`
Expected: PASS

**Step 5: Commit**

```bash
cd K-Search && git add k_search/modular/ tests/modular/
git commit -m "feat(modular): add artifact_dir to Implementation protocol"
```

---

## Task 2: ArtifactConfig and ArtifactStore protocol

**Files:**
- Modify: `k_search/modular/config.py`
- Create: `k_search/modular/artifacts/__init__.py`
- Create: `k_search/modular/artifacts/protocol.py`
- Create: `k_search/modular/artifacts/noop.py`
- Test: `tests/modular/artifacts/__init__.py`
- Test: `tests/modular/artifacts/test_stores.py`

**Step 1: Write the failing test**

Create `tests/modular/artifacts/__init__.py` (empty).

Create `tests/modular/artifacts/test_stores.py`:

```python
"""Tests for ArtifactStore implementations."""

import json
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Iterator
from unittest.mock import MagicMock, Mock, patch

import pytest

from k_search.modular.config import ArtifactConfig
from k_search.modular.types import Round


@contextmanager
def mock_artifact_dir(files: dict[str, str]) -> Iterator[Path | None]:
    """Helper to create mock artifact_dir context manager."""
    if not files:
        yield None
        return
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        for name, content in files.items():
            path = tmpdir_p / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        yield tmpdir_p


def make_outcome_mock(
    succeeded: bool = True,
    name: str = "test_impl",
    metrics: dict | None = None,
    files: dict[str, str] | None = None,
) -> Round:
    impl = Mock()
    impl.name = name
    impl.artifact_dir = lambda: mock_artifact_dir(files or {"kernel.py": "# code"})

    result = Mock()
    result.succeeded.return_value = succeeded
    result.get_metrics.return_value = metrics or {"latency_ms": 10.0}

    return Round(impl=impl, result=result)


class TestArtifactConfig:
    def test_string_to_path_coercion(self):
        config = ArtifactConfig(output_dir="/tmp/test")
        assert isinstance(config.output_dir, Path)


class TestNoOpArtifactStore:
    def test_store_does_not_raise(self):
        from k_search.modular.artifacts.noop import NoOpArtifactStore

        store = NoOpArtifactStore()
        store.store(make_outcome_mock(), round_idx=0)
```

**Step 2: Run test to verify it fails**

Run: `cd K-Search && python -m pytest tests/modular/artifacts/test_stores.py -v`
Expected: FAIL

**Step 3: Write implementation**

Add to `k_search/modular/config.py`:

```python
from pathlib import Path


@dataclass
class ArtifactConfig:
    """Configuration for artifact storage."""

    output_dir: Path | str | None = None
    only_store_successes: bool = True
    wandb: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
```

Create `k_search/modular/artifacts/protocol.py`:

```python
"""ArtifactStore protocol definition."""

from typing import Protocol, runtime_checkable

from k_search.modular.types import Round


@runtime_checkable
class ArtifactStore(Protocol):
    """Protocol for storing artifacts during search."""

    def store(self, outcome: Round, round_idx: int) -> None: ...
```

Create `k_search/modular/artifacts/noop.py`:

```python
"""No-op artifact store implementation."""

from k_search.modular.types import Round


class NoOpArtifactStore:
    """Artifact store that does nothing."""

    def store(self, outcome: Round, round_idx: int) -> None:
        pass
```

Create `k_search/modular/artifacts/__init__.py`:

```python
"""Artifact storage for modular."""

from k_search.modular.artifacts.noop import NoOpArtifactStore
from k_search.modular.artifacts.protocol import ArtifactStore

__all__ = ["ArtifactStore", "NoOpArtifactStore"]
```

**Step 4: Run test to verify it passes**

Run: `cd K-Search && python -m pytest tests/modular/artifacts/test_stores.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd K-Search && git add k_search/modular/ tests/modular/artifacts/
git commit -m "feat(modular): add ArtifactConfig, ArtifactStore protocol, NoOpArtifactStore"
```

---

## Task 3: LocalArtifactStore

**Files:**
- Create: `k_search/modular/artifacts/local.py`
- Modify: `k_search/modular/artifacts/__init__.py`
- Test: `tests/modular/artifacts/test_stores.py`

**Step 1: Write the failing test**

Add to `tests/modular/artifacts/test_stores.py`:

```python
class TestLocalArtifactStore:
    def test_copies_files_to_code_dir(self, tmp_path):
        from k_search.modular.artifacts.local import LocalArtifactStore

        config = ArtifactConfig(output_dir=tmp_path, only_store_successes=False)
        store = LocalArtifactStore(config)
        outcome = make_outcome_mock(files={"kernel.py": "def kernel(): pass"})

        store.store(outcome, round_idx=0)

        assert (tmp_path / "round_0" / "code" / "kernel.py").read_text() == "def kernel(): pass"

    def test_writes_metadata_json(self, tmp_path):
        from k_search.modular.artifacts.local import LocalArtifactStore

        config = ArtifactConfig(output_dir=tmp_path, only_store_successes=False)
        store = LocalArtifactStore(config)
        outcome = make_outcome_mock(
            name="my_impl",
            metrics={"latency_ms": 5.5, "speedup_factor": 2.0},
        )

        store.store(outcome, round_idx=3)

        metadata = json.loads((tmp_path / "round_3" / "metadata.json").read_text())
        assert metadata["name"] == "my_impl"
        assert metadata["latency_ms"] == 5.5

    def test_only_store_successes_skips_failures(self, tmp_path):
        from k_search.modular.artifacts.local import LocalArtifactStore

        config = ArtifactConfig(output_dir=tmp_path, only_store_successes=True)
        store = LocalArtifactStore(config)

        store.store(make_outcome_mock(succeeded=False), round_idx=0)
        assert not (tmp_path / "round_0").exists()

        store.store(make_outcome_mock(succeeded=True), round_idx=1)
        assert (tmp_path / "round_1").is_dir()

    def test_handles_nested_file_paths(self, tmp_path):
        from k_search.modular.artifacts.local import LocalArtifactStore

        config = ArtifactConfig(output_dir=tmp_path, only_store_successes=False)
        store = LocalArtifactStore(config)
        outcome = make_outcome_mock(files={"src/utils/helper.py": "# helper"})

        store.store(outcome, round_idx=0)

        assert (tmp_path / "round_0" / "code" / "src" / "utils" / "helper.py").read_text() == "# helper"
```

**Step 2: Run test to verify it fails**

Run: `cd K-Search && python -m pytest tests/modular/artifacts/test_stores.py::TestLocalArtifactStore -v`
Expected: FAIL

**Step 3: Write implementation**

Create `k_search/modular/artifacts/local.py`:

```python
"""Local filesystem artifact store implementation."""

import json
import shutil
from pathlib import Path

from k_search.modular.config import ArtifactConfig
from k_search.modular.types import Round


class LocalArtifactStore:
    """Artifact store that writes to local filesystem."""

    def __init__(self, config: ArtifactConfig) -> None:
        if config.output_dir is None:
            raise ValueError("output_dir required for LocalArtifactStore")
        self._output_dir = Path(config.output_dir)
        self._only_store_successes = config.only_store_successes

    def store(self, outcome: Round, round_idx: int) -> None:
        if self._only_store_successes and not outcome.result.succeeded():
            return

        round_dir = self._output_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Copy code files
        code_dir = round_dir / "code"
        with outcome.impl.artifact_dir() as src_dir:
            if src_dir:
                shutil.copytree(src_dir, code_dir)

        # Write metadata
        metadata = {
            "name": outcome.impl.name,
            "succeeded": outcome.result.succeeded(),
            **outcome.result.get_metrics(),
        }
        (round_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
```

Update `k_search/modular/artifacts/__init__.py`:

```python
"""Artifact storage for modular."""

from k_search.modular.artifacts.local import LocalArtifactStore
from k_search.modular.artifacts.noop import NoOpArtifactStore
from k_search.modular.artifacts.protocol import ArtifactStore

__all__ = ["ArtifactStore", "NoOpArtifactStore", "LocalArtifactStore"]
```

**Step 4: Run test to verify it passes**

Run: `cd K-Search && python -m pytest tests/modular/artifacts/test_stores.py::TestLocalArtifactStore -v`
Expected: PASS

**Step 5: Commit**

```bash
cd K-Search && git add k_search/modular/artifacts/
git commit -m "feat(modular): add LocalArtifactStore"
```

---

## Task 4: WandbArtifactStore

**Files:**
- Create: `k_search/modular/artifacts/wandb.py`
- Modify: `k_search/modular/artifacts/__init__.py`
- Test: `tests/modular/artifacts/test_stores.py`

**Step 1: Write the failing test**

Add to `tests/modular/artifacts/test_stores.py`:

```python
class TestWandbArtifactStore:
    def test_raises_if_no_active_run(self):
        from k_search.modular.artifacts.wandb import WandbArtifactStore

        with patch("k_search.modular.artifacts.wandb.wandb") as mock_wandb:
            mock_wandb.run = None
            with pytest.raises(RuntimeError, match="no active run"):
                WandbArtifactStore(ArtifactConfig(wandb=True))

    def test_creates_artifact_with_code_files(self):
        from k_search.modular.artifacts.wandb import WandbArtifactStore

        with patch("k_search.modular.artifacts.wandb.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            mock_wandb.run.id = "test_run_123"
            mock_artifact = MagicMock()
            mock_wandb.Artifact.return_value = mock_artifact

            config = ArtifactConfig(wandb=True, only_store_successes=False)
            store = WandbArtifactStore(config)
            outcome = make_outcome_mock(files={"kernel.py": "# code"})

            store.store(outcome, round_idx=3)

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

            store.store(make_outcome_mock(succeeded=False), round_idx=0)
            mock_wandb.Artifact.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `cd K-Search && python -m pytest tests/modular/artifacts/test_stores.py::TestWandbArtifactStore -v`
Expected: FAIL

**Step 3: Write implementation**

Create `k_search/modular/artifacts/wandb.py`:

```python
"""Wandb artifact store implementation."""

import json
import tempfile
from pathlib import Path

from k_search.modular.config import ArtifactConfig
from k_search.modular.types import Round


class WandbArtifactStore:
    """Artifact store that uploads to Weights & Biases."""

    def __init__(self, config: ArtifactConfig) -> None:
        try:
            import wandb as _wandb
        except ImportError:
            raise RuntimeError("wandb configured but not installed")

        if _wandb.run is None:
            raise RuntimeError(
                "wandb configured but no active run (call wandb.init() first)"
            )

        self._wandb = _wandb
        self._run_id = _wandb.run.id
        self._only_store_successes = config.only_store_successes

    def store(self, outcome: Round, round_idx: int) -> None:
        if self._only_store_successes and not outcome.result.succeeded():
            return

        metadata = {
            "name": outcome.impl.name,
            "round_idx": round_idx,
            "succeeded": outcome.result.succeeded(),
            **outcome.result.get_metrics(),
        }

        artifact = self._wandb.Artifact(
            name=f"{self._run_id}_r{round_idx}_code",
            type="generated-code",
            metadata=metadata,
        )

        # Add code files
        with outcome.impl.artifact_dir() as src_dir:
            if src_dir:
                for file_path in src_dir.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(src_dir)
                        artifact.add_file(str(file_path), name=f"code/{rel}")

        # Add metadata file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metadata, f, indent=2)
            f.flush()
            artifact.add_file(f.name, name="metadata.json")

        self._wandb.log_artifact(artifact)
```

Update `k_search/modular/artifacts/__init__.py`:

```python
"""Artifact storage for modular."""

from k_search.modular.artifacts.local import LocalArtifactStore
from k_search.modular.artifacts.noop import NoOpArtifactStore
from k_search.modular.artifacts.protocol import ArtifactStore
from k_search.modular.artifacts.wandb import WandbArtifactStore

__all__ = ["ArtifactStore", "NoOpArtifactStore", "LocalArtifactStore", "WandbArtifactStore"]
```

**Step 4: Run test to verify it passes**

Run: `cd K-Search && python -m pytest tests/modular/artifacts/test_stores.py::TestWandbArtifactStore -v`
Expected: PASS

**Step 5: Commit**

```bash
cd K-Search && git add k_search/modular/artifacts/
git commit -m "feat(modular): add WandbArtifactStore"
```

---

## Task 5: create_artifact_stores factory and integrate into run_search

**Files:**
- Modify: `k_search/modular/artifacts/__init__.py`
- Modify: `k_search/modular/loop.py`
- Modify: `k_search/modular/__init__.py`
- Test: `tests/modular/artifacts/test_stores.py`
- Test: `tests/modular/test_loop.py`

**Step 1: Write the failing tests**

Add to `tests/modular/artifacts/test_stores.py`:

```python
class TestCreateArtifactStores:
    def test_returns_noop_by_default(self):
        from k_search.modular.artifacts import create_artifact_stores

        stores = create_artifact_stores()
        assert len(stores) == 1
        assert isinstance(stores[0], NoOpArtifactStore)

    def test_returns_local_when_output_dir_set(self, tmp_path):
        from k_search.modular.artifacts import create_artifact_stores

        stores = create_artifact_stores(ArtifactConfig(output_dir=tmp_path))
        assert len(stores) == 1
        assert isinstance(stores[0], LocalArtifactStore)

    def test_returns_both_when_both_configured(self, tmp_path):
        from k_search.modular.artifacts import create_artifact_stores
        from k_search.modular.artifacts.wandb import WandbArtifactStore

        with patch("k_search.modular.artifacts.wandb.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            mock_wandb.run.id = "test"
            stores = create_artifact_stores(ArtifactConfig(output_dir=tmp_path, wandb=True))
            assert len(stores) == 2
            assert isinstance(stores[0], LocalArtifactStore)
            assert isinstance(stores[1], WandbArtifactStore)
```

Add to `tests/modular/test_loop.py`:

```python
class TestRunSearchWithArtifacts:
    def test_calls_store_each_round(self):
        from unittest.mock import Mock

        task = make_task_mock()
        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=3)
        store = Mock()

        run_search(task, evaluator, stub_llm, config, artifact_stores=store)

        assert store.store.call_count == 3

    def test_accepts_list_of_stores(self):
        from unittest.mock import Mock

        task = make_task_mock()
        evaluator = make_evaluator_mock()
        config = SearchConfig(max_rounds=1)
        s1, s2 = Mock(), Mock()

        run_search(task, evaluator, stub_llm, config, artifact_stores=[s1, s2])

        assert s1.store.call_count == 1
        assert s2.store.call_count == 1
```

**Step 2: Run tests to verify they fail**

Run: `cd K-Search && python -m pytest tests/modular/artifacts/test_stores.py::TestCreateArtifactStores tests/modular/test_loop.py::TestRunSearchWithArtifacts -v`
Expected: FAIL

**Step 3: Write implementation**

Update `k_search/modular/artifacts/__init__.py`:

```python
"""Artifact storage for modular."""

from k_search.modular.artifacts.local import LocalArtifactStore
from k_search.modular.artifacts.noop import NoOpArtifactStore
from k_search.modular.artifacts.protocol import ArtifactStore
from k_search.modular.artifacts.wandb import WandbArtifactStore
from k_search.modular.config import ArtifactConfig


def create_artifact_stores(config: ArtifactConfig | None = None) -> list[ArtifactStore]:
    config = config or ArtifactConfig()
    stores: list[ArtifactStore] = []

    if config.output_dir:
        stores.append(LocalArtifactStore(config))

    if config.wandb:
        stores.append(WandbArtifactStore(config))

    return stores or [NoOpArtifactStore()]


__all__ = [
    "ArtifactStore",
    "NoOpArtifactStore",
    "LocalArtifactStore",
    "WandbArtifactStore",
    "create_artifact_stores",
]
```

Update `k_search/modular/loop.py` - add import and parameter:

```python
from k_search.modular.artifacts import ArtifactStore, NoOpArtifactStore
```

Update signature:

```python
def run_search(
    task: TaskDefinition,
    evaluator: Evaluator,
    llm: LLMCall,
    config: SearchConfig,
    metrics_config: MetricsConfig | None = None,
    metrics_trackers: MetricsTracker | list[MetricsTracker] | None = None,
    artifact_stores: ArtifactStore | list[ArtifactStore] | None = None,
) -> SearchResult:
```

Add normalization after metrics normalization:

```python
    if artifact_stores is None:
        artifact_stores = [NoOpArtifactStore()]
    elif not isinstance(artifact_stores, list):
        artifact_stores = [artifact_stores]
```

Add after metrics logging in the loop (create outcome first):

```python
        outcome = Round(impl=impl, result=result)

        # ... metrics logging ...

        for store in artifact_stores:
            store.store(outcome, round_idx)
```

Update `k_search/modular/__init__.py`:

```python
from k_search.modular.config import ArtifactConfig, MetricsConfig, SearchConfig, SearchResult

__all__ = ["run_search", "SearchConfig", "SearchResult", "MetricsConfig", "ArtifactConfig"]
```

**Step 4: Run tests to verify they pass**

Run: `cd K-Search && python -m pytest tests/modular/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
cd K-Search && git add k_search/modular/ tests/modular/
git commit -m "feat(modular): add create_artifact_stores factory and integrate into run_search"
```

---

## Validation

```bash
cd K-Search && python -m pytest tests/modular/ tests/modular/ -v
cd K-Search && ruff check k_search/modular/ k_search/modular/
cd K-Search && python -c "
from k_search.modular import run_search, SearchConfig, MetricsConfig, ArtifactConfig
from k_search.modular.metrics import create_metrics_trackers
from k_search.modular.artifacts import create_artifact_stores
print('PR2 complete!')
"
```
