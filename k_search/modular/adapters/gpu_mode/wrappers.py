"""GPU Mode type wrappers implementing task_framework protocols."""

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from k_search.tasks.task_base import EvalResult, Solution


class GpuModeEvaluationResult:
    """Wraps EvalResult to implement EvaluationResult protocol."""

    def __init__(self, inner: EvalResult) -> None:
        self._inner = inner

    def is_success(self) -> bool:
        return self._inner.is_passed()

    def get_metrics(self) -> dict[str, Any]:
        return self._inner.to_dict(include_log_excerpt=False)

    def get_log(self) -> str:
        return self._inner.log_excerpt

    @property
    def status(self) -> str:
        return self._inner.status

    @property
    def latency_ms(self) -> float | None:
        return self._inner.latency_ms

    @property
    def speedup_factor(self) -> float | None:
        return self._inner.speedup_factor

    def is_passed(self) -> bool:
        return self._inner.is_passed()

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        return self._inner.to_dict(**kwargs)

    def score(self) -> float:
        return self._inner.score()


class GpuModeImplementation:
    """Wraps Solution to implement Implementation protocol."""

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
