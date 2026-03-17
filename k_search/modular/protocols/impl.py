"""Implementation protocol definition."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol


class Implementation(Protocol):
    """Data container for code to be evaluated.

    Format is task-specific:
    - str: single source file
    - dict[str, str]: multiple files {filename: content}
    - Path: reference to file on disk
    """

    name: str
    """Identifier for this implementation."""

    content: Any
    """The implementation data."""

    @contextmanager
    def artifact_dir(self) -> Iterator[Path | None]:
        """Yield directory containing files for artifact storage, or None."""
        yield None
