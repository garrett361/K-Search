"""Result protocols for task framework."""

from typing import Any, Protocol


class EvaluationResult(Protocol):
    """Generic evaluation result."""

    def is_success(self) -> bool:
        """Return True if evaluation passed."""
        ...

    def get_metrics(self) -> dict[str, Any]:
        """Return evaluation metrics as dict."""
        ...

    def get_log(self) -> str:
        """Return evaluation log/output."""
        ...


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
