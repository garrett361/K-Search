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


class SolutionArtifact(Protocol):
    """Generic solution container."""

    @property
    def name(self) -> str:
        """Solution identifier."""
        ...

    @property
    def content(self) -> Any:
        """Solution content (code, config, etc.)."""
        ...
