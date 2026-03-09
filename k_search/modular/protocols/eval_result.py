"""EvaluationResult protocol definition."""

from typing import Any, Protocol


class EvaluationResult(Protocol):
    """Generic evaluation result."""

    def succeeded(self) -> bool:
        """Return True if evaluation passed."""
        ...

    def get_metrics(self) -> dict[str, Any]:
        """Return evaluation metrics as dict."""
        ...

    def get_log(self) -> str:
        """Return evaluation log/output."""
        ...
