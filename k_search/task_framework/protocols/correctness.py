"""CorrectnessChecker protocol."""

from typing import Any, Protocol

from k_search.task_framework.types import CheckResult


class CorrectnessChecker(Protocol):
    """Compares submission output against reference."""

    def check(self, output: Any, reference_output: Any) -> CheckResult:
        """Check correctness. reference_output may be None."""
        ...
