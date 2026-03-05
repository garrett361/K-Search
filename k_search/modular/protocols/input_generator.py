"""InputGenerator protocol."""

from typing import Any, Protocol


class InputGenerator(Protocol):
    """Generates task inputs from parameters."""

    def generate(self, params: dict[str, Any], seed: int) -> Any:
        """Generate input data for evaluation."""
        ...
