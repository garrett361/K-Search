"""ReferenceImpl protocol."""

from typing import Any, Protocol


class ReferenceImpl(Protocol):
    """Reference implementation for generating ground truth."""

    def run(self, input_data: Any) -> Any:
        """Execute reference implementation on input data."""
        ...
