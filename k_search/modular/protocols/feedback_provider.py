"""FeedbackProvider protocol."""

from typing import Any, Protocol

from k_search.modular.world.round import Round


class FeedbackProvider(Protocol):
    """Routes evaluation feedback to different LLM consumers."""

    def for_codegen(self, rounds: Round | list[Round]) -> str:
        """Format rounds as feedback for codegen LLM."""
        ...

    def for_world_model(self, rounds: Round | list[Round]) -> list[dict[str, Any]]:
        """Format rounds for world model. Returns one dict per round."""
        ...
