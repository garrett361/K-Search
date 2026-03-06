"""Search configuration and result types."""

from dataclasses import dataclass
from pathlib import Path

from k_search.modular.protocols import EvaluationResult, Implementation


@dataclass
class SearchConfig:
    """Configuration for search loop."""

    max_rounds: int = 10
    timeout_secs: int | None = None


@dataclass
class MetricsConfig:
    """Configuration for metrics tracking."""

    chars_per_token: int = 4
    wandb: bool = False
    local: bool = True


@dataclass
class ArtifactConfig:
    """Configuration for artifact storage."""

    output_dir: Path | str | None = None
    only_store_successes: bool = True
    wandb: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class SearchResult:
    """Result from a search run."""

    impl: Implementation | None
    score: float
    result: EvaluationResult | None
    rounds_completed: int = 0
