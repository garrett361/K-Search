"""Local filesystem artifact store implementation."""

import json
import shutil
from pathlib import Path

from k_search.search_v2.config import ArtifactConfig
from k_search.task_framework.types import EvalOutcome


class LocalArtifactStore:
    """Artifact store that writes to local filesystem."""

    def __init__(self, config: ArtifactConfig) -> None:
        if config.output_dir is None:
            raise ValueError("output_dir required for LocalArtifactStore")
        self._output_dir = Path(config.output_dir)
        self._only_store_successes = config.only_store_successes

    def store(self, outcome: EvalOutcome, round_idx: int) -> None:
        if self._only_store_successes and not outcome.result.is_success():
            return

        round_dir = self._output_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        code_dir = round_dir / "code"
        with outcome.impl.artifact_dir() as src_dir:
            if src_dir:
                shutil.copytree(src_dir, code_dir)

        metadata = {
            "name": outcome.impl.name,
            "is_success": outcome.result.is_success(),
            **outcome.result.get_metrics(),
        }
        (round_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
