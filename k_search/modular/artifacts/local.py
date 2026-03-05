"""Local filesystem artifact store implementation."""

import json
import shutil
from pathlib import Path

from k_search.modular.config import ArtifactConfig
from k_search.modular.world.round import Round


class LocalArtifactStore:
    """Artifact store that writes to local filesystem."""

    def __init__(self, config: ArtifactConfig) -> None:
        if config.output_dir is None:
            raise ValueError("output_dir required for LocalArtifactStore")
        self._output_dir = Path(config.output_dir)
        self._only_store_successes = config.only_store_successes

    def store(self, round_: Round, round_idx: int) -> None:
        if self._only_store_successes and not round_.result.is_success():
            return

        round_dir = self._output_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        code_dir = round_dir / "code"
        with round_.impl.artifact_dir() as src_dir:
            if src_dir:
                shutil.copytree(src_dir, code_dir)

        metadata = {
            "name": round_.impl.name,
            "is_success": round_.result.is_success(),
            **round_.result.get_metrics(),
        }
        (round_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
