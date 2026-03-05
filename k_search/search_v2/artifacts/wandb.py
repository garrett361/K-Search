"""Wandb artifact store implementation."""

import json
import tempfile
from pathlib import Path

import wandb

from k_search.search_v2.config import ArtifactConfig
from k_search.task_framework.types import EvalOutcome


class WandbArtifactStore:
    """Artifact store that uploads to Weights & Biases."""

    def __init__(self, config: ArtifactConfig) -> None:
        if wandb.run is None:
            raise RuntimeError(
                "wandb configured but no active run (call wandb.init() first)"
            )

        self._run_id = wandb.run.id
        self._only_store_successes = config.only_store_successes

    def store(self, outcome: EvalOutcome, round_idx: int) -> None:
        if self._only_store_successes and not outcome.result.is_success():
            return

        metadata = {
            "name": outcome.impl.name,
            "round_idx": round_idx,
            "is_success": outcome.result.is_success(),
            **outcome.result.get_metrics(),
        }

        artifact = wandb.Artifact(
            name=f"{self._run_id}_r{round_idx}_code",
            type="files",
            metadata=metadata,
        )

        with outcome.impl.artifact_dir() as src_dir:
            if src_dir:
                for file_path in src_dir.rglob("*"):
                    if file_path.is_file():
                        rel = file_path.relative_to(src_dir)
                        artifact.add_file(str(file_path), name=f"code/{rel}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metadata, f, indent=2)
            f.flush()
            artifact.add_file(f.name, name="metadata.json")
            Path(f.name).unlink()

        wandb.log_artifact(artifact)
