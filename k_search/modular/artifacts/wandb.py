"""Wandb artifact store implementation."""

from __future__ import annotations

import json
import tempfile
from typing import TYPE_CHECKING
from pathlib import Path

import wandb

from k_search.modular.config import ArtifactConfig
from k_search.modular.world.round import Round

if TYPE_CHECKING:
    from k_search.modular.protocols import ArtifactStore


class WandbArtifactStore:
    """Artifact store that uploads to Weights & Biases."""

    def __init__(self, config: ArtifactConfig) -> None:
        if wandb.run is None:
            raise RuntimeError(
                "wandb configured but no active run (call wandb.init() first)"
            )

        self._run_id = wandb.run.id
        self._only_store_successes = config.only_store_successes

    def store(self, round_: Round, round_idx: int) -> None:
        if self._only_store_successes and not round_.result.is_success():
            return

        metadata = {
            "name": round_.impl.name,
            "round_idx": round_idx,
            "is_success": round_.result.is_success(),
            **round_.result.get_metrics(),
        }

        artifact = wandb.Artifact(
            name=f"{self._run_id}_r{round_idx}_code",
            type="files",
            metadata=metadata,
        )

        with round_.impl.artifact_dir() as src_dir:
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


def create_artifact_stores(config: ArtifactConfig | None = None) -> list[ArtifactStore]:
    """Create artifact stores based on configuration."""
    from k_search.modular.artifacts.local import LocalArtifactStore
    from k_search.modular.artifacts.noop import NoOpArtifactStore

    config = config or ArtifactConfig()
    stores: list[ArtifactStore] = []

    if config.output_dir:
        stores.append(LocalArtifactStore(config))

    if config.wandb:
        stores.append(WandbArtifactStore(config))

    return stores or [NoOpArtifactStore()]
