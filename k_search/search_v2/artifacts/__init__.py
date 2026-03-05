"""Artifact storage for search_v2."""

from k_search.search_v2.artifacts.local import LocalArtifactStore
from k_search.search_v2.artifacts.noop import NoOpArtifactStore
from k_search.search_v2.artifacts.protocol import ArtifactStore
from k_search.search_v2.artifacts.wandb import WandbArtifactStore
from k_search.search_v2.config import ArtifactConfig


def create_artifact_stores(config: ArtifactConfig | None = None) -> list[ArtifactStore]:
    """Create artifact stores based on configuration."""
    config = config or ArtifactConfig()
    stores: list[ArtifactStore] = []

    if config.output_dir:
        stores.append(LocalArtifactStore(config))

    if config.wandb:
        stores.append(WandbArtifactStore(config))

    return stores or [NoOpArtifactStore()]


__all__ = [
    "ArtifactStore",
    "NoOpArtifactStore",
    "LocalArtifactStore",
    "WandbArtifactStore",
    "create_artifact_stores",
]
