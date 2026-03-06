"""Search configuration and result types."""

import socket
import subprocess
import sys
import dataclasses
from dataclasses import dataclass
from pathlib import Path

from k_search.modular.protocols import EvaluationResult, Implementation


def collect_git_info() -> dict[str, str | bool | None]:
    """Collect K-Search git info. Returns empty dict on failure."""

    def _git(args: list[str]) -> str | None:
        try:
            r = subprocess.run(
                ["git"] + args, capture_output=True, text=True, timeout=5
            )
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    try:
        commit = _git(["rev-parse", "HEAD"])
        branch = _git(["branch", "--show-current"])
        status = _git(["status", "--porcelain"])
        if commit is None and branch is None:
            return {}
        return {
            "commit": commit,
            "branch": branch,
            "dirty": bool(status) if status is not None else None,
        }
    except Exception:
        return {}


def collect_env_info() -> dict[str, str | None]:
    """Collect runtime environment. Returns empty dict on failure."""
    try:
        result: dict[str, str | None] = {
            "hostname": socket.gethostname(),
            "python_version": sys.version.split()[0],
        }
        try:
            import torch

            result["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                result["gpu"] = torch.cuda.get_device_name()
        except ImportError:
            pass
        try:
            import triton

            result["triton_version"] = getattr(triton, "__version__", None)
        except ImportError:
            pass
        return result
    except Exception:
        return {}


def build_run_config(
    run_id: str,
    model_name: str,
    reasoning_effort: str,
    search_config: "SearchConfig",
    metrics_config: "MetricsConfig",
    artifact_config: "ArtifactConfig",
    verl_experiment_id: str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
    **task_kwargs: str,
) -> dict:
    """Build unified run config for wandb.init and local serialization."""
    return {
        "run_id": run_id,
        "verl_experiment_id": verl_experiment_id,
        "model_name": model_name,
        "reasoning_effort": reasoning_effort,
        "search": dataclasses.asdict(search_config),
        "metrics": dataclasses.asdict(metrics_config),
        "artifacts": {
            "output_dir": str(artifact_config.output_dir)
            if artifact_config.output_dir
            else None,
            "only_store_successes": artifact_config.only_store_successes,
        },
        "task_config": task_kwargs,
        "git": collect_git_info(),
        "env": collect_env_info(),
        "wandb": {
            "project": wandb_project,
            "run_name": wandb_run_name,
            "group": wandb_group,
            "tags": wandb_tags,
        },
    }


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
