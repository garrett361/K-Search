"""Modular framework: protocol-based abstractions for code optimization tasks."""

from k_search.modular.config import ArtifactConfig, MetricsConfig, SearchConfig, SearchResult
from k_search.modular.loop import LLMCall, run_search
from k_search.modular.results import AnalysisResult, CheckResult
from k_search.modular.round import Round

from k_search.modular.protocols import (
    Analyzer,
    ArtifactStore,
    CorrectnessChecker,
    EvaluationResult,
    Evaluator,
    FeedbackProvider,
    Implementation,
    InputGenerator,
    MetricsTracker,
    ReferenceImpl,
    Scorer,
    TaskDefinition,
)
from k_search.modular.adapters import (
    GpuModeEvaluationResult,
    GpuModeEvaluator,
    GpuModeImplementation,
    GpuModeTaskDefinition,
)
from k_search.modular.metrics import NoOpMetricsTracker
from k_search.modular.artifacts import NoOpArtifactStore

__all__ = [
    # Config
    "ArtifactConfig",
    "MetricsConfig",
    "SearchConfig",
    "SearchResult",
    # Loop
    "LLMCall",
    "run_search",
    # Types
    "AnalysisResult",
    "CheckResult",
    "Round",
    # Protocols
    "Analyzer",
    "ArtifactStore",
    "CorrectnessChecker",
    "EvaluationResult",
    "Evaluator",
    "FeedbackProvider",
    "Implementation",
    "InputGenerator",
    "MetricsTracker",
    "ReferenceImpl",
    "Scorer",
    "TaskDefinition",
    # Adapters
    "GpuModeEvaluationResult",
    "GpuModeEvaluator",
    "GpuModeImplementation",
    "GpuModeTaskDefinition",
    # Default implementations
    "NoOpMetricsTracker",
    "NoOpArtifactStore",
]
