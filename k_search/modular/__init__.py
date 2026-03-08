"""Modular framework: protocol-based abstractions for code optimization tasks."""

from k_search.modular.config import (
    ArtifactConfig,
    MetricsConfig,
    SearchConfig,
    SearchResult,
)
from k_search.modular.loop import LLMCall, run_search
from k_search.modular.results import AnalysisResult, CheckResult
from k_search.modular.world import Action, Cycle, Node, Round, Tree

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
    StateFormatter,
    TaskDefinition,
    WorldModel,
)
from k_search.modular.adapters import (
    GpuModeEvaluationResult,
    GpuModeEvaluator,
    GpuModeImplementation,
    GpuModeTriMulTaskDefinition,
)
from k_search.modular.metrics import NoOpMetricsTracker
from k_search.modular.artifacts import NoOpArtifactStore
from k_search.modular.timer import Timer
from k_search.modular.span import Span

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
    # World
    "Action",
    "Cycle",
    "Node",
    "Round",
    "Tree",
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
    "StateFormatter",
    "TaskDefinition",
    "WorldModel",
    # Adapters
    "GpuModeEvaluationResult",
    "GpuModeEvaluator",
    "GpuModeImplementation",
    "GpuModeTriMulTaskDefinition",
    # Default implementations
    "NoOpMetricsTracker",
    "NoOpArtifactStore",
    # Execution context
    "Timer",
    "Span",
]
