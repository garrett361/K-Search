"""Task framework: protocol-based abstractions for code optimization tasks."""

from k_search.task_framework.types import CheckResult, AnalysisResult, EvalOutcome
from k_search.task_framework.protocols import (
    EvaluationResult,
    Implementation,
    InputGenerator,
    ReferenceImpl,
    CorrectnessChecker,
    Scorer,
    FeedbackProvider,
    Evaluator,
    Analyzer,
    TaskDefinition,
)
from k_search.task_framework.adapters import (
    GpuModeAdapter,
    GpuModeEvaluationResult,
    GpuModeImplementation,
)

__all__ = [
    # Types
    "CheckResult",
    "AnalysisResult",
    "EvalOutcome",
    # Protocols
    "EvaluationResult",
    "Implementation",
    "InputGenerator",
    "ReferenceImpl",
    "CorrectnessChecker",
    "Scorer",
    "FeedbackProvider",
    "Evaluator",
    "Analyzer",
    "TaskDefinition",
    # Adapters
    "GpuModeAdapter",
    "GpuModeEvaluationResult",
    "GpuModeImplementation",
]
