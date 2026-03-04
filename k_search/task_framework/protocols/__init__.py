"""Protocol definitions for task framework."""

from k_search.task_framework.protocols.results import EvaluationResult, SolutionArtifact
from k_search.task_framework.protocols.input_generator import InputGenerator
from k_search.task_framework.protocols.reference_impl import ReferenceImpl
from k_search.task_framework.protocols.correctness import CorrectnessChecker
from k_search.task_framework.protocols.scorer import Scorer
from k_search.task_framework.protocols.feedback_provider import FeedbackProvider
from k_search.task_framework.protocols.evaluator import Evaluator
from k_search.task_framework.protocols.analyzer import Analyzer
from k_search.task_framework.protocols.task_definition import TaskDefinition

__all__ = [
    "EvaluationResult",
    "SolutionArtifact",
    "InputGenerator",
    "ReferenceImpl",
    "CorrectnessChecker",
    "Scorer",
    "FeedbackProvider",
    "Evaluator",
    "Analyzer",
    "TaskDefinition",
]
