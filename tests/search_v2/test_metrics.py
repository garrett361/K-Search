"""Tests for search_v2 metrics module."""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from k_search.search_v2.config import MetricsConfig
from k_search.search_v2.loop import _build_round_metrics
from k_search.search_v2.metrics import create_metrics_trackers
from k_search.search_v2.metrics.noop import NoOpMetricsTracker
from k_search.search_v2.metrics.wandb import WandbMetricsTracker


class TestWandbMetricsTracker:
    def test_raises_when_wandb_not_installed(self):
        with patch.dict(sys.modules, {"wandb": None}):
            with pytest.raises(RuntimeError, match="wandb configured but not installed"):
                WandbMetricsTracker(MetricsConfig(wandb=True))

    def test_raises_when_no_active_run(self):
        mock_wandb = MagicMock()
        mock_wandb.run = None

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            with pytest.raises(RuntimeError, match="no active run"):
                WandbMetricsTracker(MetricsConfig(wandb=True))

    def test_log_delegates_to_wandb(self):
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            tracker = WandbMetricsTracker(MetricsConfig(wandb=True))
            tracker.log({"score": 0.5}, step=3)

        mock_wandb.log.assert_called_once_with({"score": 0.5}, step=3)


class TestCreateMetricsTrackers:
    def test_returns_noop_by_default(self):
        trackers = create_metrics_trackers()

        assert len(trackers) == 1
        assert isinstance(trackers[0], NoOpMetricsTracker)

    def test_returns_noop_when_wandb_false(self):
        trackers = create_metrics_trackers(MetricsConfig(wandb=False))

        assert len(trackers) == 1
        assert isinstance(trackers[0], NoOpMetricsTracker)

    def test_returns_wandb_tracker_when_wandb_true(self):
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            trackers = create_metrics_trackers(MetricsConfig(wandb=True))

        assert len(trackers) == 1
        assert isinstance(trackers[0], WandbMetricsTracker)


class TestBuildRoundMetrics:
    @pytest.fixture
    def mock_eval_result(self):
        result = Mock()
        result.is_success.return_value = True
        result.get_metrics.return_value = {"speedup_factor": 1.5}
        return result

    def test_builds_correct_structure(self, mock_eval_result):
        mock_eval_result.get_metrics.return_value = {"speedup_factor": 1.2}
        metrics = _build_round_metrics(
            round_time_secs=3.5,
            score=0.6,
            result=mock_eval_result,
            best_score=0.7,
            prompt_toks=200,
            completion_toks=100,
            cumulative_prompt_toks=800,
            cumulative_completion_toks=400,
        )

        assert metrics == {
            "round_time_secs": 3.5,
            "score": 0.6,
            "is_success": 1,
            "best_score": 0.7,
            "toks/prompt": 200,
            "toks/completion": 100,
            "toks/total": 300,
            "toks/cumulative_prompt": 800,
            "toks/cumulative_completion": 400,
            "toks/cumulative_total": 1200,
            "speedup_factor": 1.2,
        }

    def test_is_success_converts_to_int(self, mock_eval_result):
        mock_eval_result.is_success.return_value = False
        metrics = _build_round_metrics(
            round_time_secs=5.0,
            score=0.0,
            result=mock_eval_result,
            best_score=0.0,
            prompt_toks=100,
            completion_toks=50,
            cumulative_prompt_toks=100,
            cumulative_completion_toks=50,
        )

        assert metrics["is_success"] == 0

    def test_includes_numeric_eval_metrics(self, mock_eval_result):
        mock_eval_result.get_metrics.return_value = {
            "speedup_factor": 2.5,
            "latency_ms": 10.3,
            "memory_mb": 512,
        }
        metrics = _build_round_metrics(
            round_time_secs=5.0,
            score=0.5,
            result=mock_eval_result,
            best_score=0.5,
            prompt_toks=100,
            completion_toks=50,
            cumulative_prompt_toks=100,
            cumulative_completion_toks=50,
        )

        assert metrics["speedup_factor"] == 2.5
        assert metrics["latency_ms"] == 10.3
        assert metrics["memory_mb"] == 512

    def test_excludes_non_numeric_and_bool_metrics(self, mock_eval_result):
        mock_eval_result.get_metrics.return_value = {
            "speedup_factor": 1.5,
            "status": "success",
            "error_msg": None,
            "passed": True,
            "has_errors": False,
        }
        metrics = _build_round_metrics(
            round_time_secs=5.0,
            score=0.5,
            result=mock_eval_result,
            best_score=0.5,
            prompt_toks=100,
            completion_toks=50,
            cumulative_prompt_toks=100,
            cumulative_completion_toks=50,
        )

        assert metrics["speedup_factor"] == 1.5
        assert "status" not in metrics
        assert "error_msg" not in metrics
        assert "passed" not in metrics
        assert "has_errors" not in metrics

    def test_returns_expected_keys(self, mock_eval_result):
        mock_eval_result.get_metrics.return_value = {"speedup_factor": 1.2}
        metrics = _build_round_metrics(
            round_time_secs=3.5,
            score=0.6,
            result=mock_eval_result,
            best_score=0.7,
            prompt_toks=200,
            completion_toks=100,
            cumulative_prompt_toks=800,
            cumulative_completion_toks=400,
        )

        expected_keys = {
            "round_time_secs",
            "score",
            "is_success",
            "best_score",
            "toks/prompt",
            "toks/completion",
            "toks/total",
            "toks/cumulative_prompt",
            "toks/cumulative_completion",
            "toks/cumulative_total",
            "speedup_factor",
        }
        assert set(metrics.keys()) == expected_keys
