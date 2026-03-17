"""Tests for modular metrics module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from k_search.modular.config import MetricsConfig
from k_search.modular.metrics import create_metrics_trackers
from k_search.modular.metrics.wandb import WandbMetricsTracker


class TestWandbMetricsTracker:
    def test_raises_when_wandb_not_installed(self):
        with patch.dict(sys.modules, {"wandb": None}):
            with pytest.raises(
                RuntimeError, match="wandb configured but not installed"
            ):
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
    def test_returns_wandb_and_local_trackers_when_both_enabled(self, tmp_path):
        from k_search.modular.metrics.local import LocalMetricsTracker

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            trackers = create_metrics_trackers(
                MetricsConfig(wandb=True, local=True), output_dir=tmp_path
            )
        assert len(trackers) == 2
        assert isinstance(trackers[0], WandbMetricsTracker)
        assert isinstance(trackers[1], LocalMetricsTracker)


class TestLocalMetricsTracker:
    def test_appends_jsonl_and_writes_config(self, tmp_path):
        from k_search.modular.metrics.local import LocalMetricsTracker
        import json

        config = {"run_id": "test-123", "model": "test"}
        tracker = LocalMetricsTracker(tmp_path, run_config=config)
        tracker.log({"score": 0.5}, step=0)
        tracker.log({"score": 0.8}, step=1)

        lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["score"] == 0.5
        assert json.loads(lines[1])["score"] == 0.8

        saved_config = json.loads((tmp_path / "config.json").read_text())
        assert saved_config == config
