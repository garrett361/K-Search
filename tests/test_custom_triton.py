"""Unit tests for k_search.tasks.custom_triton module."""

from __future__ import annotations

import pytest
import torch

from k_search.tasks.custom_triton import (
    BenchmarkConfig,
    CorrectnessConfig,
    GeometricMeanAggregation,
    MeanAggregation,
    check_correctness,
)


class TestCorrectnessConfig:
    def test_default_values(self):
        config = CorrectnessConfig()
        assert config.rtol == 1e-2
        assert config.atol == 1e-2

    def test_custom_values(self):
        config = CorrectnessConfig(rtol=1e-3, atol=1e-4)
        assert config.rtol == 1e-3
        assert config.atol == 1e-4

    def test_frozen(self):
        config = CorrectnessConfig()
        with pytest.raises(Exception):
            config.rtol = 0.5  # type: ignore[invalid-assignment]


class TestCheckCorrectness:
    @pytest.fixture
    def config(self):
        return CorrectnessConfig(rtol=1e-2, atol=1e-2)

    def test_identical_tensors(self, config):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        passed, details = check_correctness(a, b, config)
        assert passed is True
        assert details["max_abs_error"] == 0.0

    def test_within_tolerance(self, config):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.005, 2.01, 3.015])
        passed, details = check_correctness(a, b, config)
        assert passed is True

    def test_outside_tolerance(self, config):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.5, 3.0])
        passed, details = check_correctness(a, b, config)
        assert passed is False
        assert "error_message" in details

    def test_shape_mismatch(self, config):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0])
        passed, details = check_correctness(a, b, config)
        assert passed is False
        assert "Shape mismatch" in details["error_message"]

    def test_nan_mismatch(self, config):
        a = torch.tensor([1.0, float("nan"), 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        passed, details = check_correctness(a, b, config)
        assert passed is False

    def test_both_nan_passes(self, config):
        a = torch.tensor([1.0, float("nan"), 3.0])
        b = torch.tensor([1.0, float("nan"), 3.0])
        passed, details = check_correctness(a, b, config)
        assert passed is True

    def test_inf_mismatch(self, config):
        a = torch.tensor([1.0, float("inf"), 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        passed, details = check_correctness(a, b, config)
        assert passed is False

    def test_multidimensional(self, config):
        a = torch.randn(2, 3, 4)
        b = a.clone()
        passed, details = check_correctness(a, b, config)
        assert passed is True

    def test_bfloat16(self, config):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        passed, details = check_correctness(a, b, config)
        assert passed is True

    def test_returns_max_errors(self, config):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.1, 2.0, 3.0])
        passed, details = check_correctness(a, b, config)
        assert "max_abs_error" in details
        assert "max_rel_error" in details
        assert details["max_abs_error"] == pytest.approx(0.1)


class TestMeanAggregation:
    @pytest.fixture
    def agg(self):
        return MeanAggregation()

    def test_empty_latencies(self, agg):
        result = agg.aggregate(latencies=[])
        assert result["mean_latency_ms"] == float("inf")
        assert result["mean_speedup"] == 0.0
        assert result["score"] == 0.0

    def test_single_latency(self, agg):
        result = agg.aggregate(latencies=[10.0])
        assert result["mean_latency_ms"] == 10.0
        assert result["score"] == pytest.approx(0.1)

    def test_multiple_latencies(self, agg):
        result = agg.aggregate(latencies=[10.0, 20.0, 30.0])
        assert result["mean_latency_ms"] == 20.0
        assert result["score"] == pytest.approx(0.05)

    def test_with_speedups(self, agg):
        result = agg.aggregate(latencies=[10.0, 20.0], speedups=[2.0, 3.0])
        assert result["mean_latency_ms"] == 15.0
        assert result["mean_speedup"] == 2.5
        assert result["score"] == 2.5

    def test_empty_speedups_list(self, agg):
        result = agg.aggregate(latencies=[10.0], speedups=[])
        assert result["mean_speedup"] == 0.0
        assert result["score"] == pytest.approx(0.1)


class TestGeometricMeanAggregation:
    @pytest.fixture
    def agg(self):
        return GeometricMeanAggregation()

    def test_empty_latencies(self, agg):
        result = agg.aggregate(latencies=[])
        assert result["mean_latency_ms"] == float("inf")
        assert result["mean_speedup"] == 0.0
        assert result["score"] == 0.0

    def test_single_latency(self, agg):
        result = agg.aggregate(latencies=[10.0])
        assert result["mean_latency_ms"] == pytest.approx(10.0)

    def test_multiple_latencies(self, agg):
        result = agg.aggregate(latencies=[4.0, 9.0])
        assert result["mean_latency_ms"] == pytest.approx(6.0)

    def test_geometric_mean_property(self, agg):
        result = agg.aggregate(latencies=[2.0, 8.0])
        assert result["mean_latency_ms"] == pytest.approx(4.0)

    def test_with_speedups(self, agg):
        result = agg.aggregate(latencies=[4.0, 9.0], speedups=[2.0, 8.0])
        assert result["mean_latency_ms"] == pytest.approx(6.0)
        assert result["mean_speedup"] == pytest.approx(4.0)
        assert result["score"] == pytest.approx(4.0)

    def test_zero_latency_handled(self, agg):
        result = agg.aggregate(latencies=[0.0, 10.0])
        assert result["mean_latency_ms"] == float("inf")
        assert result["score"] == 0.0

    def test_negative_latency_handled(self, agg):
        result = agg.aggregate(latencies=[-1.0, 10.0])
        assert result["mean_latency_ms"] == float("inf")
        assert result["score"] == 0.0

    def test_zero_speedup_handled(self, agg):
        result = agg.aggregate(latencies=[10.0], speedups=[0.0])
        assert result["mean_speedup"] == 0.0


class TestBenchmarkConfig:
    def test_default_values(self):
        config = BenchmarkConfig()
        assert config.warmup == 25
        assert config.rep == 100

    def test_custom_values(self):
        config = BenchmarkConfig(warmup=50, rep=200)
        assert config.warmup == 50
        assert config.rep == 200

    def test_frozen(self):
        config = BenchmarkConfig()
        with pytest.raises(Exception):
            config.warmup = 20  # type: ignore[invalid-assignment]
