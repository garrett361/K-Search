"""Tests for config module utilities."""


class TestCollectGitInfo:
    def test_returns_commit_branch_dirty_when_in_repo(self):
        from k_search.modular.config import collect_git_info
        result = collect_git_info()
        # In K-Search repo, should have these keys
        if result:  # may be empty outside git
            assert set(result.keys()) == {"commit", "branch", "dirty"}
            assert isinstance(result["dirty"], bool)

    def test_returns_empty_dict_on_subprocess_failure(self):
        from k_search.modular.config import collect_git_info
        from unittest.mock import patch
        with patch("subprocess.run", side_effect=OSError("git not found")):
            assert collect_git_info() == {}


class TestCollectEnvInfo:
    def test_returns_hostname_and_python_version(self):
        from k_search.modular.config import collect_env_info
        result = collect_env_info()
        assert "hostname" in result
        assert "python_version" in result
        assert result["python_version"].startswith("3.")

    def test_returns_empty_dict_on_complete_failure(self):
        from k_search.modular.config import collect_env_info
        from unittest.mock import patch
        with patch("socket.gethostname", side_effect=OSError):
            assert collect_env_info() == {}


class TestBuildRunConfig:
    def test_builds_complete_config_structure(self):
        from k_search.modular.config import build_run_config, SearchConfig, MetricsConfig, ArtifactConfig

        result = build_run_config(
            run_id="test-run", model_name="test-model", reasoning_effort="medium",
            search_config=SearchConfig(max_rounds=5),
            metrics_config=MetricsConfig(),
            artifact_config=ArtifactConfig(output_dir="/tmp"),
            task="causal_conv1d", language="triton",
        )

        assert result["run_id"] == "test-run"
        assert result["model_name"] == "test-model"
        assert result["search"]["max_rounds"] == 5
        assert result["task_config"] == {"task": "causal_conv1d", "language": "triton"}
        assert isinstance(result["git"], dict)
        assert isinstance(result["env"], dict)
        assert result["artifacts"]["output_dir"] == "/tmp"
