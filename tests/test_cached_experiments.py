import pytest

from helpers import RUN, write_experiment

METRICS = f"{RUN}/results/test_results.csv"
CONFIG = f"{RUN}/logs/config.yaml"


def test_list_cached_experiments(call_tool, cache_dir):
    write_experiment(cache_dir, "exp")
    (cache_dir / ".hidden").mkdir()

    assert call_tool("list_cached_experiments")["experiments"] == ["exp"]


def test_summarize_finds_docs_runs_metrics_predictions_and_configs(call_tool, cache_dir):
    write_experiment(cache_dir, "exp")

    summary = call_tool("summarize_experiment", experiment_name="exp")

    assert summary["docs"] == ["README.md"]
    assert summary["run_directories"] == [RUN]
    assert summary["metrics_files"] == sorted([METRICS, f"{RUN}/results/cv_results.csv"])
    assert summary["prediction_files"] == [f"{RUN}/predictions/test_predictions.csv"]
    assert summary["config_files"] == [CONFIG]


def test_summarize_unknown_experiment_returns_error(call_tool, cache_dir):
    assert "error" in call_tool("summarize_experiment", experiment_name="missing")


def test_read_metrics_returns_columns_and_rows(call_tool, cache_dir):
    write_experiment(cache_dir, "exp")

    result = call_tool("read_experiment_metrics", experiment_name="exp", relative_path=METRICS)

    assert result["columns"] == ["metric", "value"]
    assert result["row_count"] == 2
    assert result["preview_rows"][0] == {"metric": "roc_auc", "value": 0.9}


def test_read_metrics_rejects_non_csv(call_tool, cache_dir):
    write_experiment(cache_dir, "exp")

    result = call_tool("read_experiment_metrics", experiment_name="exp", relative_path=CONFIG)

    assert "error" in result


def test_read_config_returns_parsed_yaml(call_tool, cache_dir):
    write_experiment(cache_dir, "exp")

    result = call_tool("read_experiment_config", experiment_name="exp", relative_path=CONFIG)

    assert result["content"] == {"model": "xgboost", "seed": 1}


OUTSIDE_PATHS = {
    "sibling": "../exp_secret/secret.{ext}",
    "dot-dot": RUN + "/../../../../exp_secret/secret.{ext}",
    "symlink": "link.{ext}",
    "absolute": "{outside}/secret.{ext}",
}


@pytest.mark.parametrize("route", OUTSIDE_PATHS)
def test_reads_cannot_leave_the_experiment(call_tool, cache_dir, tmp_path, route):
    secrets = {"secret.csv": "a\n1\n", "secret.yaml": "a: 1\n"}
    write_experiment(cache_dir, "exp")
    write_experiment(cache_dir, "exp_secret", secrets)
    outside = write_experiment(tmp_path, "outside", secrets)
    for ext in ("csv", "yaml"):
        (cache_dir / "exp" / f"link.{ext}").symlink_to(cache_dir / "exp_secret" / f"secret.{ext}")

    for tool, ext in (("read_experiment_metrics", "csv"), ("read_experiment_config", "yaml")):
        relative_path = OUTSIDE_PATHS[route].format(ext=ext, outside=outside)
        result = call_tool(tool, experiment_name="exp", relative_path=relative_path)

        assert "outside" in result["error"], (tool, relative_path)
