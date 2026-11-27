"""The ROC/PR plot script uses the configured cache and average precision (issue #10)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from matplotlib.figure import Figure  # noqa: E402

from helpers import RUN, write_experiment  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_roc_pr_curves.py"

# Ranked by score: positive, negative, positive, negative. Average precision is
# 0.5 * 1 (first positive) + 0.5 * 2/3 (second positive) = 5/6.
PREDICTIONS = "y_true,y_pred_proba\n1,0.9\n0,0.8\n1,0.7\n0,0.1\n"
EXPERIMENT = {f"{RUN}/predictions/test_predictions.csv": PREDICTIONS}


@pytest.fixture
def plot_script(monkeypatch):
    spec = importlib.util.spec_from_file_location("plot_roc_pr_curves", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run(*argv: str) -> int:
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), *argv])
        return module.main()

    return run


@pytest.fixture
def saved_legends(monkeypatch):
    """Capture each saved figure's legend labels, per axis title."""
    legends: dict[str, list[str]] = {}
    original = Figure.savefig

    def savefig(figure, *args, **kwargs):
        for axis in figure.axes:
            legends[axis.get_title()] = [text.get_text() for text in axis.get_legend().texts]
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    return legends


def test_script_reads_the_configured_cache_folder(plot_script, cache_dir, tmp_path):
    write_experiment(cache_dir, "exp", EXPERIMENT)

    assert plot_script("--experiment-name", "exp", "--output-dir", str(tmp_path / "out")) == 0
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["split_a_test_roc_pr.png"]


def test_cache_dir_argument_overrides_the_configured_cache(plot_script, cache_dir, tmp_path):
    other_cache = tmp_path / "other_cache"
    write_experiment(other_cache, "exp", EXPERIMENT)

    assert plot_script("--experiment-name", "exp", "--cache-dir", str(other_cache)) == 0
    assert (other_cache / "exp" / "plots" / "split_a_test_roc_pr.png").exists()
    assert not (cache_dir / "exp").exists()


def test_missing_experiment_in_configured_cache_fails(plot_script, cache_dir):
    assert plot_script("--experiment-name", "missing") == 1


def test_precision_recall_curves_are_labelled_with_average_precision(
    plot_script, cache_dir, saved_legends
):
    write_experiment(cache_dir, "exp", EXPERIMENT)

    plot_script("--experiment-name", "exp")

    pr_legend = next(labels for title, labels in saved_legends.items() if "Precision" in title)
    assert pr_legend[0] == "model_x (AP=0.833)"
