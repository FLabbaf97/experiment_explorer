"""The results loader is a public API: Claude's own scripts import it directly."""

import math

import pytest

from experiment_explorer.tables import (
    find_runs,
    fold_metric_columns,
    load_table,
    parse_metric_dict,
    run_status,
    split_config,
)
from helpers import write_experiment
from results_fixtures import (
    BACTERIA,
    CV_ONLY_RUN,
    RUN,
    cv_metrics_frame,
    results_experiment,
    run_results_frame,
)


def test_metric_dict_text_unwraps_numpy_values_and_skips_nested_ones():
    text = (
        "{'roc_auc': np.float64(0.78), 'tp': np.int64(25), 'mcc': 0.3, "
        "'npv': nan, 'gap': np.float64(nan), 'low': -inf, 'k': {1: 0.5}, 'name': 'x'}"
    )

    parsed = parse_metric_dict(text)

    assert parsed["roc_auc"] == 0.78
    assert parsed["tp"] == 25
    assert parsed["mcc"] == 0.3
    assert math.isnan(parsed["npv"]) and math.isnan(parsed["gap"])
    assert parsed["low"] == -math.inf
    assert "k" not in parsed and "name" not in parsed


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('echo unsafe')",
        "{'a': open('x')}",
        "[1, 2]",
        "{'a': ",
    ],
)
def test_metric_dict_text_is_never_evaluated(text):
    with pytest.raises(ValueError):
        parse_metric_dict(text)


def test_cv_results_become_one_column_per_fold_metric(cache_dir):
    root = write_experiment(cache_dir, "exp", results_experiment())

    frame = load_table(root / RUN / "results" / "cv_results.csv")

    expected = cv_metrics_frame()
    assert len(frame) == len(expected)
    assert list(frame["val_roc_auc"]) == pytest.approx(list(expected["roc_auc"]))
    assert list(frame["val_tp"]) == list(expected["tp"])
    assert frame["val_specificity"].isna().all()
    assert "val_metrics" not in frame.columns
    assert "val_per_bacteria" not in frame.columns
    assert set(fold_metric_columns(frame)) == {
        "val_roc_auc",
        "val_auc_prc",
        "val_tp",
        "val_specificity",
    }


def test_ids_stay_text_and_ensemble_rows_are_flagged(cache_dir):
    root = write_experiment(cache_dir, "exp", results_experiment())

    frame = load_table(root / RUN / "results" / "test_per_bacteria_results.csv")

    assert set(frame["bacteria"]) == set(BACTERIA)
    assert "007" in set(frame["bacteria"])
    assert frame["is_ensemble"].sum() == len(BACTERIA)


def test_interaction_label_becomes_y_true(cache_dir):
    root = write_experiment(cache_dir, "exp", results_experiment())

    frame = load_table(root / "other" / "predictions_normalized" / "model_test.csv")

    assert "y_true" in frame.columns
    assert "interaction" not in frame.columns


def test_test_results_keep_every_row(cache_dir):
    root = write_experiment(cache_dir, "exp", results_experiment())

    frame = load_table(root / RUN / "results" / "test_results.csv")

    assert len(frame) == len(run_results_frame())


def test_load_table_rejects_files_that_are_not_csv(cache_dir):
    root = write_experiment(cache_dir, "exp", {"notes.txt": "hello"})

    with pytest.raises(ValueError):
        load_table(root / "notes.txt")


def test_run_status_reports_why_a_run_is_incomplete(cache_dir):
    root = write_experiment(
        cache_dir,
        "exp",
        {
            "a/results_20260101_000000/logs/config.yaml": "x: 1\n",
            "a/results_20260102_000000/results/test_results.csv": "roc_auc,model_idx\n",
            "a/results_20260103_000000/results/cv_results.csv": (
                "fold_idx,val_metrics\n0,\"{'roc_auc': \"\n"
            ),
            "a/results_20260104_000000/results/notes.txt": "nothing",
        },
    )

    reasons = {path.split("/")[-1]: run_status(root / path).reason for path in find_runs(root)}

    assert reasons["results_20260101_000000"] == "no results folder"
    assert "empty" in reasons["results_20260102_000000"]
    assert "won't parse" in reasons["results_20260103_000000"]
    assert reasons["results_20260104_000000"] == "no test or CV results file"


def test_cv_only_run_has_results(cache_dir):
    root = write_experiment(cache_dir, "exp", results_experiment())

    status = run_status(root / CV_ONLY_RUN)

    assert status.has_results and status.has_cv and not status.has_test


def test_config_splits_into_evaluation_setup_and_model_settings():
    config = {
        "data": {
            "test_sources": ["a"],
            "k_folds": 5,
            "upsampling_method": "SMOTE",
            "encoder_path": "enc.pkl",
        },
        "model": {"max_depth": 3},
        "output_dir": "x",
        "wandb": {"project": "p"},
        "save_models": True,
        "random_seed": 1,
    }

    setup, settings = split_config(config)

    assert setup == {"data.test_sources": ["a"], "data.k_folds": 5, "random_seed": 1}
    assert settings == {"data.upsampling_method": "SMOTE", "model.max_depth": 3}
