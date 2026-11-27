"""Results tools answer from every row and match a direct full-file calculation (issue #6)."""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest
from fastmcp import Client
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from experiment_explorer.server import mcp
from helpers import write_experiment
from results_fixtures import (
    BACTERIA,
    CV_ONLY_RUN,
    MEMBERS,
    RUN,
    SINGLE_CLASS_BACTERIUM,
    cv_metrics_frame,
    cv_per_bacteria_frame,
    cv_predictions_frame,
    per_bacteria_frame,
    predictions_frame,
    results_experiment,
    run_files,
    run_results_frame,
)


@pytest.fixture
def experiment(cache_dir):
    write_experiment(cache_dir, "exp", results_experiment())
    return "exp"


def stats(series: pd.Series) -> dict:
    values = series.dropna()
    return {
        "n": len(values),
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
    }


def assert_stats(actual: dict, series: pd.Series) -> None:
    expected = stats(series)
    assert actual["n"] == expected["n"]
    for key in ("mean", "std", "min", "max"):
        assert actual[key] == pytest.approx(expected[key])


# --- summarize_run ---------------------------------------------------------------


def test_summary_headline_is_the_ensemble_test_value(call_tool, experiment):
    result = call_tool("summarize_run", experiment_name=experiment, run=RUN)

    ensemble = run_results_frame().query("model_idx == 'ensemble'").iloc[0]
    assert result["headline"]["partition"] == "test"
    assert result["headline"]["metrics"]["roc_auc"] == pytest.approx(ensemble["roc_auc"])
    assert result["test"]["row_count"] == len(MEMBERS) + 1


def test_summary_member_spread_never_includes_the_ensemble(call_tool, experiment):
    result = call_tool("summarize_run", experiment_name=experiment, run=RUN)

    members = run_results_frame().query("model_idx != 'ensemble'")
    assert result["test"]["member_count"] == len(MEMBERS)
    assert_stats(result["test"]["members"]["roc_auc"], members["roc_auc"])


def test_summary_cv_spread_matches_the_parsed_fold_metrics(call_tool, experiment):
    result = call_tool("summarize_run", experiment_name=experiment, run=RUN)

    folds = cv_metrics_frame()
    assert result["cv"]["fold_count"] == len(folds)
    assert_stats(result["cv"]["folds"]["roc_auc"], folds["roc_auc"])
    assert result["cv"]["folds"]["specificity"]["n"] == 0


def test_cv_only_run_headline_is_the_mean_across_folds(call_tool, experiment):
    result = call_tool("summarize_run", experiment_name=experiment, run=CV_ONLY_RUN)

    assert result["headline"]["partition"] == "cv"
    assert result["headline"]["metrics"]["roc_auc"] == pytest.approx(
        cv_metrics_frame(seed=9)["roc_auc"].mean()
    )
    assert result["test"] is None


def test_summary_of_an_incomplete_run_says_why(call_tool, cache_dir):
    write_experiment(cache_dir, "exp", {"a/results_20260101_000000/logs/config.yaml": "x: 1\n"})

    result = call_tool("summarize_run", experiment_name="exp", run="a/results_20260101_000000")

    assert "no results folder" in result["error"]


def test_summary_rejects_a_folder_that_is_not_a_run(call_tool, experiment):
    result = call_tool("summarize_run", experiment_name=experiment, run="split_a")

    assert "relative_path" in result["error"]


# --- compare_runs ----------------------------------------------------------------

FOLDER = "split_b/model_y"


def run_path(stamp: str) -> str:
    return f"{FOLDER}/results_{stamp}"


@pytest.fixture
def comparison(cache_dir):
    files: dict[str, str] = {}
    # Same evaluation setup: an exact repeat pair, and one with other model settings.
    files |= run_files(run_path("20260103_000000"), seed=1, output_dir="o1")
    files |= run_files(run_path("20260101_000000"), seed=1, output_dir="o2")
    files |= run_files(run_path("20260102_000000"), seed=2, max_depth=9)
    # Different test data.
    files |= run_files(run_path("20260104_000000"), seed=3, test_sources=("basel",))
    # Incomplete.
    files[f"{run_path('20260105_000000')}/logs/config.yaml"] = "x: 1\n"
    write_experiment(cache_dir, "cmp", files)
    return "cmp"


def only_folder(result: dict) -> dict:
    assert result["folder_count"] == 1
    return result["folders"][0]


def test_runs_are_grouped_by_evaluation_setup(call_tool, comparison):
    folder = only_folder(call_tool("compare_runs", experiment_name=comparison))

    setups = [group["evaluation_setup"] for group in folder["setup_groups"]]
    assert setups == [{"data.test_sources": ["our_data"]}, {"data.test_sources": ["basel"]}]
    assert folder["shared_evaluation_setup"]["data.split_method"] == "known_phage"


def test_exact_repeats_are_merged_and_runs_are_in_time_order(call_tool, comparison):
    folder = only_folder(call_tool("compare_runs", experiment_name=comparison))

    entries = folder["setup_groups"][0]["entries"]
    assert [entry["runs"] for entry in entries] == [
        [run_path("20260101_000000"), run_path("20260103_000000")],
        [run_path("20260102_000000")],
    ]
    assert entries[0]["repeats_agree"] is True
    assert entries[0]["results"]["roc_auc"] == pytest.approx(0.94)


def test_only_differing_model_settings_are_shown(call_tool, comparison):
    folder = only_folder(call_tool("compare_runs", experiment_name=comparison))

    entries = folder["setup_groups"][0]["entries"]
    assert entries[0]["model_settings"] == {"model.max_depth": 6}
    assert entries[1]["model_settings"] == {"model.max_depth": 9}


def test_incomplete_runs_are_listed_with_a_reason(call_tool, comparison):
    folder = only_folder(call_tool("compare_runs", experiment_name=comparison))

    assert folder["incomplete_runs"] == [
        {"run": run_path("20260105_000000"), "starred": False, "reason": "no results folder"}
    ]


def test_no_best_run_is_ever_named(call_tool, comparison):
    text = json.dumps(call_tool("compare_runs", experiment_name=comparison)).lower()

    assert '"best' not in text
    assert '"rank' not in text


def test_a_starred_run_is_the_default_answer(call_tool, cache_dir):
    starred = f"{FOLDER}/results_20260102_000000_star"
    files = run_files(run_path("20260101_000000"), seed=1) | run_files(starred, seed=2)
    write_experiment(cache_dir, "star", files)

    folder = only_folder(call_tool("compare_runs", experiment_name="star"))

    assert folder["default_answer"] == starred
    assert "1 other run" in folder["note"]


def test_repeats_with_different_results_show_each(call_tool, cache_dir):
    first, second = run_path("20260101_000000"), run_path("20260102_000000")
    files = run_files(first, seed=1) | run_files(second, seed=2)
    write_experiment(cache_dir, "rep", files)

    entry = only_folder(call_tool("compare_runs", experiment_name="rep"))["setup_groups"][0][
        "entries"
    ][0]

    assert entry["repeats_agree"] is False
    assert set(entry["results_by_run"]) == {first, second}


def test_a_group_without_test_results_everywhere_uses_cv(call_tool, cache_dir):
    files = run_files(run_path("20260101_000000"), seed=1, max_depth=3)
    files |= run_files(run_path("20260102_000000"), seed=2, test=False)
    write_experiment(cache_dir, "mix", files)

    group = only_folder(call_tool("compare_runs", experiment_name="mix"))["setup_groups"][0]

    assert group["partition"] == "cv"
    assert all(entry["results"]["partition"] == "cv" for entry in group["entries"])
    assert group["entries"][0]["results"]["roc_auc"] == pytest.approx(
        cv_metrics_frame(seed=1)["roc_auc"].mean()
    )


def test_a_run_without_a_headline_is_listed_with_a_reason(call_tool, cache_dir):
    # Several test rows and no ensemble row, and no CV: neither gives one figure.
    no_headline = run_path("20260102_000000")
    files = run_files(run_path("20260101_000000"), seed=1, max_depth=3)
    files |= run_files(no_headline, seed=2, cv=False)
    files[f"{no_headline}/results/test_results.csv"] = "roc_auc,avg_precision\n0.7,0.5\n0.8,0.6\n"
    write_experiment(cache_dir, "nohead", files)

    group = only_folder(call_tool("compare_runs", experiment_name="nohead"))["setup_groups"][0]

    assert group["partition"] is None
    results = {entry["runs"][0]: entry["results"] for entry in group["entries"]}
    assert results[run_path("20260101_000000")]["partition"] == "test"
    assert results[no_headline]["partition"] is None
    assert results[no_headline]["roc_auc"] is None
    assert "summarize_run" in results[no_headline]["reason"]


def test_compare_can_be_limited_to_one_folder(call_tool, comparison, cache_dir):
    write_experiment(cache_dir, "cmp", run_files("other/results_20260101_000000"))

    result = call_tool("compare_runs", experiment_name=comparison, folder=FOLDER)

    assert [folder["folder"] for folder in result["folders"]] == [FOLDER]


# --- bacteria_metrics ------------------------------------------------------------


def test_stats_across_bacteria_use_every_ensemble_row(call_tool, experiment):
    result = call_tool("bacteria_metrics", experiment_name=experiment, run=RUN, partition="test")

    ensemble = per_bacteria_frame().query("model_idx == 'ensemble'")
    assert result["row_count"] == len(per_bacteria_frame())
    assert result["rows_used"] == len(ensemble) > 50
    assert_stats(result["across_bacteria"]["roc_auc"], ensemble["roc_auc"])
    assert_stats(result["across_bacteria"]["precision_10"], ensemble["precision_10"])


@pytest.mark.parametrize("name", ["132", "007", "kleb14"])
def test_bacteria_are_found_by_text_id_ignoring_case(call_tool, experiment, name):
    result = call_tool(
        "bacteria_metrics",
        experiment_name=experiment,
        run=RUN,
        partition="test",
        bacteria=[name],
        metric="roc_auc",
    )

    frame = per_bacteria_frame()
    row = frame[(frame.bacteria.str.lower() == name.lower()) & (frame.model_idx == "ensemble")]
    assert result["bacteria"][0]["values"]["roc_auc"] == pytest.approx(row["roc_auc"].iloc[0])


def test_unknown_bacterium_lists_close_ids_instead_of_guessing(call_tool, experiment):
    result = call_tool(
        "bacteria_metrics",
        experiment_name=experiment,
        run=RUN,
        partition="test",
        bacteria=["KLEB1"],
    )

    assert "KLEB12" in result["error"] and "KLEB14" in result["error"]


def test_cv_lookup_lists_every_fold_and_their_mean(call_tool, experiment):
    result = call_tool(
        "bacteria_metrics",
        experiment_name=experiment,
        run=RUN,
        partition="cv",
        bacteria=["KLEB12"],
        metric="roc_auc",
    )

    rows = cv_per_bacteria_frame().query("bacteria == 'KLEB12'")
    entry = result["bacteria"][0]
    assert [fold["fold"] for fold in entry["by_fold"]] == [0, 2, 4]
    assert entry["values"]["roc_auc"] == pytest.approx(rows["roc_auc"].mean())
    assert entry["spread"]["roc_auc"]["n"] == 3


def test_cv_stats_count_each_bacterium_once(call_tool, experiment):
    result = call_tool(
        "bacteria_metrics", experiment_name=experiment, run=RUN, partition="cv", metric="roc_auc"
    )

    per_bacterium = cv_per_bacteria_frame().groupby("bacteria")["roc_auc"].mean()
    assert result["bacteria_count"] == len(BACTERIA)
    assert_stats(result["across_bacteria"]["roc_auc"], per_bacterium)


def test_threshold_count_and_worst_bacteria(call_tool, experiment):
    result = call_tool(
        "bacteria_metrics",
        experiment_name=experiment,
        run=RUN,
        partition="test",
        metric="roc_auc",
        threshold=0.7,
        worst_n=3,
        max_groups=5,
    )

    values = per_bacteria_frame().query("model_idx == 'ensemble'").set_index("bacteria")["roc_auc"]
    above = values[values > 0.7]
    assert result["threshold"]["count"] == len(above)
    assert result["threshold"]["n"] == len(values)
    assert len(result["threshold"]["bacteria"]) == 5
    assert result["threshold"]["truncated"] is True
    worst = values.sort_values().head(3)
    assert [item["bacteria"] for item in result["worst"]["bacteria"]] == list(worst.index)


def test_members_each_shows_every_member_for_a_bacterium(call_tool, experiment):
    result = call_tool(
        "bacteria_metrics",
        experiment_name=experiment,
        run=RUN,
        partition="test",
        bacteria=["132"],
        metric="roc_auc",
        members="each",
    )

    rows = per_bacteria_frame().query("bacteria == '132' and model_idx != 'ensemble'")
    entry = result["bacteria"][0]
    assert [item["member"] for item in entry["by_member"]] == list(MEMBERS)
    assert entry["values"]["roc_auc"] == pytest.approx(rows["roc_auc"].mean())


def test_threshold_needs_a_metric(call_tool, experiment):
    result = call_tool(
        "bacteria_metrics", experiment_name=experiment, run=RUN, partition="test", threshold=0.5
    )

    assert "metric" in result["error"]


# --- prediction_metrics ----------------------------------------------------------


def sklearn_metrics(frame: pd.DataFrame, label: str = "y_true") -> dict:
    y, p, yp = frame[label], frame["y_pred_proba"], frame["y_pred"]
    return {
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "precision": precision_score(y, yp),
        "recall": recall_score(y, yp),
        "f1": f1_score(y, yp),
        "mcc": matthews_corrcoef(y, yp),
    }


def test_prediction_metrics_match_sklearn_on_every_ensemble_row(call_tool, experiment):
    result = call_tool("prediction_metrics", experiment_name=experiment, run=RUN, partition="test")

    ensemble = predictions_frame().query("model_idx == 'ensemble'")
    assert result["row_count"] == len(predictions_frame())
    assert result["rows_used"] == len(ensemble) > 50
    for name, value in sklearn_metrics(ensemble).items():
        assert result["metrics"][name] == pytest.approx(value)
    matrix = result["metrics"]["confusion_matrix"]
    assert matrix["tp"] + matrix["fp"] + matrix["tn"] + matrix["fn"] == len(ensemble)


def test_prediction_metrics_per_bacteria_skip_single_class_groups(call_tool, experiment):
    result = call_tool(
        "prediction_metrics",
        experiment_name=experiment,
        run=RUN,
        partition="test",
        group_by="bacteria",
        max_groups=200,
    )

    ensemble = predictions_frame().query("model_idx == 'ensemble'")
    groups = {group["bacteria"]: group for group in result["groups"]}
    assert result["group_count"] == len(BACTERIA)
    assert groups[SINGLE_CLASS_BACTERIUM]["roc_auc"] is None
    assert "single class" in groups[SINGLE_CLASS_BACTERIUM]["reason"]
    expected = {}
    for name, rows in ensemble.groupby("bacteria"):
        if rows["y_true"].nunique() == 2:
            expected[name] = roc_auc_score(rows["y_true"], rows["y_pred_proba"])
            assert groups[name]["roc_auc"] == pytest.approx(expected[name])
    assert result["skipped"] == len(BACTERIA) - len(expected)
    assert_stats(result["across_groups"]["roc_auc"], pd.Series(expected))


def test_prediction_groups_are_capped_and_flagged(call_tool, experiment):
    result = call_tool(
        "prediction_metrics",
        experiment_name=experiment,
        run=RUN,
        partition="test",
        group_by="phage",
        max_groups=3,
    )

    assert result["group_count"] == 8
    assert len(result["groups"]) == 3
    assert result["truncated"] is True


def test_cv_prediction_metrics_by_fold(call_tool, experiment):
    result = call_tool(
        "prediction_metrics", experiment_name=experiment, run=RUN, partition="cv", group_by="fold"
    )

    frame = cv_predictions_frame()
    for group in result["groups"]:
        rows = frame[frame.fold_idx == group["fold"]]
        assert group["roc_auc"] == pytest.approx(roc_auc_score(rows.y_true, rows.y_pred_proba))
    assert result["truncated"] is False


def test_member_spread_from_predictions(call_tool, experiment):
    result = call_tool(
        "prediction_metrics",
        experiment_name=experiment,
        run=RUN,
        partition="test",
        members="each",
    )

    frame = predictions_frame().query("model_idx != 'ensemble'")
    per_member = frame.groupby("model_idx").apply(
        lambda rows: roc_auc_score(rows.y_true, rows.y_pred_proba), include_groups=False
    )
    assert [item["member"] for item in result["by_member"]] == list(MEMBERS)
    assert_stats(result["member_spread"]["roc_auc"], per_member)


def test_interaction_label_and_missing_y_pred(call_tool, experiment):
    result = call_tool(
        "prediction_metrics",
        experiment_name=experiment,
        relative_path="other/predictions_normalized/model_test.csv",
    )

    frame = predictions_frame(seed=6, label="interaction").query("model_idx == 'ensemble'")
    assert result["metrics"]["roc_auc"] == pytest.approx(
        roc_auc_score(frame.interaction, frame.y_pred_proba)
    )
    assert result["metrics"]["f1"] is None
    assert "y_pred" in result["metrics"]["threshold_metrics_reason"]
    assert result["partition"] == "test"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"run": RUN, "partition": "test", "relative_path": "x.csv"}, "not both"),
        ({"run": RUN}, "partition"),
        ({"run": RUN, "partition": "train"}, "partition"),
        ({"run": RUN, "partition": "test", "group_by": "fold"}, "fold_idx"),
        ({"run": RUN, "partition": "test", "max_groups": 0}, "max_groups"),
        ({"relative_path": "../outside.csv"}, "outside"),
    ],
)
def test_prediction_metrics_rejects_bad_input(call_tool, experiment, arguments, message):
    result = call_tool("prediction_metrics", experiment_name=experiment, **arguments)

    assert message in result["error"]


# --- Steering --------------------------------------------------------------------

CALCULATION_TOOLS = ("summarize_run", "compare_runs", "bacteria_metrics", "prediction_metrics")


def tool_descriptions() -> dict[str, str]:
    async def run() -> dict[str, str]:
        async with Client(mcp) as client:
            return {tool.name: tool.description for tool in await client.list_tools()}

    return asyncio.run(run())


def test_server_instructions_steer_calculations_to_the_tools():
    instructions = " ".join(mcp.instructions.split())

    for tool in CALCULATION_TOOLS:
        assert tool in instructions
    assert "experiment_explorer.tables" in instructions
    assert "Never name a best run" in instructions


def test_preview_tool_points_to_the_calculation_tools():
    description = tool_descriptions()["read_experiment_metrics"]

    for tool in CALCULATION_TOOLS:
        assert tool in description


@pytest.mark.parametrize("tool", CALCULATION_TOOLS)
def test_calculation_tools_open_with_the_questions_they_answer(tool):
    assert tool_descriptions()[tool].startswith("Answer")
