"""Summaries for every experiment layout in use (issue #8, docs/adr/0003).

Each fixture copies the file names of a real layout; contents are minimal.
"""

from __future__ import annotations

import pytest

from helpers import write_experiment

TS = "20260101_120000"
TS2 = "20260102_120000"

ORIGINAL = {
    "README.md": "# Baseline\n",
    "run_exp.sh": "python train.py\n",
    **{
        f"split_known_phage/model_xgboost/results_{stamp}/{name}": "x\n"
        for stamp in (TS, f"{TS2}_star")
        for name in (
            "results/test_results.csv",
            "results/cv_results.csv",
            "results/test_per_bacteria_results.csv",
            "results/cv_per_bacteria_results.csv",
            "results/summary.txt",
            "predictions/test_predictions.csv",
            "predictions/cv_predictions.csv",
            "predictions/confusion_matrix.ipynb",
            "logs/config.yaml",
            "artifacts/feature_order.txt",
        )
    },
}

PRODUCTION = {
    "README.md": "# Holdout\n",
    "run_holdout.py": "x\n",
    "run_holdout.log": "x\n",
    "config_reference.yaml": "x: 1\n",
    "holdout_run/holdout_lobo_metrics.json": "{}\n",
    "holdout_run/combined_test_predictions.csv": "x\n",
    f"holdout_run/production/production_{TS}/lobo_fold_manifest.json": "{}\n",
    f"holdout_run/production/production_{TS}/logs/config.yaml": "x: 1\n",
    f"holdout_run/production/production_{TS}/results/test_metrics.csv": "x\n",
    f"holdout_run/production/production_{TS}/results/training_metrics.csv": "x\n",
    f"holdout_run/production/production_{TS}/results/leaderboard_production.csv": "x\n",
    f"holdout_run/production/production_{TS}/predictions/test_predictions.csv": "x\n",
    f"holdout_run/production/production_{TS}/models/production_model.pkl": "x\n",
}

SWEEP = {
    "README.md": "# Sweep\n",
    f"sweep/run_{TS}/results_{TS}/results/cv_results.csv": "x\n",
    f"sweep/run_{TS}/results_{TS}/logs/config.yaml": "x: 1\n",
    f"sweep/run_{TS}/results_{TS}/predictions/cv_predictions.csv": "x\n",
    f"sweep/run_{TS}/results_{TS}/production_{TS2}/results/test_metrics.csv": "x\n",
    f"basel/results_XGBoost_known_phage/results_{TS2}/results/test_results.csv": "x\n",
}

COMPARISON = {
    "README.md": "# Comparison\n",
    "SUMMARY.md": "# Findings\n",
    "METRIC_DEFINITIONS.md": "# Metrics\n",
    "manifest.json": "{}\n",
    "comparison_config.json": "{}\n",
    "compare.py": "x\n",
    "metrics/ranking_at_k.csv": "x\n",
    "metrics/config_comparison.csv": "x\n",
    "plots/spider_plot_test.png": "x\n",
    "plots/spider_plot_test.html": "x\n",
    "predictions_normalized/genophi_test.csv": "x\n",
    "predictions_normalized/genophi_cv.csv": "x\n",
}

ANALYSIS = {
    "README.md": "# Diversity\n",
    "REPORT.md": "# Report\n",
    "outputs/REPORT.md": "# Report\n",
    "notes.md": "",
    "experiment_config.yaml": "x: 1\n",
    "analyze_diversity.py": "x\n",
    "data/manifest.csv": "x\n",
    "data/pc_features.parquet": "x\n",
    "outputs/tables/metrics.json": "{}\n",
    "outputs/tables/jaccard_clusters.csv": "x\n",
    "outputs/trees/jaccard_nj.nwk": "x\n",
    "figures/umap_jaccard.png": "x\n",
    "figures/umap_jaccard.pdf": "x\n",
}

GENOPHI = {
    "README.md": "# GenoPHI\n",
    "run_manifest.txt": "x\n",
    "external_evaluation/evaluation_metrics.csv": "x\n",
    "external_evaluation/evaluation_metrics.json": "{}\n",
    "external_evaluation/median_predictions.csv": "x\n",
    "external_evaluation/evaluation_pairs.csv": "x\n",
    "training/feature_selection/run_1/best_model_predictions.csv": "x\n",
    "training/feature_selection/run_1/feature_importances.csv": "x\n",
    "training/feature_selection/catboost_info/catboost_training.json": "{}\n",
    "merged_features/full_feature_table.csv": "x\n",
}

LEARNING_CURVES = {
    "README.md": "# Learning curves\n",
    "REPORT.md": "# Report\n",
    "experiment_config.yaml": "x: 1\n",
    "outputs/aggregated/learning_curve_means.csv": "x\n",
    "outputs/aggregated/saturation_fits.json": "{}\n",
    "outputs/raw_results/known_phage/fold_0/n100/rep0/metrics.json": "{}\n",
    "outputs/raw_results/known_phage/fold_0/n100/rep0/per_bacteria.csv": "x\n",
}

LAYOUTS = {
    "original": ORIGINAL,
    "production": PRODUCTION,
    "sweep": SWEEP,
    "comparison": COMPARISON,
    "analysis": ANALYSIS,
    "genophi": GENOPHI,
    "learning_curves": LEARNING_CURVES,
}

# What each layout's summary must list: reports, metric files (tables or other),
# configs and predictions.
EXPECTED = {
    "production": {
        "docs": ["README.md"],
        "metrics": [
            "holdout_run/holdout_lobo_metrics.json",
            f"holdout_run/production/production_{TS}/results/leaderboard_production.csv",
            f"holdout_run/production/production_{TS}/results/test_metrics.csv",
            f"holdout_run/production/production_{TS}/results/training_metrics.csv",
        ],
        "config_files": [
            "config_reference.yaml",
            f"holdout_run/production/production_{TS}/lobo_fold_manifest.json",
            f"holdout_run/production/production_{TS}/logs/config.yaml",
        ],
        "prediction_files": [
            "holdout_run/combined_test_predictions.csv",
            f"holdout_run/production/production_{TS}/predictions/test_predictions.csv",
        ],
    },
    "comparison": {
        "docs": ["METRIC_DEFINITIONS.md", "README.md", "SUMMARY.md"],
        "metrics": ["metrics/config_comparison.csv", "metrics/ranking_at_k.csv"],
        "config_files": ["comparison_config.json", "manifest.json"],
        "prediction_files": [
            "predictions_normalized/genophi_cv.csv",
            "predictions_normalized/genophi_test.csv",
        ],
    },
    "analysis": {
        "docs": ["README.md", "REPORT.md"],
        "metrics": [
            "outputs/tables/jaccard_clusters.csv",
            "outputs/tables/metrics.json",
            "outputs/trees/jaccard_nj.nwk",
        ],
        "config_files": ["experiment_config.yaml"],
        "prediction_files": [],
    },
    "genophi": {
        "docs": ["README.md"],
        "metrics": [
            "external_evaluation/evaluation_metrics.csv",
            "external_evaluation/evaluation_metrics.json",
            "training/feature_selection/run_1/feature_importances.csv",
        ],
        "config_files": ["run_manifest.txt"],
        "prediction_files": [
            "external_evaluation/evaluation_pairs.csv",
            "external_evaluation/median_predictions.csv",
            "training/feature_selection/run_1/best_model_predictions.csv",
        ],
    },
    "learning_curves": {
        "docs": ["README.md", "REPORT.md"],
        "metrics": [
            "outputs/aggregated/learning_curve_means.csv",
            "outputs/aggregated/saturation_fits.json",
            "outputs/raw_results/known_phage/fold_0/n100/rep0/metrics.json",
            "outputs/raw_results/known_phage/fold_0/n100/rep0/per_bacteria.csv",
        ],
        "config_files": ["experiment_config.yaml"],
        "prediction_files": [],
    },
}


def summarize(call_tool, cache_dir, layout: str) -> dict:
    write_experiment(cache_dir, layout, LAYOUTS[layout])
    return call_tool("summarize_experiment", experiment_name=layout)


@pytest.mark.parametrize("layout", sorted(EXPECTED))
def test_every_layout_lists_reports_metrics_configs_and_predictions(
    call_tool, cache_dir, layout
):
    summary = summarize(call_tool, cache_dir, layout)
    expected = EXPECTED[layout]

    assert sorted(summary["docs"]) == expected["docs"]
    metrics = summary["metrics_files"] + summary["other_metrics_files"]
    assert sorted(metrics) == expected["metrics"]
    assert summary["config_files"] == expected["config_files"]
    assert summary["prediction_files"] == expected["prediction_files"]


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_every_file_in_a_layout_in_use_has_a_role(call_tool, cache_dir, layout):
    summary = summarize(call_tool, cache_dir, layout)

    assert "other" not in summary["files_by_role"]
    assert sum(role["count"] for role in summary["files_by_role"].values()) == len(
        LAYOUTS[layout]
    )


def test_the_original_layout_summary_is_unchanged(call_tool, cache_dir):
    summary = summarize(call_tool, cache_dir, "original")
    runs = [
        f"split_known_phage/model_xgboost/results_{TS}",
        f"split_known_phage/model_xgboost/results_{TS2}_star",
    ]

    assert summary["docs"] == ["README.md"]
    assert summary["run_directories"] == runs
    assert summary["run_count"] == 2
    assert summary["metrics_files"] == sorted(
        f"{run}/results/{name}"
        for run in runs
        for name in (
            "cv_per_bacteria_results.csv",
            "cv_results.csv",
            "test_per_bacteria_results.csv",
            "test_results.csv",
        )
    )
    assert summary["metrics_file_count"] == 8
    assert summary["prediction_files"] == sorted(
        f"{run}/predictions/{name}"
        for run in runs
        for name in ("cv_predictions.csv", "test_predictions.csv")
    )
    assert summary["config_files"] == [f"{run}/logs/config.yaml" for run in runs]


def test_runs_have_a_kind_and_wrappers_are_not_runs(call_tool, cache_dir):
    summary = summarize(call_tool, cache_dir, "sweep")

    assert summary["runs"] == [
        {
            "path": f"basel/results_XGBoost_known_phage/results_{TS2}",
            "kind": "results",
            "starred": False,
            "has_results_folder": True,
        },
        {
            "path": f"sweep/run_{TS}/results_{TS}",
            "kind": "results",
            "starred": False,
            "has_results_folder": True,
        },
        {
            "path": f"sweep/run_{TS}/results_{TS}/production_{TS2}",
            "kind": "production",
            "starred": False,
            "has_results_folder": True,
        },
    ]


def test_production_runs_are_runs(call_tool, cache_dir):
    summary = summarize(call_tool, cache_dir, "production")

    assert summary["run_directories"] == [f"holdout_run/production/production_{TS}"]
    assert summary["runs"][0]["kind"] == "production"


def test_starred_runs_are_marked(call_tool, cache_dir):
    summary = summarize(call_tool, cache_dir, "original")

    assert [run["starred"] for run in summary["runs"]] == [False, True]


def test_identical_report_copies_are_listed_once_and_empty_reports_apart(call_tool, cache_dir):
    summary = summarize(call_tool, cache_dir, "analysis")

    assert summary["docs"] == ["README.md", "REPORT.md"]
    assert summary["report_copies"] == {"REPORT.md": ["outputs/REPORT.md"]}
    assert summary["empty_reports"] == ["notes.md"]


def test_files_by_role_counts_the_roles_only_counted(call_tool, cache_dir):
    summary = summarize(call_tool, cache_dir, "analysis")

    assert summary["files_by_role"]["figure"] == {"count": 2, "by_top_folder": {"figures": 2}}
    assert summary["files_by_role"]["data"] == {"count": 2, "by_top_folder": {"data": 2}}
    assert summary["files_by_role"]["code"]["count"] == 1


def test_list_cached_files_gives_each_files_role_and_filters_by_role(call_tool, cache_dir):
    write_experiment(cache_dir, "cmp", COMPARISON)

    every = call_tool("list_cached_files", experiment_name="cmp")
    figures = call_tool("list_cached_files", experiment_name="cmp", role="figure")

    roles = {entry["relative_path"]: entry["role"] for entry in every["files"]}
    assert roles["compare.py"] == "code"
    assert roles["predictions_normalized/genophi_test.csv"] == "predictions"
    assert [entry["relative_path"] for entry in figures["files"]] == [
        "plots/spider_plot_test.html",
        "plots/spider_plot_test.png",
    ]
    assert figures["total_count"] == 2


def test_list_cached_files_rejects_an_unknown_role(call_tool, cache_dir):
    write_experiment(cache_dir, "cmp", COMPARISON)

    result = call_tool("list_cached_files", experiment_name="cmp", role="images")

    assert "role must be one of" in result["error"]


# --- A new layout needs only a rules file -----------------------------------------


NEW_PROJECT = {
    "writeup/paper.docx": "x\n",
    f"trials/trial_{TS}/scores.tsv": "x\n",
    f"trials/trial_{TS}/results/x.csv": "x\n",
}

NEW_PROJECT_RULES = """
roles:
  - role: report
    match: ["*.docx"]
  - role: metrics
    match: ["*/scores.tsv"]
run_folder:
  name_pattern: '^(?P<kind>[a-z]+)_\\d{8}_\\d{6}$'
  kinds: [trial]
"""


def test_a_project_rules_file_adds_a_layout_without_code_changes(
    call_tool, cache_dir, tmp_path, monkeypatch
):
    rules = tmp_path / "rules.yaml"
    rules.write_text(NEW_PROJECT_RULES)
    write_experiment(cache_dir, "new", NEW_PROJECT)

    before = call_tool("summarize_experiment", experiment_name="new")
    monkeypatch.setenv("EXPERIMENT_EXPLORER_LAYOUT_RULES", str(rules))
    after = call_tool("summarize_experiment", experiment_name="new")

    assert before["docs"] == [] and before["runs"] == []
    assert after["docs"] == ["writeup/paper.docx"]
    assert after["other_metrics_files"] == [f"trials/trial_{TS}/scores.tsv"]
    assert [run["kind"] for run in after["runs"]] == ["trial"]
    assert after["layout_rules"][0] == str(rules.resolve())


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("roles: [{role: pictures, match: ['*.png']}]", "needs a role"),
        ("roles: [{role: figure, match: '*.png'}]", "needs match"),
        ("run_folder: {name_pattern: 'x_\\d+', kinds: [x]}", "(?P<kind>"),
        ("roles: [", "not valid YAML"),
    ],
)
def test_a_broken_rules_file_returns_an_error_naming_it(
    call_tool, cache_dir, tmp_path, monkeypatch, content, message
):
    rules = tmp_path / "rules.yaml"
    rules.write_text(content)
    monkeypatch.setenv("EXPERIMENT_EXPLORER_LAYOUT_RULES", str(rules))
    write_experiment(cache_dir, "exp", ORIGINAL)

    error = call_tool("summarize_experiment", experiment_name="exp")["error"]

    assert message in error
    assert "rules.yaml" in error


def test_a_missing_rules_file_returns_an_error(call_tool, cache_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_EXPLORER_LAYOUT_RULES", str(tmp_path / "missing.yaml"))
    write_experiment(cache_dir, "exp", ORIGINAL)

    assert "Cannot read layout rules file" in call_tool(
        "summarize_experiment", experiment_name="exp"
    )["error"]
