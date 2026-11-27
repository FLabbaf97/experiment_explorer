"""FastMCP server exposing experiment exploration tools."""

from __future__ import annotations

import functools
import inspect
import json
import logging
from typing import Any, Callable

import yaml
from fastmcp import FastMCP

from experiment_explorer import calculations
from experiment_explorer.config import get_cache_dir, get_s3_bucket
from experiment_explorer.experiment_scan import (
    list_files,
    read_config_yaml,
    read_metrics_csv,
    read_text_file,
    resolve_experiment_path,
    scan_experiment,
)
from experiment_explorer.paths import experiment_cache_path, validate_experiment_name
from experiment_explorer.s3 import (
    ExperimentNotFoundError,
    S3ExperimentError,
    download_prefix,
    list_folder_prefixes,
    list_objects,
)

LOGGER = logging.getLogger("experiment_explorer")


def _server_instructions() -> str:
    """Describe the server, naming the bucket this process is configured to use."""
    return (
        "Tools for listing and fetching phage-match ML experiments from S3 "
        f"(bucket: {get_s3_bucket()}), then inspecting metrics, configs, reports and "
        "other files from the local cache for plotting and analysis. "
        "Any calculation over results (averages, spreads, counts, rankings, per-bacteria or "
        "per-phage figures, metrics from predictions) must use summarize_run, compare_runs, "
        "bacteria_metrics or prediction_metrics, which read every row, or the "
        "experiment_explorer.tables loader in your own code. Never calculate from "
        "read_experiment_metrics preview rows. Always say whether a figure is from the test "
        "set or CV, which run it came from, and how many rows it used. Never name a best run: "
        "a starred run is the default answer; otherwise list the runs and let the user choose."
    )


# The configuration is read once, from the environment the server process starts with.
mcp = FastMCP(name="Experiment Explorer", instructions=_server_instructions())

# Failures caused by bad input, missing files, or S3 access. They are returned to the
# caller as {"error": ...} and logged without a traceback. Anything else is a bug.
EXPECTED_ERRORS = (ValueError, OSError, yaml.YAMLError, S3ExperimentError)


def _jsonable(value: Any) -> Any:
    """Make dict keys JSON-safe (YAML can produce date or number keys)."""
    if isinstance(value, dict):
        return {key if isinstance(key, str) else str(key): _jsonable(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(_jsonable(payload), indent=2, default=str)


def _returns_errors(tool: Callable[..., str]) -> Callable[..., str]:
    """Turn any exception raised by a tool into a JSON error response."""
    signature = inspect.signature(tool)

    @functools.wraps(tool)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return tool(*args, **kwargs)
        except Exception as exc:
            arguments = dict(signature.bind_partial(*args, **kwargs).arguments)
            if isinstance(exc, EXPECTED_ERRORS):
                level = logging.WARNING if isinstance(exc, S3ExperimentError) else logging.INFO
                LOGGER.log(level, "%s failed: %s", tool.__name__, exc)
                message = str(exc)
            else:
                LOGGER.exception("Unexpected error in %s", tool.__name__)
                message = f"Unexpected error: {exc}"
            return _json_response({**arguments, "error": message, "error_type": type(exc).__name__})

    return wrapper


@mcp.tool
@_returns_errors
def list_experiments(prefix: str = "") -> str:
    """List experiment folders in the S3 bucket.

    Args:
      prefix: Optional S3 prefix to scope the listing (e.g. a parent folder).

    Returns:
      JSON with bucket name and experiment folder names.
    """
    bucket = get_s3_bucket()
    experiments = list_folder_prefixes(bucket, prefix)
    return _json_response(
        {
            "bucket": bucket,
            "prefix": prefix,
            "experiments": experiments,
            "count": len(experiments),
            "s3_uri": f"s3://{bucket}/{prefix}".rstrip("/"),
        }
    )


@mcp.tool
@_returns_errors
def list_experiment_files(experiment_name: str, max_files: int = 200) -> str:
    """List files for one experiment prefix in S3.

    Returns at most `max_files` files. When `truncated` is true the listing is
    partial, so do not count or total files from it.

    Args:
      experiment_name: Top-level experiment folder name in the bucket.
      max_files: Maximum number of files to return (at least 1).

    Returns:
      JSON with file keys, sizes, last-modified timestamps, and `truncated`.
    """
    bucket = get_s3_bucket()
    experiment_name = validate_experiment_name(experiment_name)
    if max_files < 1:
        raise ValueError(f"max_files must be at least 1, got {max_files}")

    listed = list_objects(bucket, experiment_name, max_keys=max_files + 1)
    if not listed:
        raise ExperimentNotFoundError(bucket, experiment_name)
    objects = listed[:max_files]
    return _json_response(
        {
            "bucket": bucket,
            "experiment_name": experiment_name,
            "s3_uri": f"s3://{bucket}/{experiment_name}",
            "files": [
                {
                    "key": obj.key,
                    "relative_path": obj.key.removeprefix(f"{experiment_name}/"),
                    "size_bytes": obj.size_bytes,
                    "last_modified": obj.last_modified,
                }
                for obj in objects
            ],
            "count": len(objects),
            "truncated": len(listed) > max_files,
        }
    )


@mcp.tool
@_returns_errors
def fetch_experiment(experiment_name: str, force_refresh: bool = False) -> str:
    """Download an experiment folder from S3 into the local cache, or bring it up to date.

    Files that changed in S3 are downloaded again; unchanged files are skipped.

    Args:
      experiment_name: Top-level experiment folder name in the bucket.
      force_refresh: Re-download every file, and remove cached files that were
        deleted from S3. Without it, such files are only listed in `missing_in_s3`.

    Returns:
      JSON with the local cache path, downloaded/skipped/removed/blocked files,
      and a summary of the experiment.
    """
    experiment_name = validate_experiment_name(experiment_name)
    destination = experiment_cache_path(get_cache_dir(), experiment_name)
    result = download_prefix(
        get_s3_bucket(),
        experiment_name,
        destination,
        skip_existing=not force_refresh,
    )
    result["experiment_summary"] = scan_experiment(destination)
    return _json_response(result)


@mcp.tool
@_returns_errors
def list_cached_experiments() -> str:
    """List experiment folders available in the local cache.

    Returns:
      JSON with cached experiment names and local paths.
    """
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached = sorted(
        path.name for path in cache_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    return _json_response(
        {
            "cache_dir": str(cache_dir),
            "experiments": cached,
            "count": len(cached),
        }
    )


@mcp.tool
@_returns_errors
def list_cached_files(
    experiment_name: str, max_files: int = 1000, role: str | None = None
) -> str:
    """List every file in a cached experiment, at any depth, with its size and role.

    Each file's role is what it is for: report, metrics, predictions, config, figure,
    code, model, log, data or other. Pass `role` to list only one kind, for example
    role="report" for written reports or role="predictions". Returns at most `max_files`
    files; when `truncated` is true the listing is partial (`total_count` is the full
    number), so do not count or total from it.

    Args:
      experiment_name: Name of a locally cached experiment folder.
      max_files: Maximum number of files to return (at least 1).
      role: Optional role to filter by.

    Returns:
      JSON with each file's relative path, size and role, `count`, `total_count`, and
      `truncated`.
    """
    experiment_dir = resolve_experiment_path(get_cache_dir(), experiment_name)
    listing = list_files(experiment_dir, max_files=max_files, role=role)
    return _json_response({"experiment_name": experiment_name, "role": role, **listing})


@mcp.tool
@_returns_errors
def read_cached_file(experiment_name: str, relative_path: str, max_bytes: int = 50_000) -> str:
    """Read a Markdown, plain-text, JSON, YAML or log file from a cached experiment.

    Use it for written reports, READMEs, notes and JSON summaries; find them with
    list_cached_files. Returns at most `max_bytes` bytes. When `truncated` is true the
    content is cut short (`size_bytes` is the full size): say so, and call again with a
    larger `max_bytes` if the rest matters. Binary files such as images, pickles and
    models are refused, never opened. Use read_experiment_metrics for CSVs.

    Args:
      experiment_name: Name of a locally cached experiment folder.
      relative_path: Path to the file relative to the experiment root
        (e.g. analysis/REPORT.md).
      max_bytes: Maximum number of bytes to return (at least 1).

    Returns:
      JSON with the file's content, size_bytes, bytes_returned, and truncated.
    """
    experiment_dir = resolve_experiment_path(get_cache_dir(), experiment_name)
    return _json_response(read_text_file(experiment_dir, relative_path, max_bytes=max_bytes))


@mcp.tool
@_returns_errors
def summarize_experiment(experiment_name: str) -> str:
    """Summarize a cached experiment of any layout: its runs, reports, metrics, predictions
    and configs, and how many files of each role it holds.

    Every file is given a role (report, metrics, predictions, config, figure, code,
    model, log, data or other) by the layout rules. `docs` lists the written reports,
    top-level first: a README states the question and setup, REPORT, results and SUMMARY
    documents state the findings. Identical copies of a report are listed once (see
    `report_copies`) and empty ones are set apart in `empty_reports`. `metrics_files`
    are metric tables (CSV) for read_experiment_metrics and the calculation tools;
    `other_metrics_files` are JSON or text results for read_cached_file. `runs` lists
    every run folder with its kind (results or production) and whether it is starred.
    Use list_cached_files with a role to see the files `files_by_role` only counts.

    Args:
      experiment_name: Name of a locally cached experiment folder.

    Returns:
      JSON with `docs`, `runs`, `run_directories`, `metrics_files`,
      `other_metrics_files`, `prediction_files`, `config_files` and `files_by_role`.
    """
    experiment_dir = resolve_experiment_path(get_cache_dir(), experiment_name)
    return _json_response(scan_experiment(experiment_dir))


@mcp.tool
@_returns_errors
def read_experiment_metrics(
    experiment_name: str,
    relative_path: str,
    max_rows: int = 50,
) -> str:
    """Preview a metrics CSV from a cached experiment, to see its columns and a few rows.

    For any calculation use summarize_run, compare_runs, bacteria_metrics or
    prediction_metrics instead: they read every row. This returns only the first
    `max_rows` rows. `row_count` is the total number of
    rows in the file and `truncated` is true when the preview is partial. Do not
    calculate averages, totals, or other statistics from a truncated preview: it is
    not the whole file.

    Args:
      experiment_name: Name of a locally cached experiment folder.
      relative_path: Path to the CSV relative to the experiment root
        (e.g. results_20260108_154928/results/test_results.csv).
      max_rows: Maximum number of preview rows to return (at least 1).

    Returns:
      JSON with columns, row_count, rows_returned, truncated, and preview_rows.
    """
    experiment_dir = resolve_experiment_path(get_cache_dir(), experiment_name)
    return _json_response(read_metrics_csv(experiment_dir, relative_path, max_rows=max_rows))


@mcp.tool
@_returns_errors
def read_experiment_config(experiment_name: str, relative_path: str) -> str:
    """Read a YAML config file from a cached experiment.

    Args:
      experiment_name: Name of a locally cached experiment folder.
      relative_path: Path to the YAML file relative to the experiment root.

    Returns:
      JSON with parsed YAML content.
    """
    experiment_dir = resolve_experiment_path(get_cache_dir(), experiment_name)
    return _json_response(read_config_yaml(experiment_dir, relative_path))


@mcp.tool
@_returns_errors
def summarize_run(experiment_name: str, run: str) -> str:
    """Answer "how good is this run?" and "how stable is it?" from every row.

    Gives the run's headline metrics (the ensemble's test value, or the mean across CV
    folds for a run without test results) and their spread: mean, std, min and max
    across ensemble members (test) and across folds (CV). Each statistic has `n`, the
    number of values it used. The ensemble row is never averaged with member rows.

    Args:
      experiment_name: Name of a locally cached experiment folder.
      run: The run folder relative to the experiment root, for example
        split_known_phage/model_xgboost/results_20260108_163011_star.

    Returns:
      JSON with `headline` (partition "test" or "cv", its source and values), `test`
      and `cv` sections with row counts and spreads, and whether the run is starred.
    """
    experiment_dir = resolve_experiment_path(get_cache_dir(), experiment_name)
    return _json_response(calculations.summarize_run(experiment_dir, run))


@mcp.tool
@_returns_errors
def compare_runs(
    experiment_name: str, folder: str | None = None, runs: list[str] | None = None
) -> str:
    """Answer "how do these runs compare?" and "which run is the experiment's result?".

    Lists every run in the experiment (or in one folder, or the given runs), grouped
    by folder and then by evaluation setup (the data a run is trained and tested on),
    in time order. Each entry shows the model settings that differ and the headline
    ROC AUC and average precision. Runs with identical configs are merged. A starred
    run is the default answer. Incomplete runs are listed with a reason.

    Runs are never ranked and no best run is named: only runs in the same setup group
    are fairly comparable. Present the differences and let the user choose.

    Args:
      experiment_name: Name of a locally cached experiment folder.
      folder: Optional folder inside the experiment to limit the comparison to.
      runs: Optional list of run folders to compare instead.

    Returns:
      JSON with `folders`, each holding `setup_groups`, `incomplete_runs`, the starred
      runs and the evaluation setup its runs share.
    """
    experiment_dir = resolve_experiment_path(get_cache_dir(), experiment_name)
    return _json_response(calculations.compare_runs(experiment_dir, folder=folder, runs=runs))


@mcp.tool
@_returns_errors
def bacteria_metrics(
    experiment_name: str,
    run: str | None = None,
    partition: str | None = None,
    relative_path: str | None = None,
    bacteria: list[str] | None = None,
    metric: str | None = None,
    threshold: float | None = None,
    direction: str = "above",
    best_n: int | None = None,
    worst_n: int | None = None,
    members: str = "ensemble",
    max_groups: int = 50,
) -> str:
    """Answer per-bacteria questions from every row: "what is the ROC AUC for KLEB12?",
    "how does the model do on a typical strain?", "how many bacteria have ROC AUC above
    0.7?", "which bacteria does it get most wrong?".

    Reads the run's per-bacteria results. Test results use the ensemble's rows by
    default. In CV a bacterium can appear in several folds: its value is the mean of
    its stored per-fold values, and a lookup also lists each fold. Missing values are
    left out and each statistic reports `n`. Bacterium lookup matches IDs exactly,
    ignoring case; an unknown ID returns an error listing close IDs, never a guess.

    Args:
      experiment_name: Name of a locally cached experiment folder.
      run: Run folder relative to the experiment root (use with partition).
      partition: "test" or "cv".
      relative_path: Instead of run and partition, any per-bacteria CSV in the experiment.
      bacteria: IDs to look up.
      metric: One metric column, for example roc_auc, avg_precision or hit_5. Required
        for threshold, best_n and worst_n.
      threshold: Count bacteria whose metric is strictly above (or below) this value.
      direction: "above" or "below" for threshold.
      best_n: List the N bacteria with the highest metric.
      worst_n: List the N bacteria with the lowest metric.
      members: "ensemble" (default) or "each" (the members' rows; lookups show each one).
      max_groups: Maximum bacteria listed (at least 1); `truncated` says when cut short.

    Returns:
      JSON with row counts, `across_bacteria` stats, and `bacteria`, `threshold`,
      `best` or `worst` sections when asked for.
    """
    experiment_dir = resolve_experiment_path(get_cache_dir(), experiment_name)
    return _json_response(
        calculations.bacteria_metrics(
            experiment_dir,
            run=run,
            partition=partition,
            relative_path=relative_path,
            bacteria=bacteria,
            metric=metric,
            threshold=threshold,
            direction=direction,
            best_n=best_n,
            worst_n=worst_n,
            members=members,
            max_groups=max_groups,
        )
    )


@mcp.tool
@_returns_errors
def prediction_metrics(
    experiment_name: str,
    run: str | None = None,
    partition: str | None = None,
    relative_path: str | None = None,
    group_by: str = "none",
    members: str = "ensemble",
    max_groups: int = 50,
) -> str:
    """Answer questions that need metrics computed from predictions: "what is the ROC AUC
    per phage?", "what is the pooled AUC for each bacterium?", "what is the confusion
    matrix?", "do the ensemble members agree?".

    Computes ROC AUC, average precision, precision, recall, F1, MCC and the confusion
    matrix from every row of a predictions file. Threshold metrics use the stored
    y_pred. A group with only one class gets null ROC AUC and average precision, a
    reason, and counts toward `skipped`; it is left out of `across_groups`.

    Args:
      experiment_name: Name of a locally cached experiment folder.
      run: Run folder relative to the experiment root (use with partition).
      partition: "test" or "cv".
      relative_path: Instead of run and partition, any predictions CSV in the experiment.
      group_by: "none", "bacteria", "phage" or "fold".
      members: "ensemble" (default) or "each" (one result per member plus their spread;
        only with group_by "none").
      max_groups: Maximum groups listed (at least 1); `truncated` says when cut short.

    Returns:
      JSON with row counts and `metrics`, or `groups` with `across_groups` stats, or
      `by_member` with `member_spread`.
    """
    experiment_dir = resolve_experiment_path(get_cache_dir(), experiment_name)
    return _json_response(
        calculations.prediction_metrics(
            experiment_dir,
            run=run,
            partition=partition,
            relative_path=relative_path,
            group_by=group_by,
            members=members,
            max_groups=max_groups,
        )
    )


@mcp.resource("experiment://layout")
def experiment_layout() -> str:
    """Describe the phage-match experiment folder layouts."""
    layout = {
        "note": (
            "Experiments mix several layouts. summarize_experiment gives every file a role "
            "from the layout rules (layout_rules.yaml, plus EXPERIMENT_EXPLORER_LAYOUT_RULES "
            "if set), so it works for all of them."
        ),
        "top_level": ["README.md", "run_exp.sh", "split_*/", "model_*/"],
        "run_directory": {
            "pattern": "results_YYYYMMDD_HHMMSS/",
            "children": {
                "results/": [
                    "cv_results.csv",
                    "test_results.csv",
                    "cv_per_bacteria_results.csv",
                    "test_per_bacteria_results.csv",
                ],
                "predictions/": ["cv_predictions.csv", "test_predictions.csv"],
                "logs/": ["config.yaml"],
                "artifacts/": ["feature_order.txt"],
            },
        },
        "other_layouts": {
            "production run": (
                "production_YYYYMMDD_HHMMSS/ with results/test_metrics.csv, "
                "results/training_metrics.csv and predictions/test_predictions.csv; no CV"
            ),
            "comparison": (
                "README.md, SUMMARY.md, manifest.json, comparison_config.json, metrics/, "
                "plots/, predictions_normalized/<series>_{test,cv}.csv"
            ),
            "analysis": "README.md, REPORT.md, experiment_config.yaml, data/, outputs/, figures/",
            "GenoPHI pipeline": "training/, external_evaluation/ (metrics and predictions)",
        },
        "workflow": [
            "1. list_experiments",
            "2. fetch_experiment",
            "3. summarize_experiment",
            "4. compare_runs / summarize_run for results; bacteria_metrics and "
            "prediction_metrics for breakdowns",
            "5. read_experiment_metrics (preview only) / read_experiment_config",
            "6. list_cached_files (filter by role) / read_cached_file for reports, notes "
            "and JSON",
        ],
    }
    return json.dumps(layout, indent=2)


def main() -> None:
    """Run the MCP server over stdio."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
