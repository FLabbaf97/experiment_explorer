"""FastMCP server exposing experiment exploration tools."""

from __future__ import annotations

import functools
import inspect
import json
import logging
from typing import Any, Callable

import yaml
from fastmcp import FastMCP

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
        "other files from the local cache for plotting and analysis."
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
def list_cached_files(experiment_name: str, max_files: int = 1000) -> str:
    """List every file in a cached experiment, at any depth, with its size.

    Use it to find written reports, READMEs, notes and JSON summaries anywhere in the
    experiment. Returns at most `max_files` files; when `truncated` is true the listing
    is partial (`total_count` is the full number), so do not count or total from it.

    Args:
      experiment_name: Name of a locally cached experiment folder.
      max_files: Maximum number of files to return (at least 1).

    Returns:
      JSON with each file's relative path and size, `count`, `total_count`, and
      `truncated`.
    """
    experiment_dir = resolve_experiment_path(get_cache_dir(), experiment_name)
    return _json_response(
        {"experiment_name": experiment_name, **list_files(experiment_dir, max_files=max_files)}
    )


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
    """Summarize a cached experiment: docs, runs, metrics, and config files.

    Args:
      experiment_name: Name of a locally cached experiment folder.

    Returns:
      JSON summary of experiment artifacts for analysis and plotting.
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
    """Preview a metrics CSV from a cached experiment.

    This returns only the first `max_rows` rows. `row_count` is the total number of
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


@mcp.resource("experiment://layout")
def experiment_layout() -> str:
    """Describe the typical phage-match experiment folder layout."""
    layout = {
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
        "workflow": [
            "1. list_experiments",
            "2. fetch_experiment",
            "3. summarize_experiment",
            "4. read_experiment_metrics / read_experiment_config",
            "5. list_cached_files / read_cached_file for reports, notes and JSON",
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
