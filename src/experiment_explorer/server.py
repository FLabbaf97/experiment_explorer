"""FastMCP server exposing experiment exploration tools."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp import FastMCP

from experiment_explorer.config import get_cache_dir, get_s3_bucket
from experiment_explorer.experiment_scan import (
    read_config_yaml,
    read_metrics_csv,
    resolve_experiment_path,
    scan_experiment,
)
from experiment_explorer.s3 import (
    S3ExperimentError,
    download_prefix,
    list_folder_prefixes,
    list_objects,
)

LOGGER = logging.getLogger("experiment_explorer")
mcp = FastMCP(
    name="Experiment Explorer",
    instructions=(
        "Tools for listing and fetching phage-match ML experiments from S3 "
        "(bucket: phage-match-ml-experiments), then inspecting metrics and configs "
        "from the local cache for plotting and analysis."
    ),
)


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


@mcp.tool
def list_experiments(prefix: str = "") -> str:
    """List experiment folders in the S3 bucket.

    Args:
      prefix: Optional S3 prefix to scope the listing (e.g. a parent folder).

    Returns:
      JSON with bucket name and experiment folder names.
    """
    bucket = get_s3_bucket()
    try:
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
    except S3ExperimentError as exc:
        LOGGER.exception("Failed to list experiments", extra={"bucket": bucket, "prefix": prefix})
        return _json_response({"error": str(exc), "bucket": bucket, "prefix": prefix})


@mcp.tool
def list_experiment_files(experiment_name: str, max_files: int = 200) -> str:
    """List files for one experiment prefix in S3.

    Args:
      experiment_name: Top-level experiment folder name in the bucket.
      max_files: Maximum number of files to return.

    Returns:
      JSON with file keys, sizes, and last-modified timestamps.
    """
    bucket = get_s3_bucket()
    try:
        objects = list_objects(bucket, experiment_name, max_keys=max_files)
        return _json_response(
            {
                "bucket": bucket,
                "experiment_name": experiment_name,
                "s3_uri": f"s3://{bucket}/{experiment_name}",
                "files": [
                    {
                        "key": obj.key,
                        "relative_path": obj.key.removeprefix(f"{experiment_name.rstrip('/')}/"),
                        "size_bytes": obj.size_bytes,
                        "last_modified": obj.last_modified,
                    }
                    for obj in objects
                ],
                "count": len(objects),
                "truncated": len(objects) >= max_files,
            }
        )
    except S3ExperimentError as exc:
        LOGGER.exception(
            "Failed to list experiment files",
            extra={"bucket": bucket, "experiment_name": experiment_name},
        )
        return _json_response({"error": str(exc), "experiment_name": experiment_name})


@mcp.tool
def fetch_experiment(experiment_name: str, force_refresh: bool = False) -> str:
    """Download an experiment folder from S3 into the local cache.

    Args:
      experiment_name: Top-level experiment folder name in the bucket.
      force_refresh: Re-download files even if they already exist locally.

    Returns:
      JSON with local cache path and download statistics.
    """
    bucket = get_s3_bucket()
    cache_dir = get_cache_dir()
    destination = cache_dir / experiment_name

    try:
        result = download_prefix(
            bucket,
            experiment_name,
            destination,
            skip_existing=not force_refresh,
        )
        summary = scan_experiment(destination)
        result["experiment_summary"] = summary
        return _json_response(result)
    except S3ExperimentError as exc:
        LOGGER.exception(
            "Failed to fetch experiment",
            extra={"bucket": bucket, "experiment_name": experiment_name},
        )
        return _json_response({"error": str(exc), "experiment_name": experiment_name})
    except Exception as exc:
        LOGGER.exception(
            "Unexpected error while fetching experiment",
            extra={"experiment_name": experiment_name},
        )
        return _json_response({"error": str(exc), "experiment_name": experiment_name})


@mcp.tool
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
def summarize_experiment(experiment_name: str) -> str:
    """Summarize a cached experiment: docs, runs, metrics, and config files.

    Args:
      experiment_name: Name of a locally cached experiment folder.

    Returns:
      JSON summary of experiment artifacts for analysis and plotting.
    """
    cache_dir = get_cache_dir()
    try:
        experiment_dir = resolve_experiment_path(cache_dir, experiment_name)
        summary = scan_experiment(experiment_dir)
        return _json_response(summary)
    except FileNotFoundError as exc:
        return _json_response({"error": str(exc), "experiment_name": experiment_name})


@mcp.tool
def read_experiment_metrics(
    experiment_name: str,
    relative_path: str,
    max_rows: int = 50,
) -> str:
    """Read a metrics CSV from a cached experiment.

    Args:
      experiment_name: Name of a locally cached experiment folder.
      relative_path: Path to the CSV relative to the experiment root
        (e.g. results_20260108_154928/results/test_results.csv).
      max_rows: Maximum number of preview rows to return.

    Returns:
      JSON with columns, row count, and preview rows.
    """
    cache_dir = get_cache_dir()
    try:
        experiment_dir = resolve_experiment_path(cache_dir, experiment_name)
        metrics = read_metrics_csv(experiment_dir, relative_path, max_rows=max_rows)
        return _json_response(metrics)
    except (FileNotFoundError, ValueError) as exc:
        return _json_response(
            {
                "error": str(exc),
                "experiment_name": experiment_name,
                "relative_path": relative_path,
            }
        )


@mcp.tool
def read_experiment_config(experiment_name: str, relative_path: str) -> str:
    """Read a YAML config file from a cached experiment.

    Args:
      experiment_name: Name of a locally cached experiment folder.
      relative_path: Path to the YAML file relative to the experiment root.

    Returns:
      JSON with parsed YAML content.
    """
    cache_dir = get_cache_dir()
    try:
        experiment_dir = resolve_experiment_path(cache_dir, experiment_name)
        config = read_config_yaml(experiment_dir, relative_path)
        return _json_response(config)
    except (FileNotFoundError, ValueError) as exc:
        return _json_response(
            {
                "error": str(exc),
                "experiment_name": experiment_name,
                "relative_path": relative_path,
            }
        )


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
