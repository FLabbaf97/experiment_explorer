"""Configuration for the experiment explorer MCP server."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_S3_BUCKET = "phage-match-ml-experiments"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
METRICS_FILE_PATTERNS = (
    "cv_results.csv",
    "test_results.csv",
    "cv_per_bacteria_results.csv",
    "test_per_bacteria_results.csv",
)
DOC_FILE_NAMES = ("README.md", "results.md")


def get_s3_bucket() -> str:
    """Return the S3 bucket used for ML experiments."""
    return os.environ.get("EXPERIMENT_EXPLORER_S3_BUCKET", DEFAULT_S3_BUCKET)


def get_cache_dir() -> Path:
    """Return the local directory where fetched experiments are stored."""
    configured = os.environ.get("EXPERIMENT_EXPLORER_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_CACHE_DIR.resolve()


def get_aws_region() -> str | None:
    """Return optional AWS region override."""
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
