"""Scan local experiment folders for metrics, docs, and run artifacts."""

from __future__ import annotations

import codecs
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from experiment_explorer.config import DOC_FILE_SUFFIXES, METRICS_FILE_PATTERNS
from experiment_explorer.paths import experiment_cache_path, resolve_within
from experiment_explorer.s3 import MANIFEST_NAME

LOGGER = logging.getLogger(__name__)

RESULTS_DIR_NAME = "results"
PREDICTIONS_DIR_NAME = "predictions"
CONFIG_FILE_NAME = "config.yaml"
# Files read_text_file opens. Everything else (images, pickles, models, CSVs) is only listed.
TEXT_FILE_SUFFIXES = (".md", ".markdown", ".txt", ".json", ".yaml", ".yml", ".log")


def resolve_experiment_path(cache_dir: Path, experiment_name: str) -> Path:
    """Resolve an experiment name to a local directory path."""
    candidate = experiment_cache_path(cache_dir, experiment_name)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Experiment '{experiment_name}' is not cached locally at {candidate}. "
        "Use fetch_experiment first."
    )


def scan_experiment(experiment_dir: Path) -> dict[str, Any]:
    """Build a structured summary of a local experiment directory."""
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    docs = _find_docs(experiment_dir)
    run_dirs = _find_run_directories(experiment_dir)
    metrics_files = _find_metrics_files(experiment_dir)
    prediction_files = _find_prediction_files(experiment_dir)
    config_files = _find_config_files(experiment_dir)

    return {
        "experiment_path": str(experiment_dir),
        "docs": docs,
        "run_directories": run_dirs,
        "metrics_files": metrics_files,
        "prediction_files": prediction_files,
        "config_files": config_files,
        "metrics_file_count": len(metrics_files),
        "run_count": len(run_dirs),
    }


def read_metrics_csv(
    experiment_dir: Path, relative_path: str, *, max_rows: int = 50
) -> dict[str, Any]:
    """Load a metrics CSV from a cached experiment and return a preview of its rows."""
    if max_rows < 1:
        raise ValueError(f"max_rows must be at least 1, got {max_rows}")
    metrics_path = resolve_within(experiment_dir, relative_path)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {relative_path}")

    if metrics_path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV metrics files are supported, got: {relative_path}")

    try:
        frame = pd.read_csv(metrics_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV at {relative_path}: {exc}") from exc

    preview = frame.head(max_rows)
    return {
        "path": relative_path,
        "columns": list(frame.columns),
        "row_count": len(frame),
        "rows_returned": len(preview),
        "truncated": len(preview) < len(frame),
        "preview_rows": json.loads(preview.to_json(orient="records")),
    }


def list_files(experiment_dir: Path, *, max_files: int = 1000) -> dict[str, Any]:
    """List every file in a cached experiment, at any depth, with its size."""
    if max_files < 1:
        raise ValueError(f"max_files must be at least 1, got {max_files}")
    files = _walk_files(experiment_dir)
    listed = files[:max_files]
    return {
        "files": [
            {"relative_path": relative_path, "size_bytes": path.stat().st_size}
            for relative_path, path in listed
        ],
        "count": len(listed),
        "total_count": len(files),
        "truncated": len(listed) < len(files),
    }


def read_text_file(
    experiment_dir: Path, relative_path: str, *, max_bytes: int = 50_000
) -> dict[str, Any]:
    """Read a Markdown, plain-text or JSON file from a cached experiment, up to max_bytes."""
    if max_bytes < 1:
        raise ValueError(f"max_bytes must be at least 1, got {max_bytes}")
    path = resolve_within(experiment_dir, relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {relative_path}")
    if path.suffix.lower() not in TEXT_FILE_SUFFIXES:
        raise ValueError(_not_text_message(relative_path))

    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if b"\0" in data:
        raise ValueError(_not_text_message(relative_path))
    truncated = len(data) > max_bytes
    data = data[:max_bytes]
    # A cut can split a multi-byte character; drop that partial character, not more.
    content = codecs.getincrementaldecoder("utf-8")(errors="replace").decode(
        data, final=not truncated
    )
    return {
        "path": relative_path,
        "size_bytes": path.stat().st_size,
        "bytes_returned": len(data),
        "truncated": truncated,
        "content": content,
    }


def _not_text_message(relative_path: str) -> str:
    if relative_path.lower().endswith(".csv"):
        return f"{relative_path} is a CSV; read it with read_experiment_metrics instead."
    return (
        f"{relative_path} is not a text file this tool reads "
        f"({', '.join(TEXT_FILE_SUFFIXES)}). Binary files are listed but never opened."
    )


def _walk_files(experiment_dir: Path) -> list[tuple[str, Path]]:
    """Return (relative path, path) for every file inside the experiment, sorted.

    Symlinked folders are not followed, and symlinked files that point outside the
    experiment are left out. The cache manifest is bookkeeping, not experiment data.
    """
    root = experiment_dir.resolve()
    found: list[tuple[str, Path]] = []
    for folder, _, names in os.walk(root):
        for name in names:
            path = Path(folder) / name
            relative_path = path.relative_to(root).as_posix()
            if relative_path == MANIFEST_NAME:
                continue
            if not path.is_file() or not path.resolve().is_relative_to(root):
                continue
            found.append((relative_path, path))
    return sorted(found)


def read_config_yaml(experiment_dir: Path, relative_path: str) -> dict[str, Any]:
    """Load a YAML config file from a cached experiment."""
    config_path = resolve_within(experiment_dir, relative_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {relative_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)

    return {
        "path": relative_path,
        "content": content,
    }


def _find_docs(experiment_dir: Path) -> list[str]:
    """Markdown reports and READMEs at any depth, top-level ones first."""
    docs = [
        relative_path
        for relative_path, path in _walk_files(experiment_dir)
        if path.suffix.lower() in DOC_FILE_SUFFIXES
    ]
    return sorted(docs, key=lambda relative_path: (relative_path.count("/"), relative_path))


def _find_run_directories(experiment_dir: Path) -> list[str]:
    run_dirs: list[str] = []
    for path in sorted(experiment_dir.rglob("results_*")):
        if path.is_dir() and (path / RESULTS_DIR_NAME).exists():
            run_dirs.append(str(path.relative_to(experiment_dir)))
    return run_dirs


def _find_metrics_files(experiment_dir: Path) -> list[str]:
    metrics: list[str] = []
    for pattern in METRICS_FILE_PATTERNS:
        for path in experiment_dir.rglob(pattern):
            if path.is_file():
                metrics.append(str(path.relative_to(experiment_dir)))
    return sorted(set(metrics))


def _find_prediction_files(experiment_dir: Path) -> list[str]:
    predictions: list[str] = []
    for path in experiment_dir.rglob(f"{PREDICTIONS_DIR_NAME}/*.csv"):
        if path.is_file():
            predictions.append(str(path.relative_to(experiment_dir)))
    return sorted(predictions)


def _find_config_files(experiment_dir: Path) -> list[str]:
    configs: list[str] = []
    for path in experiment_dir.rglob(CONFIG_FILE_NAME):
        if path.is_file():
            configs.append(str(path.relative_to(experiment_dir)))
    return sorted(configs)
