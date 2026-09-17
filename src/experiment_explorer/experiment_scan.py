"""Scan local experiment folders for metrics, docs, and run artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from experiment_explorer.config import DOC_FILE_NAMES, METRICS_FILE_PATTERNS

LOGGER = logging.getLogger(__name__)

RESULTS_DIR_NAME = "results"
PREDICTIONS_DIR_NAME = "predictions"
LOGS_DIR_NAME = "logs"
CONFIG_FILE_NAME = "config.yaml"


def resolve_experiment_path(cache_dir: Path, experiment_name: str) -> Path:
    """Resolve an experiment name to a local directory path."""
    candidate = cache_dir / experiment_name
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
    """Load a metrics CSV from a cached experiment."""
    metrics_path = (experiment_dir / relative_path).resolve()
    experiment_root = experiment_dir.resolve()

    if not str(metrics_path).startswith(str(experiment_root)):
        raise ValueError(f"Refusing to read path outside experiment directory: {relative_path}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {relative_path}")

    if metrics_path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV metrics files are supported, got: {relative_path}")

    try:
        frame = pd.read_csv(metrics_path)
    except Exception as exc:
        LOGGER.exception("Failed to read metrics CSV", extra={"path": str(metrics_path)})
        raise ValueError(f"Failed to parse CSV at {relative_path}: {exc}") from exc

    preview = frame.head(max_rows)
    return {
        "path": relative_path,
        "columns": list(frame.columns),
        "row_count": int(len(frame)),
        "preview_rows": json.loads(preview.to_json(orient="records")),
    }


def read_config_yaml(experiment_dir: Path, relative_path: str) -> dict[str, Any]:
    """Load a YAML config file from a cached experiment."""
    config_path = (experiment_dir / relative_path).resolve()
    experiment_root = experiment_dir.resolve()

    if not str(config_path).startswith(str(experiment_root)):
        raise ValueError(f"Refusing to read path outside experiment directory: {relative_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {relative_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)

    return {
        "path": relative_path,
        "content": content,
    }


def _find_docs(experiment_dir: Path) -> list[str]:
    docs: list[str] = []
    for doc_name in DOC_FILE_NAMES:
        if (experiment_dir / doc_name).exists():
            docs.append(doc_name)
    return docs


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
