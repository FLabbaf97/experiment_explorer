"""Summarize, list and read the files of a cached experiment."""

from __future__ import annotations

import codecs
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from experiment_explorer.config import DOC_FILE_SUFFIXES
from experiment_explorer.layout import ROLES, find_run_folders, load_rules, walk_files
from experiment_explorer.paths import experiment_cache_path, resolve_within

LOGGER = logging.getLogger(__name__)

RESULTS_DIR_NAME = "results"
# Files read_text_file opens. Everything else (images, pickles, models, CSVs) is only listed.
TEXT_FILE_SUFFIXES = (*DOC_FILE_SUFFIXES, ".txt", ".json", ".yaml", ".yml", ".log")


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
    """Summarize a cached experiment: its runs, and its files by role (docs/adr/0003)."""
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    rules = load_rules()
    by_role: dict[str, list[tuple[str, Path]]] = {role: [] for role in ROLES}
    for relative_path, path in walk_files(experiment_dir):
        by_role[rules.role_of(relative_path)].append((relative_path, path))

    docs, report_copies, empty_reports = _distinct_reports(by_role["report"])
    runs = find_run_folders(experiment_dir, rules=rules)
    run_entries = [
        {
            "path": run.path,
            "kind": run.kind,
            "starred": run.starred,
            "has_results_folder": (experiment_dir / run.path / RESULTS_DIR_NAME).is_dir(),
        }
        for run in runs
    ]
    run_dirs = [entry["path"] for entry in run_entries if entry["has_results_folder"]]
    metrics = [relative_path for relative_path, _ in by_role["metrics"]]
    metric_tables = [path for path in metrics if path.lower().endswith(".csv")]

    return {
        "experiment_path": str(experiment_dir),
        "docs": docs,
        "run_directories": run_dirs,
        "metrics_files": metric_tables,
        "prediction_files": [relative_path for relative_path, _ in by_role["predictions"]],
        "config_files": [relative_path for relative_path, _ in by_role["config"]],
        "metrics_file_count": len(metric_tables),
        "run_count": len(run_dirs),
        "other_metrics_files": [path for path in metrics if not path.lower().endswith(".csv")],
        "runs": run_entries,
        "report_copies": report_copies,
        "empty_reports": empty_reports,
        "files_by_role": {
            role: {"count": len(files), "by_top_folder": _count_by_top_folder(files)}
            for role, files in by_role.items()
            if files
        },
        "layout_rules": list(rules.sources),
    }


def _distinct_reports(
    reports: list[tuple[str, Path]],
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Reports top-level first, each identical copy listed once, and empty ones apart.

    Returns (reports, {listed report: [its identical copies]}, empty reports).
    """
    ordered = sorted(reports, key=lambda item: (item[0].count("/"), item[0]))
    listed: list[str] = []
    copies: dict[str, list[str]] = {}
    first_with_content: dict[str, str] = {}
    empty: list[str] = []
    for relative_path, path in ordered:
        content = path.read_bytes()
        if not content.strip():
            empty.append(relative_path)
            continue
        digest = hashlib.sha256(content).hexdigest()
        if digest in first_with_content:
            copies.setdefault(first_with_content[digest], []).append(relative_path)
            continue
        first_with_content[digest] = relative_path
        listed.append(relative_path)
    return listed, copies, empty


def _count_by_top_folder(files: list[tuple[str, Path]]) -> dict[str, int]:
    """How many of the files are in each top-level folder ("." for the experiment root)."""
    counts = Counter(
        relative_path.split("/", 1)[0] if "/" in relative_path else "."
        for relative_path, _ in files
    )
    return dict(sorted(counts.items()))


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


def list_files(
    experiment_dir: Path, *, max_files: int = 1000, role: str | None = None
) -> dict[str, Any]:
    """List every file in a cached experiment, at any depth, with its size and role."""
    if max_files < 1:
        raise ValueError(f"max_files must be at least 1, got {max_files}")
    if role is not None and role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}, got {role!r}")
    rules = load_rules()
    files = [
        (relative_path, path, rules.role_of(relative_path))
        for relative_path, path in walk_files(experiment_dir)
    ]
    if role is not None:
        files = [item for item in files if item[2] == role]
    listed = files[:max_files]
    return {
        "files": [
            {"relative_path": relative_path, "size_bytes": path.stat().st_size, "role": file_role}
            for relative_path, path, file_role in listed
        ],
        "count": len(listed),
        "total_count": len(files),
        "truncated": len(listed) < len(files),
    }


def read_text_file(
    experiment_dir: Path, relative_path: str, *, max_bytes: int = 50_000
) -> dict[str, Any]:
    """Read a text file (see TEXT_FILE_SUFFIXES) from a cached experiment, up to max_bytes."""
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
