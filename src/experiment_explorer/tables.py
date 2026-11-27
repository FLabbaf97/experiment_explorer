"""Load experiment results and predictions correctly, from every row.

This is the shared loader behind the calculation tools, and it is meant to be
imported directly when a question needs a calculation no tool offers::

    from experiment_explorer.tables import load_table, find_runs

It handles the traps in these files: ensemble rows mixed with member rows, CV fold
metrics stored as dict text, two names for the label column, and numeric IDs.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from experiment_explorer.paths import resolve_within

ENSEMBLE = "ensemble"
PARTITIONS = ("test", "cv")
KINDS = ("results", "per_bacteria", "predictions")
RUN_PREFIX = "results_"
STAR_SUFFIX = "_star"

ID_COLUMNS = ("bacteria", "phage", "model_idx")
LABEL_COLUMNS = ("y_true", "interaction")
CV_DICT_COLUMNS = ("train_metrics", "val_metrics")
DROPPED_COLUMNS = ("val_per_bacteria",)
NON_METRIC_COLUMNS = frozenset(
    {"model_idx", "fold_idx", "is_ensemble", "bacteria", "phage", "group", "source"}
)

# Settings that decide which data a run is trained and tested on. Runs are only
# fairly comparable when all of these match (see GLOSSARY.md, "Evaluation setup").
EVALUATION_SETUP_KEYS = (
    "data.training_sources",
    "data.test_sources",
    "data.split_method",
    "data.k_folds",
    "data.test_size",
    "data.group_similar_bacteria_for_test",
    "data.group_similar_bacteria_for_k_fold",
    "random_seed",
    "data.random_seed",
    "data.validation_mode",
    "fold_limit",
    "data.binarize_interactions",
    "data_path",
    "test_data_path",
)

# Settings that never affect results; left out when comparing configs.
BOOKKEEPING_KEY_PATTERNS = (
    "output_dir",
    "experiment_name",
    "wandb",
    "wandb.*",
    "verbose",
    "log_level",
    "save_*",
    "data.encoder_path",
    "data.scaler_path",
    "data.feature_order_path",
    "data.dtype_path",
)

_RUN_TIMESTAMP = re.compile(r"(\d{8}_\d{6})")
_NUMPY_CONSTANT = re.compile(r"\bnp\.(nan|inf)\b")
_NUMPY_CALL = re.compile(r"\bnp\.\w+\(\s*([^()]*?)\s*\)")
_BARE_NAN = re.compile(r"(?<![\w.'\"])nan(?![\w'\"])")
_BARE_INF = re.compile(r"(?<![\w.'\"])inf(?![\w'\"])")


# --- Reading files ---------------------------------------------------------------


def parse_metric_dict(text: Any) -> dict[str, float]:
    """Parse metric dict text such as ``"{'roc_auc': np.float64(0.78), 'tp': np.int64(25)}"``.

    Only literal syntax is accepted; nothing is evaluated. Nested values (for example
    per-k dicts or tables) are skipped. Raises ValueError on malformed text.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return {}
    if not isinstance(text, str):
        raise ValueError(f"Expected metric dict text, got {type(text).__name__}")
    cleaned = _NUMPY_CONSTANT.sub(r"\1", text)
    cleaned = _NUMPY_CALL.sub(r"\1", cleaned)
    cleaned = _BARE_NAN.sub("None", cleaned)
    cleaned = _BARE_INF.sub("1e999", cleaned)
    try:
        value = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ValueError(f"Could not parse metric dict text: {text[:80]!r}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Metric text is not a dict: {text[:80]!r}")

    metrics: dict[str, float] = {}
    for key, item in value.items():
        if item is None:
            metrics[str(key)] = math.nan
        elif isinstance(item, bool):
            metrics[str(key)] = float(item)
        elif isinstance(item, (int, float)):
            metrics[str(key)] = float(item)
    return metrics


def load_table(path: Path | str) -> pd.DataFrame:
    """Load a results or predictions CSV from every row, in a clean form.

    - ``bacteria``, ``phage`` and ``model_idx`` are read as text.
    - ``is_ensemble`` is added when the file has ``model_idx``.
    - CV dict-text columns (``train_metrics``, ``val_metrics``) become one column per
      metric, prefixed ``train_`` / ``val_``; ``val_per_bacteria`` is dropped.
    - Prediction files get one label column, ``y_true`` (from ``y_true`` or ``interaction``).
    """
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV files can be loaded as tables, got: {path.name}")
    try:
        frame = pd.read_csv(path, dtype={column: str for column in ID_COLUMNS})
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV {path.name}: {exc}") from exc

    if "model_idx" in frame.columns:
        frame["is_ensemble"] = frame["model_idx"].str.strip().str.lower() == ENSEMBLE

    fold_metrics: dict[str, list[str]] = {}
    for column in CV_DICT_COLUMNS:
        if column in frame.columns:
            prefix = column.split("_")[0]
            parsed = pd.DataFrame([parse_metric_dict(text) for text in frame[column]])
            parsed = parsed.add_prefix(f"{prefix}_")
            parsed.index = frame.index
            frame = pd.concat([frame.drop(columns=[column]), parsed], axis=1)
            fold_metrics[prefix] = list(parsed.columns)
    frame = frame.drop(columns=[c for c in DROPPED_COLUMNS if c in frame.columns])

    if "y_pred_proba" in frame.columns and "y_true" not in frame.columns:
        if "interaction" in frame.columns:
            frame = frame.rename(columns={"interaction": "y_true"})
    frame.attrs["fold_metric_columns"] = fold_metrics
    return frame


def fold_metric_columns(frame: pd.DataFrame, which: str = "val") -> list[str]:
    """Columns parsed from a CV results file's ``val_metrics`` (or ``train_metrics``) text."""
    return list(frame.attrs.get("fold_metric_columns", {}).get(which, []))


def metric_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric columns that hold metrics (not IDs, flags or fold numbers)."""
    return [
        column
        for column in frame.columns
        if column not in NON_METRIC_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]


def select_members(frame: pd.DataFrame, members: str) -> pd.DataFrame:
    """Keep the ensemble's rows (``"ensemble"``) or the members' rows (``"each"``).

    Files without ``model_idx`` hold one model's rows and are returned unchanged.
    """
    if members not in ("ensemble", "each"):
        raise ValueError(f"members must be 'ensemble' or 'each', got {members!r}")
    if "is_ensemble" not in frame.columns:
        return frame
    if members == "ensemble":
        selected = frame[frame["is_ensemble"]]
        if selected.empty:
            raise ValueError("This file has no ensemble rows; use members='each'.")
        return selected
    return frame[~frame["is_ensemble"]]


# --- Runs ------------------------------------------------------------------------


@dataclass(frozen=True)
class RunStatus:
    has_results: bool
    reason: str | None
    has_test: bool
    has_cv: bool


def find_runs(experiment_dir: Path, folder: str | None = None) -> list[str]:
    """Relative paths of every run folder (``results_*``), including incomplete runs."""
    root = resolve_within(experiment_dir, folder) if folder else experiment_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found in experiment: {folder}")
    base = experiment_dir.resolve()
    runs = [
        path.relative_to(base).as_posix()
        for path in root.rglob(f"{RUN_PREFIX}*")
        if path.is_dir() and not path.is_symlink()
    ]
    return sorted(runs)


def resolve_run(experiment_dir: Path, run: str) -> Path:
    """Resolve a run path inside the experiment, or raise."""
    run_dir = resolve_within(experiment_dir, run)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run}")
    if not run_dir.name.startswith(RUN_PREFIX):
        raise ValueError(
            f"{run} is not a run folder (results_*). Pass the file as relative_path instead."
        )
    return run_dir


def run_file(experiment_dir: Path, run: str, partition: str, kind: str) -> Path:
    """Path of one of a run's standard files, for example its test predictions."""
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    run_dir = resolve_run(experiment_dir, run)
    names = {
        "results": f"results/{partition}_results.csv",
        "per_bacteria": f"results/{partition}_per_bacteria_results.csv",
        "predictions": f"predictions/{partition}_predictions.csv",
    }
    path = run_dir / names[kind]
    if not path.is_file():
        raise FileNotFoundError(f"Run {run} has no {names[kind]}")
    return path


def is_starred(run: str) -> bool:
    return Path(run).name.endswith(STAR_SUFFIX)


def run_timestamp(run: str) -> str:
    """The run's timestamp (``YYYYMMDD_HHMMSS``) for time ordering, or its name."""
    name = Path(run).name
    match = _RUN_TIMESTAMP.search(name)
    return match.group(1) if match else name


def _results_file_state(path: Path) -> str | None:
    """None if the file is usable; otherwise why not."""
    if not path.is_file():
        return "missing"
    try:
        frame = load_table(path)
    except ValueError:
        return f"{path.name} won't parse"
    if frame.empty:
        return f"{path.name} is empty"
    return None


def run_status(run_dir: Path) -> RunStatus:
    """Whether a run has results: CV or test results that parse and have rows."""
    results_dir = run_dir / "results"
    if not results_dir.is_dir():
        return RunStatus(False, "no results folder", False, False)
    test_state = _results_file_state(results_dir / "test_results.csv")
    cv_state = _results_file_state(results_dir / "cv_results.csv")
    has_test, has_cv = test_state is None, cv_state is None
    if has_test or has_cv:
        return RunStatus(True, None, has_test, has_cv)
    problems = [s for s in (test_state, cv_state) if s and s != "missing"]
    reason = "; ".join(problems) if problems else "no test or CV results file"
    return RunStatus(False, reason, False, False)


# --- Configs ---------------------------------------------------------------------


def load_run_config(run_dir: Path) -> dict[str, Any] | None:
    """The run's ``config.yaml`` (``logs/`` first), or None if it has none."""
    candidates = [run_dir / "logs" / "config.yaml", run_dir / "config.yaml"]
    candidates += sorted(run_dir.rglob("config.yaml"))
    for path in candidates:
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                content = yaml.safe_load(handle)
            return content if isinstance(content, dict) else {}
    return None


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested settings into dotted keys such as ``data.test_sources``."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            flat.update(flatten_config(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def is_bookkeeping_key(key: str) -> bool:
    return any(fnmatch.fnmatchcase(key, pattern) for pattern in BOOKKEEPING_KEY_PATTERNS)


def split_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a config into (evaluation setup, model settings), dropping bookkeeping keys."""
    flat = flatten_config(config)
    setup = {key: flat.get(key) for key in EVALUATION_SETUP_KEYS if key in flat}
    settings = {
        key: value
        for key, value in flat.items()
        if key not in EVALUATION_SETUP_KEYS and not is_bookkeeping_key(key)
    }
    return setup, settings


def differing_keys(settings: list[dict[str, Any]]) -> list[str]:
    """Keys whose values are not the same across all of the given flat configs."""
    keys = sorted({key for item in settings for key in item})
    return [key for key in keys if len({canonical(item.get(key)) for item in settings}) > 1]


def canonical(value: Any) -> str:
    """A stable text form of a value, for grouping and equality checks."""
    return json.dumps(value, sort_keys=True, default=str)
