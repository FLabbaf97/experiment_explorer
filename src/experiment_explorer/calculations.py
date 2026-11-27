"""Calculations behind the results tools. Every figure is computed from every row."""

from __future__ import annotations

import difflib
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from experiment_explorer import tables
from experiment_explorer.paths import resolve_within

GROUP_BY_COLUMNS = {"bacteria": "bacteria", "phage": "phage", "fold": "fold_idx"}
AVERAGE_PRECISION_COLUMNS = ("avg_precision", "average_precision", "auc_prc", "pr_auc")
CV_BOOKKEEPING_COLUMNS = frozenset({"train_size", "val_size", "fold", "val_feature_noise_fraction"})
PREDICTION_SUMMARY_METRICS = ("roc_auc", "average_precision", "precision", "recall", "f1", "mcc")
SINGLE_CLASS = "single class: ROC AUC and average precision need both positives and negatives"
NO_Y_PRED = "the file has no y_pred column, so threshold metrics cannot be computed"
NO_HEADLINE = (
    "no single figure: the test file has several rows and no ensemble row, and the run "
    "has no CV results; use summarize_run to see its rows"
)


# --- Small helpers ---------------------------------------------------------------


def number(value: Any) -> float | int | None:
    """A JSON-safe number: numpy types become Python ones, NaN and inf become None."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def describe(values: pd.Series | list[Any]) -> dict[str, Any]:
    """Mean, std, min, max and median of the non-missing values, with their count ``n``."""
    series = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").astype(float)
    series = series[np.isfinite(series)]
    n = len(series)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "median": None}
    return {
        "n": n,
        "mean": number(series.mean()),
        "std": number(series.std()) if n > 1 else None,
        "min": number(series.min()),
        "max": number(series.max()),
        "median": number(series.median()),
    }


def _check_limit(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")


def _relative(experiment_dir: Path, path: Path) -> str:
    return path.relative_to(experiment_dir.resolve()).as_posix()


def _infer_partition(path: Path) -> str | None:
    name = path.name.lower()
    for partition in tables.PARTITIONS:
        if name.startswith(f"{partition}_") or f"_{partition}." in name:
            return partition
    return None


def _resolve_file(
    experiment_dir: Path,
    run: str | None,
    partition: str | None,
    relative_path: str | None,
    kind: str,
) -> tuple[Path, str, str | None]:
    """Find the file a tool reads: from run + partition, or from an explicit path."""
    if relative_path and (run or partition):
        raise ValueError("Pass either run and partition, or relative_path, not both.")
    if relative_path:
        path = resolve_within(experiment_dir, relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {relative_path}")
        return path, _relative(experiment_dir, path), _infer_partition(path)
    if not run or not partition:
        raise ValueError("Pass run and partition ('test' or 'cv'), or relative_path.")
    path = tables.run_file(experiment_dir, run, partition, kind)
    return path, _relative(experiment_dir, path), partition


# --- Run headlines ---------------------------------------------------------------


def _test_values(frame: pd.DataFrame) -> tuple[dict[str, Any] | None, str | None]:
    """The run's own test values: the ensemble row, or the only row of a single model."""
    metrics = tables.metric_columns(frame)
    if "is_ensemble" in frame.columns:
        ensemble = frame[frame["is_ensemble"]]
        if len(ensemble) == 1:
            return {m: number(ensemble[m].iloc[0]) for m in metrics}, "the ensemble's test value"
        return None, None
    if len(frame) == 1:
        return {m: number(frame[m].iloc[0]) for m in metrics}, "the single model's test value"
    return None, None


def _cv_fold_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Map metric name to column for CV fold metrics."""
    parsed = tables.fold_metric_columns(frame, "val")
    if parsed:
        return {column.removeprefix("val_"): column for column in parsed}
    return {
        column: column
        for column in tables.metric_columns(frame)
        if column not in CV_BOOKKEEPING_COLUMNS
    }


def _headline_pair(values: dict[str, Any]) -> dict[str, Any]:
    ap_column = next((c for c in AVERAGE_PRECISION_COLUMNS if values.get(c) is not None), None)
    return {
        "roc_auc": values.get("roc_auc"),
        "average_precision": values.get(ap_column) if ap_column else None,
        "average_precision_column": ap_column,
    }


def _summarize_test(path: Path, experiment_dir: Path) -> dict[str, Any]:
    frame = tables.load_table(path)
    metrics = tables.metric_columns(frame)
    values, source = _test_values(frame)
    summary: dict[str, Any] = {"file": _relative(experiment_dir, path), "row_count": len(frame)}
    if "is_ensemble" in frame.columns:
        members = frame[~frame["is_ensemble"]]
        summary["ensemble"] = values
        summary["member_count"] = len(members)
        summary["members"] = {m: describe(members[m]) for m in metrics}
    elif values is not None:
        summary["single_model"] = values
    else:
        summary["rows"] = {m: describe(frame[m]) for m in metrics}
    summary["_values"], summary["_source"] = values, source
    return summary


def _summarize_cv(path: Path, experiment_dir: Path) -> dict[str, Any]:
    frame = tables.load_table(path)
    columns = _cv_fold_columns(frame)
    fold_count = frame["fold_idx"].nunique() if "fold_idx" in frame.columns else len(frame)
    return {
        "file": _relative(experiment_dir, path),
        "row_count": len(frame),
        "fold_count": int(fold_count),
        "folds": {name: describe(frame[column]) for name, column in columns.items()},
    }


def _run_headlines(run_dir: Path, experiment_dir: Path, status: tables.RunStatus) -> dict[str, Any]:
    """Headline ROC AUC and average precision from test and from CV, where available."""
    headlines: dict[str, Any] = {"test": None, "cv": None}
    if status.has_test:
        test = _summarize_test(run_dir / "results" / "test_results.csv", experiment_dir)
        if test["_values"] is not None:
            headlines["test"] = _headline_pair(test["_values"])
    if status.has_cv:
        cv = _summarize_cv(run_dir / "results" / "cv_results.csv", experiment_dir)
        headlines["cv"] = _headline_pair({name: s["mean"] for name, s in cv["folds"].items()})
    return headlines


# --- summarize_run ---------------------------------------------------------------


def summarize_run(experiment_dir: Path, run: str) -> dict[str, Any]:
    """Headline metrics and their spread (across members and folds) for one run."""
    run_dir = tables.resolve_run(experiment_dir, run)
    status = tables.run_status(run_dir)
    if not status.has_results:
        raise ValueError(f"Run {run} is incomplete: {status.reason}")

    result: dict[str, Any] = {"run": run, "starred": tables.is_starred(run)}
    test = (
        _summarize_test(run_dir / "results" / "test_results.csv", experiment_dir)
        if status.has_test
        else None
    )
    cv = (
        _summarize_cv(run_dir / "results" / "cv_results.csv", experiment_dir)
        if status.has_cv
        else None
    )

    if test is not None and test["_values"] is not None:
        headline = {"partition": "test", "source": test["_source"], "metrics": test["_values"]}
    elif cv is not None:
        headline = {
            "partition": "cv",
            "source": "the mean across CV folds",
            "metrics": {name: s["mean"] for name, s in cv["folds"].items()},
        }
    else:
        headline = None

    if test is not None:
        test.pop("_values")
        test.pop("_source")
    result["headline"] = headline
    result["test"] = test
    result["cv"] = cv
    if headline is None:
        result["note"] = "The test file has no ensemble row and several rows; see test.rows."
    return result


# --- compare_runs ----------------------------------------------------------------


def _parent(run: str) -> str:
    parent = PurePosixPath(run).parent.as_posix()
    return "" if parent == "." else parent


def _run_record(experiment_dir: Path, run: str) -> dict[str, Any]:
    run_dir = experiment_dir / run
    status = tables.run_status(run_dir)
    record: dict[str, Any] = {"run": run, "starred": tables.is_starred(run), "status": status}
    if not status.has_results:
        return record
    config = tables.load_run_config(run_dir)
    setup, settings = tables.split_config(config) if config is not None else (None, None)
    record.update(
        setup=setup,
        settings=settings,
        headlines=_run_headlines(run_dir, experiment_dir, status),
    )
    return record


def _own_headline(headlines: dict[str, Any]) -> dict[str, Any]:
    """The run's own headline: test if it has one, else CV, else none with a reason."""
    for partition in ("test", "cv"):
        if headlines[partition]:
            return {"partition": partition, **headlines[partition]}
    return {
        "partition": None,
        **_headline_pair({}),
        "reason": NO_HEADLINE,
    }


def _entry_results(records: list[dict[str, Any]], partition: str | None) -> dict[str, Any]:
    by_run: dict[str, Any] = {}
    for record in records:
        if partition is not None:
            by_run[record["run"]] = {"partition": partition, **record["headlines"][partition]}
        else:
            by_run[record["run"]] = _own_headline(record["headlines"])
    distinct = {tables.canonical(value) for value in by_run.values()}
    if len(distinct) == 1:
        return {"results": next(iter(by_run.values())), "repeats_agree": len(by_run) > 1 or None}
    return {"results_by_run": by_run, "repeats_agree": False}


def _setup_group(
    records: list[dict[str, Any]], setup_keys: list[str], settings_keys: list[str]
) -> dict[str, Any]:
    setup = records[0]["setup"]
    if all(r["headlines"]["test"] for r in records):
        partition: str | None = "test"
    elif all(r["headlines"]["cv"] for r in records):
        partition = "cv"
    else:
        partition = None

    entries: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["settings"] is None:
            key = f"no-config:{record['run']}"
        else:
            key = tables.canonical(record["settings"])
        entries.setdefault(key, []).append(record)

    group_entries = []
    for members in entries.values():
        settings = members[0]["settings"]
        entry: dict[str, Any] = {
            "runs": [r["run"] for r in members],
            "starred": any(r["starred"] for r in members),
            "config_found": settings is not None,
            "model_settings": (
                {key: settings.get(key) for key in settings_keys} if settings is not None else None
            ),
        }
        entry.update(_entry_results(members, partition))
        if entry["repeats_agree"] is None:
            entry.pop("repeats_agree")
        group_entries.append(entry)

    group: dict[str, Any] = {
        "evaluation_setup": (
            {key: setup.get(key) for key in setup_keys} if setup is not None else None
        ),
        "partition": partition,
        "run_count": len(records),
        "entries": group_entries,
    }
    if setup is None:
        group["note"] = "These runs have no config.yaml, so their evaluation setup is unknown."
    if partition is None:
        group["note"] = (
            "Not every run here has the same kind of headline (test or CV), so no single "
            "figure is shared. Each result says which it is; do not compare them directly."
        )
    return group


def _compare_folder(folder: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [r for r in records if r["status"].has_results]
    incomplete = [
        {"run": r["run"], "starred": r["starred"], "reason": r["status"].reason}
        for r in records
        if not r["status"].has_results
    ]
    setups = [r["setup"] for r in complete if r["setup"] is not None]
    setup_keys = tables.differing_keys(setups)
    settings_keys = tables.differing_keys(
        [r["settings"] for r in complete if r["settings"] is not None]
    )
    shared_setup = (
        {key: value for key, value in setups[0].items() if key not in setup_keys} if setups else {}
    )

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in complete:
        key = tables.canonical(record["setup"])
        groups.setdefault(key, []).append(record)

    starred = [r["run"] for r in records if r["starred"]]
    result: dict[str, Any] = {
        "folder": folder,
        "run_count": len(records),
        "starred_runs": starred,
        "shared_evaluation_setup": shared_setup,
        "setup_groups": [
            _setup_group(group, setup_keys, settings_keys) for group in groups.values()
        ],
        "incomplete_runs": incomplete,
    }
    if starred:
        others = len(records) - len(starred)
        result["default_answer"] = starred[0] if len(starred) == 1 else starred
        result["note"] = (
            f"{', '.join(starred)} is starred, so it is this folder's result. "
            f"{others} other run(s) are listed for comparison."
        )
    return result


def compare_runs(
    experiment_dir: Path, folder: str | None = None, runs: list[str] | None = None
) -> dict[str, Any]:
    """Every run, grouped by folder and evaluation setup, in time order. Never ranked."""
    if folder and runs:
        raise ValueError("Pass either folder or runs, not both.")
    if runs:
        for run in runs:
            tables.resolve_run(experiment_dir, run)
        run_list = sorted(set(runs))
    else:
        run_list = tables.find_runs(experiment_dir, folder)
    if not run_list:
        where = f" under {folder}" if folder else ""
        raise FileNotFoundError(
            f"No runs (results_* folders) found{where}. This experiment may use another "
            "layout: use summarize_experiment to find its files, and pass them to "
            "bacteria_metrics or prediction_metrics as relative_path."
        )

    by_folder: dict[str, list[str]] = defaultdict(list)
    for run in run_list:
        by_folder[_parent(run)].append(run)

    folders = []
    for name in sorted(by_folder):
        ordered = sorted(by_folder[name], key=lambda r: (tables.run_timestamp(r), r))
        folders.append(_compare_folder(name, [_run_record(experiment_dir, r) for r in ordered]))

    return {
        "run_count": len(run_list),
        "folder_count": len(folders),
        "folders": folders,
        "how_to_read": (
            "Runs are listed in time order and never ranked. Only runs in the same setup group "
            "share an evaluation setup and can be compared fairly; present the differences "
            "and let the user choose."
        ),
    }


# --- bacteria_metrics ------------------------------------------------------------


def _match_bacteria(requested: list[str], available: pd.Index, file: str) -> list[str]:
    lookup = {str(item).lower(): str(item) for item in available}
    matched = []
    for name in requested:
        found = lookup.get(str(name).strip().lower())
        if found is None:
            close = difflib.get_close_matches(str(name).lower(), list(lookup), n=5, cutoff=0.5)
            suggestions = [lookup[c] for c in close]
            raise ValueError(
                f"No bacterium {name!r} in {file}. Close IDs: {suggestions or 'none'}. "
                "Ask the user which one they meant."
            )
        matched.append(found)
    return matched


def bacteria_metrics(
    experiment_dir: Path,
    *,
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
) -> dict[str, Any]:
    """Per-bacteria results: lookups, stats across bacteria, threshold counts and rankings."""
    _check_limit("max_groups", max_groups)
    _check_limit("best_n", best_n)
    _check_limit("worst_n", worst_n)
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")
    if (threshold is not None or best_n or worst_n) and not metric:
        raise ValueError("threshold, best_n and worst_n need a metric.")

    path, file, file_partition = _resolve_file(
        experiment_dir, run, partition, relative_path, "per_bacteria"
    )
    frame = tables.load_table(path)
    if "bacteria" not in frame.columns:
        raise ValueError(f"{file} has no bacteria column.")
    selected = tables.select_members(frame, members)
    metrics = tables.metric_columns(selected)
    if metric and metric not in metrics:
        raise ValueError(f"Unknown metric {metric!r}. Available: {metrics}")
    shown = [metric] if metric else metrics

    if "fold_idx" in selected.columns:
        replicate, replicate_name = "fold_idx", "fold"
        aggregation = "each bacterium's value is the mean of its stored per-fold values"
    elif members == "each" and "model_idx" in selected.columns:
        replicate, replicate_name = "model_idx", "member"
        aggregation = "each bacterium's value is the mean across members"
    else:
        replicate, replicate_name = None, None
        aggregation = "one row per bacterium"
    per_bacterium = selected.groupby("bacteria", sort=True)[metrics].mean()

    result: dict[str, Any] = {
        "file": file,
        "partition": file_partition,
        "row_count": len(frame),
        "rows_used": len(selected),
        "members": members if "model_idx" in frame.columns else None,
        "bacteria_count": len(per_bacterium),
        "aggregation": aggregation,
    }

    if bacteria:
        matched = _match_bacteria(bacteria, per_bacterium.index, file)
        found = []
        for name in matched:
            entry: dict[str, Any] = {
                "bacteria": name,
                "values": {m: number(per_bacterium.loc[name, m]) for m in shown},
            }
            if replicate:
                rows = selected[selected["bacteria"] == name]
                entry[f"by_{replicate_name}"] = [
                    {replicate_name: row[replicate], **{m: number(row[m]) for m in shown}}
                    for _, row in rows.iterrows()
                ]
                entry["spread"] = {m: describe(rows[m]) for m in shown}
            found.append(entry)
        result["bacteria"] = found[:max_groups]
        result["group_count"] = len(found)
        result["truncated"] = len(found) > max_groups
        return result

    result["across_bacteria"] = {m: describe(per_bacterium[m]) for m in shown}
    if metric and replicate == "model_idx":
        by_member = selected.groupby("model_idx")[metric]
        result["member_means"] = {
            str(idx): {"mean": number(values.mean()), "n": int(values.notna().sum())}
            for idx, values in by_member
        }

    if metric is not None:
        values = per_bacterium[metric].dropna()
        values = values[np.isfinite(values.astype(float))]
        if threshold is not None:
            passing = (
                values[values > threshold] if direction == "above" else values[values < threshold]
            )
            passing = passing.sort_values(ascending=direction == "below")
            result["threshold"] = {
                "metric": metric,
                "rule": f"{metric} {'>' if direction == 'above' else '<'} {threshold}",
                "count": len(passing),
                "n": len(values),
                "fraction": number(len(passing) / len(values)) if len(values) else None,
                "bacteria": [
                    {"bacteria": k, metric: number(v)} for k, v in passing.head(max_groups).items()
                ],
                "truncated": len(passing) > max_groups,
            }
        for label, count, ascending in (("best", best_n, False), ("worst", worst_n, True)):
            if count:
                ranked = values.sort_values(ascending=ascending).head(min(count, max_groups))
                result[label] = {
                    "metric": metric,
                    "order": "highest first" if label == "best" else "lowest first",
                    "bacteria": [{"bacteria": k, metric: number(v)} for k, v in ranked.items()],
                    "n": len(values),
                    "truncated": count > max_groups and len(values) > max_groups,
                }
    return result


# --- prediction_metrics ----------------------------------------------------------


def binary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Metrics for one set of predictions with ``y_true``, ``y_pred_proba`` and maybe ``y_pred``."""
    y_true = frame["y_true"].astype(int).to_numpy()
    proba = frame["y_pred_proba"].astype(float).to_numpy()
    n = len(y_true)
    positives = int(y_true.sum())
    metrics: dict[str, Any] = {
        "n": n,
        "positives": positives,
        "positive_rate": number(positives / n) if n else None,
    }
    if len(set(y_true.tolist())) < 2:
        metrics.update(roc_auc=None, average_precision=None, reason=SINGLE_CLASS)
    else:
        metrics["roc_auc"] = number(roc_auc_score(y_true, proba))
        metrics["average_precision"] = number(average_precision_score(y_true, proba))

    if "y_pred" not in frame.columns:
        metrics.update(precision=None, recall=None, f1=None, mcc=None, confusion_matrix=None)
        metrics["threshold_metrics_reason"] = NO_Y_PRED
        return metrics
    y_pred = frame["y_pred"].astype(int).to_numpy()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics.update(
        precision=number(tp / (tp + fp)) if tp + fp else None,
        recall=number(tp / (tp + fn)) if tp + fn else None,
        f1=number(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else None,
        mcc=number((tp * tn - fp * fn) / mcc_denominator) if mcc_denominator else None,
        confusion_matrix={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    )
    return metrics


def prediction_metrics(
    experiment_dir: Path,
    *,
    run: str | None = None,
    partition: str | None = None,
    relative_path: str | None = None,
    group_by: str = "none",
    members: str = "ensemble",
    max_groups: int = 50,
) -> dict[str, Any]:
    """Metrics computed from every row of a predictions file, overall or per group."""
    _check_limit("max_groups", max_groups)
    if group_by != "none" and group_by not in GROUP_BY_COLUMNS:
        raise ValueError(f"group_by must be 'none' or one of {list(GROUP_BY_COLUMNS)}")

    path, file, file_partition = _resolve_file(
        experiment_dir, run, partition, relative_path, "predictions"
    )
    frame = tables.load_table(path)
    for column in ("y_true", "y_pred_proba"):
        if column not in frame.columns:
            raise ValueError(f"{file} has no {column} column (or interaction for labels).")
    has_members = "model_idx" in frame.columns
    if members == "each" and has_members and group_by != "none":
        raise ValueError("members='each' works with group_by='none' only.")
    group_column = GROUP_BY_COLUMNS.get(group_by)
    if group_column and group_column not in frame.columns:
        raise ValueError(f"{file} has no {group_column} column to group by {group_by}.")

    selected = tables.select_members(frame, members)
    usable = selected.dropna(subset=["y_true", "y_pred_proba"])
    labels = set(pd.unique(usable["y_true"].astype(float)))
    if not labels <= {0.0, 1.0}:
        raise ValueError(f"{file} has labels other than 0 and 1: {sorted(labels)[:5]}")

    result: dict[str, Any] = {
        "file": file,
        "partition": file_partition,
        "row_count": len(frame),
        "rows_used": len(usable),
        "rows_dropped_missing": len(selected) - len(usable),
        "members": members if has_members else None,
        "threshold_metrics_from": "stored y_pred" if "y_pred" in frame.columns else None,
    }

    if members == "each" and has_members:
        per_member = [
            {"member": str(idx), **binary_metrics(group)}
            for idx, group in usable.groupby("model_idx", sort=False)
        ]
        per_member.sort(key=lambda m: (len(m["member"]), m["member"]))
        result["by_member"] = per_member
        result["member_spread"] = {
            m: describe([item[m] for item in per_member]) for m in PREDICTION_SUMMARY_METRICS
        }
        return result

    if group_column is None:
        result["metrics"] = binary_metrics(usable)
        return result

    groups = [
        (key, binary_metrics(group)) for key, group in usable.groupby(group_column, sort=True)
    ]
    result["group_by"] = group_by
    result["group_count"] = len(groups)
    result["skipped"] = sum(1 for _, m in groups if m["roc_auc"] is None)
    if result["skipped"]:
        result["skipped_reason"] = SINGLE_CLASS
    result["across_groups"] = {
        m: describe([metrics[m] for _, metrics in groups]) for m in PREDICTION_SUMMARY_METRICS
    }
    result["groups"] = [
        {group_by: number(key) if group_by == "fold" else str(key), **metrics}
        for key, metrics in groups[:max_groups]
    ]
    result["truncated"] = len(groups) > max_groups
    return result
