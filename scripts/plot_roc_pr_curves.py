"""Plot ROC and PR curves for a cached phage-match experiment."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from experiment_explorer.config import get_cache_dir

LOGGER = logging.getLogger(__name__)

RUN_DIR_PATTERN = re.compile(r"_(\d{8}_\d{6})(?:_star)?$")
NORMALIZED_FILE_PATTERN = re.compile(r"(?P<series>.+)_(?P<kind>test|cv)\.csv")
PREDICTION_KINDS = ("test", "cv")
LABEL_COLUMNS = ("y_true", "interaction")
ROOT_GROUP = "root"

MODEL_LABELS = {
    "model_xgboost": "XGBoost baseline",
    "model_xgboost_avoid_overfit": "XGBoost avoid overfit",
    "model_xgboost_middle_land": "XGBoost middle ground",
}

MODEL_COLORS = {
    "model_xgboost": "#1f77b4",
    "model_xgboost_avoid_overfit": "#ff7f0e",
    "model_xgboost_middle_land": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-name",
        default="2026-01-08_baseline_run",
        help="Cached experiment folder name.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Local experiment cache directory (defaults to EXPERIMENT_EXPLORER_CACHE_DIR, "
            "or the server's default cache)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saved figures (defaults to <experiment>/plots).",
    )
    return parser.parse_args()


def _run_sort_key(run_dir_name: str) -> tuple[int, str]:
    """Prefer starred runs, then latest timestamp."""
    starred = 1 if run_dir_name.endswith("_star") else 0
    match = RUN_DIR_PATTERN.search(run_dir_name)
    timestamp = match.group(1) if match else run_dir_name
    return starred, timestamp


def _run_group_and_series(relative_parts: tuple[str, ...]) -> tuple[str, str]:
    """Name a run's plot group and curve from the folders above it."""
    parents = relative_parts[:-1]
    if not parents:
        return ROOT_GROUP, "run"
    if len(parents) == 1:
        return parents[0], parents[0]
    return "/".join(parents[:-1]), parents[-1]


def select_latest_runs(experiment_dir: Path) -> dict[tuple[str, str], Path]:
    """Pick one run directory per (group, series), ignoring root runs if nested ones exist."""
    selected: dict[tuple[str, str], Path] = {}

    for kind in PREDICTION_KINDS:
        for prediction_path in experiment_dir.rglob(f"predictions/{kind}_predictions.csv"):
            run_dir = prediction_path.parent.parent
            key = _run_group_and_series(run_dir.relative_to(experiment_dir).parts)
            current = selected.get(key)
            if current is None or _run_sort_key(run_dir.name) > _run_sort_key(current.name):
                selected[key] = run_dir

    if any(group != ROOT_GROUP for group, _ in selected):
        selected = {key: run for key, run in selected.items() if key[0] != ROOT_GROUP}
    return selected


def discover_curve_sources(experiment_dir: Path) -> dict[tuple[str, str], dict[str, Path]]:
    """Map (group, kind) to the prediction file behind each curve."""
    sources: dict[tuple[str, str], dict[str, Path]] = {}

    for (group, series), run_dir in select_latest_runs(experiment_dir).items():
        for kind in PREDICTION_KINDS:
            prediction_path = run_dir / "predictions" / f"{kind}_predictions.csv"
            if prediction_path.exists():
                sources.setdefault((group, kind), {})[series] = prediction_path

    for prediction_path in experiment_dir.rglob("predictions_normalized/*.csv"):
        match = NORMALIZED_FILE_PATTERN.fullmatch(prediction_path.name)
        if match is None:
            continue
        parent = prediction_path.parent.parent.relative_to(experiment_dir)
        group = parent.as_posix() if parent.parts else ROOT_GROUP
        sources.setdefault((group, match["kind"]), {})[match["series"]] = prediction_path

    return sources


def load_prediction_frame(prediction_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(prediction_path)

    if "model_idx" in frame.columns and "ensemble" in frame["model_idx"].astype(str).unique():
        frame = frame[frame["model_idx"].astype(str) == "ensemble"].copy()

    label_col = next((col for col in LABEL_COLUMNS if col in frame.columns), None)
    if label_col is None:
        raise ValueError(f"Expected one of {LABEL_COLUMNS} in {prediction_path}")

    if "y_pred_proba" not in frame.columns:
        raise ValueError(f"Expected probability column in {prediction_path}")

    frame = frame.rename(columns={label_col: "y_true"})
    return frame[["y_true", "y_pred_proba"]].dropna()


def compute_curve_metrics(y_true: pd.Series, y_pred_proba: pd.Series) -> dict[str, float | list[float]]:
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "roc_auc": float(auc(fpr, tpr)),
        "average_precision": float(average_precision_score(y_true, y_pred_proba)),
        "baseline_precision": float(y_true.mean()),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def plot_split_curves(
    *,
    experiment_name: str,
    split_name: str,
    prediction_kind: str,
    run_map: dict[str, Path],
    output_path: Path,
) -> None:
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(13, 5))

    for model_name, prediction_path in sorted(run_map.items()):
        frame = load_prediction_frame(prediction_path)
        metrics = compute_curve_metrics(frame["y_true"], frame["y_pred_proba"])
        label = MODEL_LABELS.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, None)

        roc_ax.plot(
            metrics["fpr"],
            metrics["tpr"],
            color=color,
            lw=2,
            label=f"{label} (AUC={metrics['roc_auc']:.3f})",
        )
        pr_ax.plot(
            metrics["recall"],
            metrics["precision"],
            color=color,
            lw=2,
            label=f"{label} (AP={metrics['average_precision']:.3f})",
        )

    prediction_label = "test" if prediction_kind == "test" else "cross-validation"
    if split_name == ROOT_GROUP:
        figure_label = prediction_label
    else:
        figure_label = f"{split_name.replace('_', ' ')} ({prediction_label})"
    roc_ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random")
    roc_ax.set_title(f"ROC — {figure_label}")
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1.02)
    roc_ax.grid(True, alpha=0.3)
    roc_ax.legend(loc="lower right", fontsize=9)

    if run_map:
        sample_frame = load_prediction_frame(next(iter(run_map.values())))
        baseline = float(sample_frame["y_true"].mean())
        pr_ax.axhline(
            y=baseline,
            color="gray",
            linestyle="--",
            lw=1,
            alpha=0.8,
            label=f"Baseline precision={baseline:.3f}",
        )

    pr_ax.set_title(f"Precision-Recall — {figure_label}")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_xlim(0, 1)
    pr_ax.set_ylim(0, 1.02)
    pr_ax.grid(True, alpha=0.3)
    pr_ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        f"{experiment_name} — {figure_label}",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", output_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    cache_dir = args.cache_dir or get_cache_dir()
    experiment_dir = cache_dir / args.experiment_name
    if not experiment_dir.exists():
        LOGGER.error("Experiment not cached at %s. Run fetch_experiment first.", experiment_dir)
        return 1

    output_dir = args.output_dir or (experiment_dir / "plots")
    curve_sources = discover_curve_sources(experiment_dir)
    if not curve_sources:
        LOGGER.error("No prediction files found under %s", experiment_dir)
        return 1

    saved_names: list[str] = []
    for (group, prediction_kind), series_paths in sorted(curve_sources.items()):
        output_name = f"{group.replace('/', '__')}_{prediction_kind}_roc_pr.png"
        plot_split_curves(
            experiment_name=args.experiment_name,
            split_name=group,
            prediction_kind=prediction_kind,
            run_map=series_paths,
            output_path=output_dir / output_name,
        )
        saved_names.append(output_name)

    print(f"Saved ROC/PR plots to {output_dir}")
    for name in saved_names:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
