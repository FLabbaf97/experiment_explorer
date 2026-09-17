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
from sklearn.metrics import auc, precision_recall_curve, roc_curve

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
RUN_DIR_PATTERN = re.compile(r"results_(\d{8}_\d{6})(?:_star)?$")

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
        default=DEFAULT_CACHE_DIR,
        help="Local experiment cache directory.",
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


def select_latest_runs(experiment_dir: Path) -> dict[tuple[str, str], Path]:
    """Pick one run directory per (split, model) group."""
    selected: dict[tuple[str, str], Path] = {}

    for prediction_path in experiment_dir.rglob("predictions/test_predictions.csv"):
        run_dir = prediction_path.parent.parent
        relative_parts = run_dir.relative_to(experiment_dir).parts

        if len(relative_parts) == 1:
            split_name = "root"
            model_name = "model_xgboost"
        elif relative_parts[0].startswith("split_") and relative_parts[1].startswith("model_"):
            split_name = relative_parts[0]
            model_name = relative_parts[1]
        else:
            continue

        key = (split_name, model_name)
        current = selected.get(key)
        if current is None or _run_sort_key(run_dir.name) > _run_sort_key(current.name):
            selected[key] = run_dir

    return selected


def load_prediction_frame(run_dir: Path, prediction_kind: str) -> pd.DataFrame:
    prediction_path = run_dir / "predictions" / f"{prediction_kind}_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {prediction_path}")

    frame = pd.read_csv(prediction_path)

    if prediction_kind == "test":
        if "ensemble" in frame["model_idx"].astype(str).unique():
            frame = frame[frame["model_idx"].astype(str) == "ensemble"].copy()
        label_col = "interaction"
    else:
        label_col = "y_true"

    if label_col not in frame.columns:
        raise ValueError(f"Expected label column '{label_col}' in {prediction_path}")

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
        "pr_auc": float(auc(recall, precision)),
        "baseline_precision": float(y_true.mean()),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def plot_split_curves(
    *,
    split_name: str,
    prediction_kind: str,
    run_map: dict[str, Path],
    output_path: Path,
) -> None:
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(13, 5))

    for model_name, run_dir in sorted(run_map.items()):
        frame = load_prediction_frame(run_dir, prediction_kind)
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
            label=f"{label} (AUC={metrics['pr_auc']:.3f})",
        )

    split_label = split_name.replace("_", " ")
    prediction_label = "test" if prediction_kind == "test" else "cross-validation"
    roc_ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random")
    roc_ax.set_title(f"ROC — {split_label} ({prediction_label})")
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1.02)
    roc_ax.grid(True, alpha=0.3)
    roc_ax.legend(loc="lower right", fontsize=9)

    if run_map:
        sample_frame = load_prediction_frame(next(iter(run_map.values())), prediction_kind)
        baseline = float(sample_frame["y_true"].mean())
        pr_ax.axhline(
            y=baseline,
            color="gray",
            linestyle="--",
            lw=1,
            alpha=0.8,
            label=f"Baseline precision={baseline:.3f}",
        )

    pr_ax.set_title(f"Precision-Recall — {split_label} ({prediction_label})")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_xlim(0, 1)
    pr_ax.set_ylim(0, 1.02)
    pr_ax.grid(True, alpha=0.3)
    pr_ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        f"2026-01-08 baseline run — {split_label} ({prediction_label})",
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

    experiment_dir = args.cache_dir / args.experiment_name
    if not experiment_dir.exists():
        LOGGER.error("Experiment not cached at %s. Run fetch_experiment first.", experiment_dir)
        return 1

    output_dir = args.output_dir or (experiment_dir / "plots")
    selected_runs = select_latest_runs(experiment_dir)
    if not selected_runs:
        LOGGER.error("No test prediction files found under %s", experiment_dir)
        return 1

    runs_by_split: dict[str, dict[str, Path]] = {}
    for (split_name, model_name), run_dir in selected_runs.items():
        runs_by_split.setdefault(split_name, {})[model_name] = run_dir

    for split_name, model_runs in sorted(runs_by_split.items()):
        if split_name == "root":
            continue
        for prediction_kind in ("test", "cv"):
            output_name = f"{split_name}_{prediction_kind}_roc_pr.png"
            plot_split_curves(
                split_name=split_name,
                prediction_kind=prediction_kind,
                run_map=model_runs,
                output_path=output_dir / output_name,
            )

    print(f"Saved ROC/PR plots to {output_dir}")
    for plot_path in sorted(output_dir.glob("*.png")):
        print(f"  - {plot_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
