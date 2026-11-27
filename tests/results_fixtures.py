"""Synthetic experiment files that hit the traps in real results files.

Every table is built as a DataFrame first, so tests can compute the expected answer
directly from the same full data with pandas or scikit-learn.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import yaml

RUN = "split_a/model_x/results_20260101_120000"
CV_ONLY_RUN = "split_a/model_x/results_20260102_120000"
MEMBERS = ("0", "1", "2")
# Numeric-looking IDs must stay text; "007" would lose its zeros as a number.
SPECIAL_BACTERIA = ("132", "1446", "007", "KLEB12", "KLEB14")
BACTERIA = SPECIAL_BACTERIA + tuple(f"B{i:03d}" for i in range(60))
PHAGES = tuple(f"P{i}" for i in range(8))
SINGLE_CLASS_BACTERIUM = "B000"


def to_csv(frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


def config_yaml(
    test_sources: tuple[str, ...] = ("our_data",),
    max_depth: int = 6,
    output_dir: str = "out",
) -> str:
    config: dict[str, Any] = {
        "data": {
            "training_sources": ["train_a"],
            "test_sources": list(test_sources),
            "split_method": "known_phage",
            "k_folds": 5,
            "encoder_path": "enc.pkl",
            "upsampling_method": "SMOTE",
        },
        "model": {"max_depth": max_depth, "learning_rate": 0.1},
        "output_dir": output_dir,
        "wandb": {"project": "x"},
        "random_seed": 42,
    }
    return yaml.safe_dump(config)


def run_results_frame(seed: int = 0) -> pd.DataFrame:
    """One row per member plus an ensemble row far from the members' values."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "roc_auc": rng.uniform(0.6, 0.7),
            "avg_precision": rng.uniform(0.4, 0.5),
            "f1": rng.uniform(0.3, 0.4),
            "tp": int(rng.integers(10, 20)),
            "model_idx": idx,
        }
        for idx in MEMBERS
    ]
    rows.append(
        {
            "roc_auc": 0.95 - seed / 100,
            "avg_precision": 0.9,
            "f1": 0.8,
            "tp": 50,
            "model_idx": "ensemble",
        }
    )
    return pd.DataFrame(rows)


def cv_metrics_frame(seed: int = 1) -> pd.DataFrame:
    """Parsed fold metrics, used to build the dict text and to compute expectations."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "roc_auc": rng.uniform(0.7, 0.9, 5),
            "auc_prc": rng.uniform(0.5, 0.7, 5),
            "tp": rng.integers(5, 30, 5),
        }
    )


def cv_results_csv(metrics: pd.DataFrame) -> str:
    rows = []
    for fold, row in metrics.iterrows():
        val = (
            f"{{'roc_auc': np.float64({float(row.roc_auc)!r}), 'auc_prc': {float(row.auc_prc)!r}, "
            f"'tp': np.int64({int(row.tp)}), 'specificity': nan}}"
        )
        rows.append(
            {
                "fold_idx": fold,
                "train_metrics": "{}",
                "val_metrics": val,
                # A multi-line cell, as in the real files.
                "val_per_bacteria": "{'per_bacteria_df':   bacteria  precision\n0  132  0.5\n1  KLEB12  0.1}",
                "train_size": 900,
                "val_size": 100,
                "model_name": "XGBoost",
            }
        )
    return to_csv(pd.DataFrame(rows))


def per_bacteria_frame(seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for bacterium in BACTERIA:
        for idx in (*MEMBERS, "ensemble"):
            rows.append(
                {
                    "bacteria": bacterium,
                    "roc_auc": rng.uniform(0, 1),
                    # Often missing, as precision_10 is in real files.
                    "precision_10": rng.uniform(0, 1) if rng.uniform() > 0.4 else np.nan,
                    "positive_count": int(rng.integers(0, 10)),
                    "model_idx": idx,
                }
            )
    return pd.DataFrame(rows)


def cv_per_bacteria_frame(seed: int = 3) -> pd.DataFrame:
    """Most bacteria appear in one fold; KLEB12 appears in three."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, bacterium in enumerate(BACTERIA):
        folds = (0, 2, 4) if bacterium == "KLEB12" else (i % 5,)
        for fold in folds:
            rows.append({"bacteria": bacterium, "roc_auc": rng.uniform(0, 1), "fold_idx": fold})
    return pd.DataFrame(rows)


def predictions_frame(seed: int = 4, label: str = "y_true") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for bacterium in BACTERIA:
        for phage in PHAGES:
            truth = 0 if bacterium == SINGLE_CLASS_BACTERIUM else int(rng.uniform() < 0.4)
            for idx in (*MEMBERS, "ensemble"):
                proba = float(np.clip(truth * 0.3 + rng.uniform(0, 0.7), 0, 1))
                rows.append(
                    {
                        "phage": phage,
                        "bacteria": bacterium,
                        label: truth,
                        "model_idx": idx,
                        "y_pred": int(proba > 0.5),
                        "y_pred_proba": proba,
                    }
                )
    return pd.DataFrame(rows)


def cv_predictions_frame(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, bacterium in enumerate(BACTERIA):
        for phage in PHAGES:
            truth = int(rng.uniform() < 0.4)
            proba = float(np.clip(truth * 0.3 + rng.uniform(0, 0.7), 0, 1))
            rows.append(
                {
                    "fold_idx": i % 5,
                    "y_true": truth,
                    "y_pred": int(proba > 0.5),
                    "y_pred_proba": proba,
                    "bacteria": bacterium,
                    "phage": phage,
                }
            )
    return pd.DataFrame(rows)


def results_experiment() -> dict[str, str]:
    """One full run (test and CV) and one CV-only run, as files for write_experiment."""
    return {
        f"{RUN}/results/test_results.csv": to_csv(run_results_frame()),
        f"{RUN}/results/cv_results.csv": cv_results_csv(cv_metrics_frame()),
        f"{RUN}/results/test_per_bacteria_results.csv": to_csv(per_bacteria_frame()),
        f"{RUN}/results/cv_per_bacteria_results.csv": to_csv(cv_per_bacteria_frame()),
        f"{RUN}/predictions/test_predictions.csv": to_csv(predictions_frame()),
        f"{RUN}/predictions/cv_predictions.csv": to_csv(cv_predictions_frame()),
        f"{RUN}/logs/config.yaml": config_yaml(),
        f"{CV_ONLY_RUN}/results/cv_results.csv": cv_results_csv(cv_metrics_frame(seed=9)),
        f"{CV_ONLY_RUN}/logs/config.yaml": config_yaml(),
        "other/predictions_normalized/model_test.csv": to_csv(
            predictions_frame(seed=6, label="interaction")
            .query("model_idx == 'ensemble'")
            .drop(columns=["model_idx", "y_pred"])
        ),
    }


def run_files(
    run: str,
    *,
    test_sources: tuple[str, ...] = ("our_data",),
    max_depth: int = 6,
    seed: int = 0,
    test: bool = True,
    cv: bool = True,
    output_dir: str = "out",
) -> dict[str, str]:
    """Files for one run, for comparison tests."""
    files = {
        f"{run}/logs/config.yaml": config_yaml(test_sources, max_depth, output_dir),
    }
    if test:
        files[f"{run}/results/test_results.csv"] = to_csv(run_results_frame(seed))
    if cv:
        files[f"{run}/results/cv_results.csv"] = cv_results_csv(cv_metrics_frame(seed))
    return files
