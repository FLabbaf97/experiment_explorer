"""Test data and builders shared across test modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

BUCKET = "test-experiments"
RUN = "split_a/model_x/results_20260101_120000"

STANDARD_EXPERIMENT: dict[str, str] = {
    "README.md": "# Example experiment\n",
    f"{RUN}/results/test_results.csv": "metric,value\nroc_auc,0.9\npr_auc,0.8\n",
    f"{RUN}/results/cv_results.csv": "fold,roc_auc\n0,0.88\n1,0.91\n",
    f"{RUN}/predictions/test_predictions.csv": "interaction,y_pred_proba\n1,0.9\n0,0.2\n",
    f"{RUN}/logs/config.yaml": "model: xgboost\nseed: 1\n",
}


def put_experiment(s3: Any, name: str, files: dict[str, str] | None = None) -> None:
    for relative_path, body in (files or STANDARD_EXPERIMENT).items():
        s3.put_object(Bucket=BUCKET, Key=f"{name}/{relative_path}", Body=body.encode())


def write_experiment(cache_dir: Path, name: str, files: dict[str, str] | None = None) -> Path:
    root = cache_dir / name
    for relative_path, body in (files or STANDARD_EXPERIMENT).items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root
