"""Shared fixtures: a fake S3 bucket, an isolated cache, and a tool caller."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import boto3
import pytest
from fastmcp import Client
from moto import mock_aws

from experiment_explorer.server import mcp

BUCKET = "test-experiments"
RUN = "split_a/model_x/results_20260101_120000"

STANDARD_EXPERIMENT: dict[str, str] = {
    "README.md": "# Example experiment\n",
    f"{RUN}/results/test_results.csv": "metric,value\nroc_auc,0.9\npr_auc,0.8\n",
    f"{RUN}/results/cv_results.csv": "fold,roc_auc\n0,0.88\n1,0.91\n",
    f"{RUN}/predictions/test_predictions.csv": "interaction,y_pred_proba\n1,0.9\n0,0.2\n",
    f"{RUN}/logs/config.yaml": "model: xgboost\nseed: 1\n",
}


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep tests away from real AWS credentials, config, and the real cache."""
    for name in ("AWS_PROFILE", "AWS_REGION", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws-credentials"))
    monkeypatch.setenv("EXPERIMENT_EXPLORER_S3_BUCKET", BUCKET)
    monkeypatch.setenv("EXPERIMENT_EXPLORER_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    path = tmp_path / "cache"
    path.mkdir()
    return path


@pytest.fixture
def s3() -> Any:
    """A fake S3 bucket; the server's real boto3 code runs against it."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


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


@pytest.fixture
def call_tool() -> Callable[..., dict[str, Any]]:
    """Call a server tool the way an MCP client does and decode its JSON reply."""

    def call(name: str, **arguments: Any) -> dict[str, Any]:
        async def run() -> str:
            async with Client(mcp) as client:
                result = await client.call_tool(name, arguments)
                return result.content[0].text

        return json.loads(asyncio.run(run()))

    return call
