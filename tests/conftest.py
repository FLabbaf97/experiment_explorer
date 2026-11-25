"""Shared fixtures: a fake S3 bucket, an isolated cache, and a tool caller."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

import boto3
import pytest
from fastmcp import Client
from moto import mock_aws

from experiment_explorer.server import mcp
from helpers import BUCKET


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """Keep every test away from real AWS, real credentials, and the real cache."""
    for name in (
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_SESSION_TOKEN",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws-credentials"))
    monkeypatch.setenv("EXPERIMENT_EXPLORER_S3_BUCKET", BUCKET)
    monkeypatch.setenv("EXPERIMENT_EXPLORER_CACHE_DIR", str(tmp_path / "cache"))
    with mock_aws():
        yield


@pytest.fixture
def cache_dir() -> Path:
    """The isolated cache folder the server is configured to use."""
    path = Path(os.environ["EXPERIMENT_EXPLORER_CACHE_DIR"])
    path.mkdir()
    return path


@pytest.fixture
def s3() -> Any:
    """A fake S3 bucket; the server's real boto3 code runs against it."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)
    return client


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
