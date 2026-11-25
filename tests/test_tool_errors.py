"""Every tool failure returns {"error": ...} instead of crashing (issue #2)."""

import asyncio
import json
import logging
import os

import pytest
from fastmcp import Client

from experiment_explorer.server import mcp
from helpers import put_experiment, write_experiment


@pytest.mark.parametrize(
    ("files", "relative_path"),
    [
        ({"bad.yaml": "a: [1\n"}, "bad.yaml"),
        ({"folder.yaml/inner.txt": "x"}, "folder.yaml"),
    ],
    ids=["malformed-yaml", "folder-instead-of-file"],
)
def test_bad_config_files_return_errors(call_tool, cache_dir, files, relative_path):
    write_experiment(cache_dir, "exp", files)

    result = call_tool("read_experiment_config", experiment_name="exp", relative_path=relative_path)

    assert result["error"]
    assert result["relative_path"] == relative_path


def test_yaml_with_date_keys_is_returned(call_tool, cache_dir):
    write_experiment(cache_dir, "exp", {"c.yaml": "2026-01-08: baseline\nruns: 3\n"})

    result = call_tool("read_experiment_config", experiment_name="exp", relative_path="c.yaml")

    assert result["content"] == {"2026-01-08": "baseline", "runs": 3}


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX non-root")
def test_unreadable_file_returns_error(call_tool, cache_dir):
    path = write_experiment(cache_dir, "exp", {"c.yaml": "a: 1\n"}) / "c.yaml"
    path.chmod(0)
    try:
        result = call_tool("read_experiment_config", experiment_name="exp", relative_path="c.yaml")
    finally:
        path.chmod(0o644)

    assert result["error"]


def test_very_long_experiment_name_returns_error(call_tool, cache_dir):
    assert call_tool("summarize_experiment", experiment_name="a" * 300)["error"]


def test_cache_path_that_is_a_file_returns_error(call_tool, cache_dir):
    cache_dir.rmdir()
    cache_dir.write_text("not a folder")

    assert call_tool("list_cached_experiments")["error"]


def test_s3_failure_returns_error(call_tool, s3, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_EXPLORER_S3_BUCKET", "bucket-that-does-not-exist")

    assert "NoSuchBucket" in call_tool("list_experiments")["error"]


def test_unexpected_errors_are_returned_and_logged_with_traceback(call_tool, monkeypatch, caplog):
    monkeypatch.setenv("EXPERIMENT_EXPLORER_CACHE_DIR", "~user-that-does-not-exist-xyz/cache")

    with caplog.at_level(logging.ERROR):
        result = call_tool("summarize_experiment", experiment_name="exp")

    assert result["error"]
    assert any(record.exc_info for record in caplog.records if record.levelno >= logging.ERROR)


def test_expected_errors_are_not_logged_as_crashes(call_tool, s3, cache_dir, caplog):
    put_experiment(s3, "exp")

    with caplog.at_level(logging.WARNING):
        call_tool("summarize_experiment", experiment_name="missing")
        call_tool("fetch_experiment", experiment_name="missing")
        call_tool("read_experiment_metrics", experiment_name="exp", relative_path="x.csv")

    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


def test_malformed_metrics_csv_is_an_expected_error(call_tool, cache_dir, caplog):
    write_experiment(cache_dir, "exp", {"bad.csv": 'a,b\n1,"unterminated\n'})

    with caplog.at_level(logging.WARNING):
        result = call_tool(
            "read_experiment_metrics", experiment_name="exp", relative_path="bad.csv"
        )

    assert result["error"]
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


def test_server_keeps_answering_after_a_failure(cache_dir):
    write_experiment(cache_dir, "exp", {"bad.yaml": "a: [1\n", "good.yaml": "a: 1\n"})

    async def two_calls() -> list[dict]:
        async with Client(mcp) as client:
            replies = []
            for path in ("bad.yaml", "good.yaml"):
                result = await client.call_tool(
                    "read_experiment_config", {"experiment_name": "exp", "relative_path": path}
                )
                replies.append(json.loads(result.content[0].text))
            return replies

    bad, good = asyncio.run(two_calls())

    assert bad["error"]
    assert good["content"] == {"a": 1}
