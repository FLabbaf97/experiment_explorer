"""Every partial result must say it is partial (issue #5)."""

import asyncio

import pytest
from fastmcp import Client

from experiment_explorer.server import mcp
from helpers import put_experiment, write_experiment

THREE_ROWS = {"m.csv": "a\n1\n2\n3\n"}


@pytest.mark.parametrize(
    ("max_rows", "returned", "truncated"),
    [(2, 2, True), (3, 3, False), (10, 3, False)],
)
def test_metric_preview_says_whether_it_is_partial(
    call_tool, cache_dir, max_rows, returned, truncated
):
    write_experiment(cache_dir, "exp", THREE_ROWS)

    result = call_tool(
        "read_experiment_metrics", experiment_name="exp", relative_path="m.csv", max_rows=max_rows
    )

    assert result["row_count"] == 3
    assert result["rows_returned"] == returned
    assert len(result["preview_rows"]) == returned
    assert result["truncated"] is truncated


@pytest.mark.parametrize("max_rows", [0, -1])
def test_metric_preview_rejects_limits_below_one(call_tool, cache_dir, max_rows):
    write_experiment(cache_dir, "exp", THREE_ROWS)

    result = call_tool(
        "read_experiment_metrics", experiment_name="exp", relative_path="m.csv", max_rows=max_rows
    )

    assert "max_rows" in result["error"]


@pytest.mark.parametrize(("max_files", "truncated"), [(2, True), (3, False), (5, False)])
def test_file_listing_is_truncated_only_when_more_files_exist(call_tool, s3, max_files, truncated):
    put_experiment(s3, "exp", {"a.csv": "1", "b.csv": "2", "c.csv": "3"})

    result = call_tool("list_experiment_files", experiment_name="exp", max_files=max_files)

    assert result["count"] == min(max_files, 3)
    assert result["truncated"] is truncated


@pytest.mark.parametrize("max_files", [0, -1])
def test_file_listing_rejects_limits_below_one(call_tool, s3, max_files):
    put_experiment(s3, "exp", {"a.csv": "1"})

    result = call_tool("list_experiment_files", experiment_name="exp", max_files=max_files)

    assert "max_files" in result["error"]


def test_metrics_tool_tells_the_model_not_to_calculate_from_previews():
    async def description() -> str:
        async with Client(mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            return tools["read_experiment_metrics"].description

    text = asyncio.run(description()).lower()

    assert "preview" in text
    assert "row_count" in text
    assert "not" in text and "calculat" in text
