"""The installed server starts over stdio and describes its configured bucket (issue #9)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

import experiment_explorer

# Run the same code the tests import, whether or not the install is editable.
PACKAGE_ROOT = Path(experiment_explorer.__file__).resolve().parents[1]


def test_server_starts_over_stdio_and_names_the_configured_bucket(tmp_path):
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "experiment_explorer"],
        env={
            "PYTHONPATH": str(PACKAGE_ROOT),
            "EXPERIMENT_EXPLORER_S3_BUCKET": "another-bucket",
            "EXPERIMENT_EXPLORER_CACHE_DIR": str(tmp_path / "cache"),
        },
    )

    async def handshake() -> tuple[str, set[str]]:
        async with Client(transport) as client:
            tools = {tool.name for tool in await client.list_tools()}
            # fastmcp 4 exposes client.instructions; earlier majors only initialize_result.
            instructions = getattr(client, "instructions", None)
            if instructions is None:
                instructions = client.initialize_result.instructions
            return instructions, tools

    instructions, tools = asyncio.run(handshake())

    assert "another-bucket" in instructions
    assert "phage-match-ml-experiments" not in instructions
    assert {"list_experiments", "fetch_experiment", "read_cached_file"} <= tools
