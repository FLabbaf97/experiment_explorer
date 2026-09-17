# Experiment Explorer MCP

MCP server for exploring phage-match ML experiments stored in S3 (`phage-match-ml-experiments`). It lists experiment folders, downloads them to a local cache, and exposes metrics/config files for analysis and plotting in Cursor or Claude Code.

## Setup

### 1. Install the server

From this directory, create the isolated conda environment (installs the package editable via `pip`):

```bash
cd experiment_explorer
conda env create -f environment.yml
conda activate experiment-explorer-mcp
```

If the environment already exists, update the install with:

```bash
conda activate experiment-explorer-mcp
pip install -e .
```

Confirm the Python used by MCP is the conda env interpreter:

```bash
which python
# example: ~/miniconda3/envs/experiment-explorer-mcp/bin/python
```

### 2. Configure AWS credentials

Authenticate with AWS before using S3 tools:

```bash
aws login
# or use your usual AWS profile / SSO setup
```

### 3. Register the MCP server in Cursor

Add the block below to `~/.cursor/mcp.json` (see `mcp.json.example`). Replace the Python path and cache directory with your machine’s paths.

```json
{
  "mcpServers": {
    "experiment_explorer": {
      "command": "/path/to/miniconda3/envs/experiment-explorer-mcp/bin/python",
      "args": ["-m", "experiment_explorer"],
      "env": {
        "EXPERIMENT_EXPLORER_S3_BUCKET": "phage-match-ml-experiments",
        "EXPERIMENT_EXPLORER_CACHE_DIR": "/path/to/experiment_explorer/.cache"
      }
    }
  }
}
```

Restart Cursor after updating MCP settings. In **Settings → MCP**, `experiment_explorer` should appear as connected.

### 4. Register the MCP server in Claude Code

Add the same stdio server with the Claude Code CLI. Use the **absolute** path to the conda env Python:

```bash
claude mcp add --transport stdio \
  --env EXPERIMENT_EXPLORER_S3_BUCKET=phage-match-ml-experiments \
  --env EXPERIMENT_EXPLORER_CACHE_DIR=/path/to/experiment_explorer/.cache \
  experiment_explorer -- \
  /path/to/miniconda3/envs/experiment-explorer-mcp/bin/python -m experiment_explorer
```

- User-wide (default): omit `--scope`, or pass `--scope user`.
- This repo only: add `--scope project` (writes `.mcp.json` in the project).

Check that it is registered:

```bash
claude mcp list
```

Restart Claude Code after adding the server. You can also paste the same JSON as in `mcp.json.example` into Claude Code MCP settings if you prefer the UI over the CLI.

## Tools

| Tool | Description |
|------|-------------|
| `list_experiments` | List top-level experiment folders in S3 |
| `list_experiment_files` | List files under one experiment prefix |
| `fetch_experiment` | Download an experiment folder to local cache |
| `list_cached_experiments` | List experiments already downloaded locally |
| `summarize_experiment` | Find README, run dirs, metrics, predictions, configs |
| `read_experiment_metrics` | Load a metrics CSV (preview rows) |
| `read_experiment_config` | Load a YAML config file |

## Typical workflow

1. `list_experiments` — browse available runs in S3
2. `fetch_experiment` with `experiment_name="2026-04-20_Baseline"`
3. `summarize_experiment` — discover metrics file paths
4. `read_experiment_metrics` — inspect `test_results.csv`, etc.
5. Use the returned data for plotting or further analysis

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `EXPERIMENT_EXPLORER_S3_BUCKET` | `phage-match-ml-experiments` | S3 bucket name |
| `EXPERIMENT_EXPLORER_CACHE_DIR` | `experiment_explorer/.cache` | Local download directory |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | boto3 default | Optional region override |

## Experiment layout

Experiments follow the structure used in `phage_match_RnD/experiments/`:

```
experiment_name/
├── README.md
├── run_exp.sh
├── split_*/model_*/
│   └── results_YYYYMMDD_HHMMSS/
│       ├── results/
│       │   ├── cv_results.csv
│       │   ├── test_results.csv
│       │   └── ...
│       ├── predictions/
│       ├── logs/config.yaml
│       └── artifacts/
```

## Local development

```bash
conda activate experiment-explorer-mcp
python -m experiment_explorer
```

The server communicates over stdio (standard MCP transport for Cursor and Claude Code).
