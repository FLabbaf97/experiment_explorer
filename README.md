# Experiment Explorer MCP

MCP server for exploring phage-match ML experiments stored in S3 (`phage-match-ml-experiments`). It lists experiment folders, downloads them to a local cache, and exposes metrics/config files for analysis and plotting in Cursor.

## Setup

### 1. Create the isolated environment

```bash
cd experiment_explorer
conda env create -f environment.yml
conda activate experiment-explorer-mcp
```

### 2. Configure AWS credentials

Authenticate with AWS before using S3 tools:

```bash
aws login
# or use your usual AWS profile / SSO setup
```

### 3. Register the MCP server in Cursor

Add to `~/.cursor/mcp.json` (see `mcp.json.example`):

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

Restart Cursor after updating MCP settings.

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

The server communicates over stdio (standard MCP transport for Cursor).
