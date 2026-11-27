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
pip install -e ".[plot,dev]"
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
claude mcp add --transport stdio experiment_explorer \
  --env EXPERIMENT_EXPLORER_S3_BUCKET=phage-match-ml-experiments \
  --env EXPERIMENT_EXPLORER_CACHE_DIR=/path/to/experiment_explorer/.cache \
  -- /path/to/miniconda3/envs/experiment-explorer-mcp/bin/python -m experiment_explorer
```

Keep the server name before `--env`: `--env` takes every value up to the next option, so a name placed after it is read as a malformed variable.

- Only you, only the current project (default): omit `--scope`, or pass `--scope local`.
- You, in every project: add `--scope user`.
- Everyone working in the current project: add `--scope project` (writes `.mcp.json` there, which can be committed).

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
| `summarize_experiment` | Find reports and READMEs at any depth, run dirs, metrics, predictions, configs |
| `list_cached_files` | List every file in a cached experiment with its size |
| `read_cached_file` | Read a Markdown, text, JSON, YAML or log file (size-limited; binaries are never opened) |
| `read_experiment_metrics` | Preview a metrics CSV (first rows only; not for calculations) |
| `read_experiment_config` | Load a YAML config file |
| `compare_runs` | List an experiment's runs grouped by evaluation setup, with config differences and headline metrics (never ranked) |
| `summarize_run` | A run's headline metrics and their spread across ensemble members and CV folds |
| `bacteria_metrics` | Per-bacteria answers: one bacterium's metrics, stats across bacteria, threshold counts, best/worst N |
| `prediction_metrics` | ROC AUC, average precision, precision, recall, F1, MCC and confusion matrix from every prediction, optionally per bacteria, phage or fold |

The four calculation tools read every row of a file and say how many rows they used. For calculations they don't cover, import the same loader in your own Python: `from experiment_explorer.tables import load_table`. The `skills/experiment-results` skill tells Claude when to use each tool and the rules of the data. Terms are defined in `GLOSSARY.md`, and design decisions are recorded in `docs/adr/`.

## Typical workflow

1. `list_experiments` — browse available runs in S3
2. `fetch_experiment` with `experiment_name="2026-04-20_Baseline"`
3. `compare_runs` — see the experiment's runs, how they differ, and their headline metrics
4. `summarize_run`, `bacteria_metrics`, `prediction_metrics` — answer questions from every row
5. `summarize_experiment` / `read_experiment_config` — discover files and read configs
6. `list_cached_files` / `read_cached_file` — read written reports, notes and JSON summaries

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

## Plotting

Install the plot dependencies, then plot any cached experiment that has per-sample predictions:

```bash
pip install -e ".[plot]"
python scripts/plot_roc_pr_curves.py --experiment-name 2026-07-16_ecoli_selected_features_holdout
```

The script reads the same cache as the server (`EXPERIMENT_EXPLORER_CACHE_DIR`, or `.cache` in this folder); pass `--cache-dir` to use another. ROC curves are labelled with ROC AUC and precision-recall curves with average precision (AP). Figures are saved to `<experiment>/plots` unless you pass `--output-dir`. The script reads `predictions/{test,cv}_predictions.csv` in run folders and `predictions_normalized/<name>_{test,cv}.csv`.

## Local development

```bash
conda activate experiment-explorer-mcp
python -m experiment_explorer
```

Run the tests and linter (no AWS access needed; S3 is faked):

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

The server communicates over stdio (standard MCP transport for Cursor and Claude Code).
