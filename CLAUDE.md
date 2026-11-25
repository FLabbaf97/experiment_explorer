# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and running

- Environment: `conda env create -f environment.yml && conda activate experiment-explorer-mcp`. It installs this package in editable mode.
- Keep the install editable. The default cache path is derived from the location of `config.py`, so a regular install would put `.cache` inside the env.
- Run the MCP server over stdio: `python -m experiment_explorer`.
- AWS auth uses the default credential chain (`aws login`, or a profile/SSO).

## Verifying changes

- Install test tools with `pip install -e ".[dev]"`, then run `ruff check .` and `pytest`. CI runs both on every push and pull request.
- Tests run against a fake S3 (moto) and a temporary cache, so they need no AWS access. Call tools through the `call_tool` fixture, the way an MCP client does.
- Add or update tests with every behaviour change.

## MCP server rules

- Never write to stdout from `src/`. It carries the stdio protocol. Log through the module-level `LOGGER`, which goes to stderr.
- Tools return a JSON string through `_json_response` in `server.py`.
- Expected failures return `{"error": ...}` rather than raising.

## Data layout

- A run directory is `results_*` containing `results/`. Configs are files named exactly `config.yaml`.
- Metrics are recognized by exact filename at any depth: `cv_results.csv`, `test_results.csv`, `cv_per_bacteria_results.csv`, `test_per_bacteria_results.csv`.

## Cache and S3 safety

- Keep S3 access read-only.
- Experiment names go through `validate_experiment_name` in `paths.py`, which rejects blank, root, absolute, and `..` names. Keep new tools on it.
- `download_prefix` skips S3 keys that would land outside the experiment folder and reports them in `blocked_keys`.
- Cache freshness is checked by file size only, and `force_refresh` does not remove files deleted from S3. Cached files can be stale.
- A nonexistent experiment name creates an empty cache directory and reports success with 0 files.

## Plot script

- `scripts/plot_roc_pr_curves.py` needs the `plot` extra: `pip install -e ".[plot]"`.
- It ignores `EXPERIMENT_EXPLORER_CACHE_DIR`. Pass `--cache-dir` if the cache is not at `<repo>/.cache`.
- It plots `predictions/{test,cv}_predictions.csv` from any run folder and `predictions_normalized/<name>_{test,cv}.csv`. Experiments without per-sample predictions, such as `2026-09-16_*`, cannot be plotted.

@AGENTS.md
