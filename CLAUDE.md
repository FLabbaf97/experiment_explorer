# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and running

- Environment: `conda env create -f environment.yml && conda activate experiment-explorer-mcp`. It installs this package in editable mode.
- Keep the install editable. The default cache path is derived from the location of `config.py`, so a regular install would put `.cache` inside the env.
- Run the MCP server over stdio: `python -m experiment_explorer`.
- AWS auth uses the default credential chain (`aws login`, or a profile/SSO).

## Verifying changes

- There are no tests or CI. Verify changes by running the affected code, and say what was not run.
- Ruff is configured (line length 100, target py310) but is not a declared dependency. Install it separately to run `ruff check .`.

## MCP server rules

- Never write to stdout from `src/`. It carries the stdio protocol. Log through the module-level `LOGGER`, which goes to stderr.
- Tools return a JSON string through `_json_response` in `server.py`.
- Expected failures return `{"error": ...}` rather than raising.

## Data layout

- A run directory is `results_*` containing `results/`. Configs are files named exactly `config.yaml`.
- Metrics are recognized by exact filename at any depth: `cv_results.csv`, `test_results.csv`, `cv_per_bacteria_results.csv`, `test_per_bacteria_results.csv`.

## Cache and S3 safety

- Keep S3 access read-only.
- Never call `fetch_experiment` with an empty `experiment_name`. It resolves to the bucket root and downloads the whole bucket.
- Cache freshness is checked by file size only, and `force_refresh` does not remove files deleted from S3. Cached files can be stale.
- A nonexistent experiment name creates an empty cache directory and reports success with 0 files.

## Plot script

- `scripts/plot_roc_pr_curves.py` needs `matplotlib` and `scikit-learn`, which are not declared in `pyproject.toml` or `environment.yml`.
- It ignores `EXPERIMENT_EXPLORER_CACHE_DIR`. Pass `--cache-dir` if the cache is not at `<repo>/.cache`.
- Its figure title is hardcoded to the 2026-01-08 baseline, so plots of other experiments are mislabeled.
- It requires `split_*/model_*/results_*` nesting and aborts if `cv_predictions.csv` is missing.

@AGENTS.md
