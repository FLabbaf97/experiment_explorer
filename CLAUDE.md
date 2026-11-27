# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and running

- Environment: `conda env create -f environment.yml && conda activate experiment-explorer-mcp`. It installs this package in editable mode.
- Keep the install editable. The default cache path is derived from the location of `config.py`, so a regular install would put `.cache` inside the env.
- Run the MCP server over stdio: `python -m experiment_explorer`.
- AWS auth uses the default credential chain (`aws login`, or a profile/SSO).

## Verifying changes

- The conda environment includes the test tools (`.[dev]`). Run `ruff check .` and `pytest`. CI runs both on every push and pull request.
- CI also runs the tests with the lowest declared dependency versions (Python 3.10, `uv pip install --resolution lowest-direct`). Raise a minimum in `pyproject.toml` when code needs a newer API, and keep an upper bound below the next untested major.
- Tests run against a fake S3 (moto) and a temporary cache, so they need no AWS access. Test tool behaviour through the `call_tool` fixture, the way an MCP client does. Pure helpers such as the name validator can be tested directly.
- Add or update tests with every behaviour change.

## MCP server rules

- Never write to stdout from `src/`. It carries the stdio protocol. Log through the module-level `LOGGER`, which goes to stderr.
- Tools return a JSON string through `_json_response` in `server.py`.
- Expected failures return `{"error": ...}` rather than raising. Put `@_returns_errors` directly under `@mcp.tool` on every tool; it converts exceptions to error responses and logs unexpected ones with a traceback. Tools can simply raise `ValueError`, `OSError`, or `S3ExperimentError` for bad input.

## Data layout

- Experiments mix layouts (runs, production/LOBO runs, comparisons, analyses, GenoPHI, learning curves). Each file gets a role (report, metrics, predictions, config, figure, code, model, log, data, other) from ordered glob rules in `src/experiment_explorer/layout_rules.yaml`; first match wins (docs/adr/0003). Support a new layout by editing the rules, not the scanning code. `EXPERIMENT_EXPLORER_LAYOUT_RULES` adds a project's rules file, checked first.
- A run folder is `<kind>_YYYYMMDD_HHMMSS[_star]` with a kind listed in the rules (`results`, `production`). `run_*` sweep wrappers and timestamp-less `results_<name>` folders are not runs. The calculation tools compare `results` runs only; `compare_runs` names production runs in `runs_not_compared`.
- The original layout's `summarize_experiment` keys (`docs`, `run_directories`, `metrics_files`, `prediction_files`, `config_files`) must not change; `tests/test_layouts.py` pins them. Add a fixture there for every new layout.

## Cache and S3 safety

- Keep S3 access read-only.
- Experiment names go through `validate_experiment_name` in `paths.py`, which rejects blank, root, absolute, and `..` names. Keep new tools on it.
- `download_prefix` skips S3 keys that would land outside the experiment folder or overwrite the cache manifest, and reports them in `blocked_keys`.
- Each cached experiment has a `.s3_manifest.json` recording the S3 ETag of every downloaded file. A file is re-downloaded when its ETag changes; files cached before the manifest existed are re-downloaded once.
- Only files listed in the manifest are ever deleted: `force_refresh` removes them when they disappear from S3 and still match the downloaded size, and a normal fetch reports them in `missing_in_s3`. Files not in the manifest are never touched.
- Fetching an experiment with no files in S3 returns a not-found error and creates nothing in the cache.

## Results calculations

- The calculation tools (`summarize_run`, `compare_runs`, `bacteria_metrics`, `prediction_metrics`) live in `calculations.py`. They read files through the public loader in `tables.py`, which Claude also imports in its own scripts. Put file-reading rules in the loader so the tools and scripts stay consistent.
- The evaluation-setup keys and the bookkeeping keys each have one list in `tables.py`.
- `compare_runs` lists runs and never ranks them or names a best one (docs/adr/0001). A starred run is the default answer.
- Every calculation result reports its row counts and the `n` behind each statistic. NaN and inf become `null`.
- `skills/experiment-results/SKILL.md` tells Claude how to use these tools. Update it when tool behaviour changes.

## Plot script

- `scripts/plot_roc_pr_curves.py` needs the `plot` extra: `pip install -e ".[plot]"`.
- It reads the same cache as the server (`EXPERIMENT_EXPLORER_CACHE_DIR`, else `<repo>/.cache`); `--cache-dir` overrides it.
- Precision-recall curves are labelled with average precision (`AP=`), not a trapezoid PR-AUC.
- It plots `predictions/{test,cv}_predictions.csv` from any run folder and `predictions_normalized/<name>_{test,cv}.csv`. Experiments without per-sample predictions, such as `2026-09-16_*`, cannot be plotted.

@AGENTS.md
