---
name: experiment-results
description: Answer questions about phage-match experiment results (how good a run is, how runs compare, per-bacteria or per-phage figures, metrics from predictions, or any new metric) from every row, using the experiment_explorer tools and loader. Use whenever a question needs a number calculated from experiment results.
---

# Experiment results

Every figure comes from every row of a file. The four calculation tools and the `experiment_explorer.tables` loader read whole files. `read_experiment_metrics` only shows a preview, so use it to see columns, not for numbers. Terms (run, ensemble, member, fold, starred run, evaluation setup, headline metric, incomplete run) are defined in the repo's `GLOSSARY.md`.

## Pick the tool

| Question | Tool |
|---|---|
| Which run is the result? How do runs compare? | `compare_runs` |
| How good is this run? How stable is it? | `summarize_run` |
| One bacterium's metric, stats across bacteria, how many pass a threshold, best or worst N | `bacteria_metrics` |
| Metrics from predictions, overall or per bacteria, phage or fold; do members agree? | `prediction_metrics` |
| Anything else | your own code, through the loader (below) |

Start an experiment question with `compare_runs` to find its runs, then go deeper with the others. To address a run, pass its folder path (from `compare_runs`) and a partition (`test` or `cv`). For files outside the standard run layout (for example production runs, which `compare_runs` lists under `runs_not_compared`, or comparison folders), find them with `summarize_experiment` and pass them as `relative_path`.

## Choosing the run

- A **starred run** is the answer. Mention that other runs exist, as `compare_runs` reports.
- Without a star, lay out every run: its setup group, the settings that differ, and its headline ROC AUC and average precision. Point out trade-offs, for example when ROC AUC and average precision disagree. **The user chooses.** Leave the choice to the user even when asked "which is best?": scores from different setup groups come from different test data, and the right trade-off is theirs (docs/adr/0001).
- Only runs in the same setup group are comparable. When groups differ, say what differs (for example "tested on BASEL_data only").
- Name incomplete runs and give their reason.

## Data rules

- **Ensemble vs members.** The headline is the ensemble's value. Member rows describe spread only. The tools already separate them; in your own code use `select_members`.
- **Test vs CV.** A run without test results has a CV headline: the mean across folds. State which one every figure is. Within one comparison, compare test with test and CV with CV.
- **CV per bacterium.** In unseen-phage splits a bacterium appears in several folds. Its figure is the mean of its fold values, unless you computed a pooled figure from predictions (`prediction_metrics` with `group_by="bacteria"`). Say which.
- **IDs are text.** `132` and `007` are bacterium names, not numbers. When a lookup fails, show the user the close IDs from the error and ask which they meant.
- **Missing values** are left out. Report the `n` behind each statistic, especially when it is small.

## Every answer states

- the run (folder name) and whether it is starred
- test or CV
- how many rows or values the figure used (`row_count` / `rows_used` / `n`)
- when a listing is `truncated`, that it is partial: narrow the request, raise `max_groups`, or use the loader

## New metrics with the loader

When no tool computes what the user asks, write Python with the server's own interpreter (the `experiment-explorer-mcp` conda environment). Run it against the cache folder that `list_cached_experiments` reports:

```python
from pathlib import Path
from experiment_explorer.tables import load_table, select_members, run_file, find_runs

exp = Path("<cache_dir>") / "<experiment_name>"
preds = select_members(load_table(run_file(exp, "<run>", "test", "predictions")), "ensemble")
```

`load_table` turns CV dict text into `val_*` columns, merges the label into `y_true`, flags `is_ensemble`, and keeps IDs as text. Load files through it every time, so your numbers follow the same rules as the tools.
