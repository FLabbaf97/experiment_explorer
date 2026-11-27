---
status: accepted
---

# Calculations go through named tools, a shared loader and a skill, not a generic query tool

Results questions are answered over every row of a file by a few named MCP tools (`summarize_run`, `compare_runs`, `bacteria_metrics`, `prediction_metrics`). These tools run in any Claude client. They share one loader module (`experiment_explorer.tables`) that reads results and prediction files in a clean form. For calculations no tool covers, a skill teaches Claude the data's traps and tells it to import that loader into its own code. We do not offer a generic tool with group-by, filter and aggregate parameters: Claude already runs code. The hard part is reading the files correctly (ensemble rows mixed in with member rows, CV metrics stored as dict text, two names for the label column), and a query language would not fix that.

## Consequences

- Correctness rules live in the loader, so the tools and Claude's own scripts cannot drift apart.
- Custom calculations need a client that can run Python in this package's environment. Other clients get the named tools only.
