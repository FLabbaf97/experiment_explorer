---
status: accepted
---

# File roles come from one ordered list of path rules, not from per-experiment layouts

Experiments are written by different pipelines (training runs, production and LOBO runs, comparisons, diversity analyses, GenoPHI, learning curves), and one experiment often mixes several of them. So the explorer does not detect "the layout" of an experiment. Every file is classified on its own by an ordered list of path rules (first match wins) into a file role: report, metrics, predictions, config, figure, code, model, log, data or other. Which folders are runs is decided by a name pattern in the same file. The rules live in a YAML file shipped with the package. A project with other layouts points `EXPERIMENT_EXPLORER_LAYOUT_RULES` at its own rules file, whose rules are checked first, so new layouts need configuration, not code.

## Considered Options

- One named layout per experiment, detected from its folders: rejected because experiments mix layouts, and every new pipeline would need detection code.
- A layout file inside each experiment in S3: rejected because none of the existing experiments has one and the explorer only reads S3; it may be added later as a third rules source.
- Sniffing file contents (for example, a `y_pred_proba` column means predictions): rejected for now; names and folders were enough for every layout in use, and reading every file would slow summaries.
