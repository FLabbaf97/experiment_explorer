---
status: accepted
---

# Run comparisons never name a best run

When an experiment holds several runs for the same question, the explorer lists them, grouped by evaluation setup and in time order, with each run's config differences and headline metrics. It never marks one as best, and the skill tells Claude not to pick one even when asked. The only default is a starred run, because its author already chose. We decided this because "best" is rarely clear: runs in one folder often differ in their test data, so their scores are not comparable. Even when the evaluation setup matches, the right trade-off (ROC AUC against average precision, stability against peak score) is the user's call.

## Considered Options

- Name the run with the highest test ROC AUC: rejected because it ranks scores computed on different test sets.
- Name a best only among runs with the same evaluation setup: rejected because the metric trade-off still belongs to the user.
