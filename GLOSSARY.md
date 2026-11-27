# Experiment Explorer

Answers questions about phage-match ML experiments (models that predict which phages infect which bacteria) from the files those experiments produced.

## Language

**Run**:
One training and evaluation of a model configuration, with its own metrics and predictions.
_Avoid_: Trial, attempt

**Starred run**:
A run its author marked as the one that stands for its group of runs; when present, it is the group's result.
_Avoid_: Best run, chosen run

**Ensemble**:
The combined model a run produces from its members; its results are the run's own results.
_Avoid_: Final model

**Member**:
One of the models trained inside a run whose predictions are combined into the ensemble.
_Avoid_: Seed model, sub-model

**Fold**:
One cross-validation split of the training data, with its own validation metrics.

**Evaluation setup**:
The settings that decide which data a run is trained and tested on, such as its data sources and how they are split into folds and a test set. Runs are only fairly comparable when their evaluation setups match.
_Avoid_: Data config

**Model settings**:
Every setting of a run that is not part of its evaluation setup, such as the model's parameters, its features and how classes are balanced.
_Avoid_: Hyperparameters (too narrow)

**Headline metric**:
The ensemble's value for a metric on the test set when the run has one, otherwise the mean of that metric across the run's folds: the single number that answers "how good is this run".
_Avoid_: Final score, best score

**Incomplete run**:
A run that left no usable results, for example because it stopped before evaluating.
