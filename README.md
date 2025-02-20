# `bayes_evals`: A lightweight library for Bayesian analysis of LLM evals

## Installation
```bash
pip install -e .
```

## Usage
```python
import bayes_evals as be
import pandas as pd

# Load the data (should NOT contain an index column)
eval_data = pd.from_csv('data/evals.csv')

# Analyse the data
evals = be.analyse(eval_data)

# Get the results either for individual LLMs (each column in the data)...
indep_intervals = evals.independent_intervals()

# ... in which case you can then do independent LLM comparisons...
indep_comparisons = evals.independent_comparisons()

# ... or get pairwise comparisons between LLMs
paired_comparisons = evals.paired_comparisons()
```

Each of the `indep_results`, `indep_comparisons`, and `paired_comparisons` objects are `Result` objects, which have methods for displaying the results in two ways:
1. As a table/dataframe (print to console via `pandas`)
2. As a plot (using `matplotlib`)

## Data format
The data should be in a `pandas` DataFrame, with $Q$ = no. questions rows and $M$ = no. LLMs columns.
The data should be binary, with 1 indicating a correct answer and 0 indicating an incorrect answer.
The columns should be named with the LLMs' names.

Alternatively, the data can just be an $[M,Q]$ numpy array, optionally with a list of $M$ LLM names.


#### Laurence's README
Two kinds of question we allow to ask:
* Paired comparison, question-level data
  - Paired comparison (i.e. two models) makes sense; we could do a comparison with more models, but not clear how that would help us.
  - No tasks (probably best to assume no task-level structure).
  - No clusters (I'm not sure how often that comes up anyway.
  - Input data is a binary tensor [M, Q], where M is the number of models, and Q is the number of questions.
  - Individual model error bars (e.g. for plotting) from single-model Beta-Binomial model.
  - Paired comparisons from the four-way Dirichlet.
* Indepdendent comparison, aggregate-level data.
  - Input data is binary tensor [M,Q]

Then, what happens?  After the first call that ingest the data, we get an intermediate Result object.  Call methods on this intermediate Result object to display the results in different ways.