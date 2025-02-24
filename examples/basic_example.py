import bayes_evals as be
import pandas as pd

# Load the data (should NOT contain an index column)
eval_data = pd.read_csv('data/langchain_data_tool_use_FULL.csv')
print(f"Data shape: {eval_data.shape}")
print(eval_data)

# Analyse the data
evals = be.analyse(eval_data)
print(evals)

# Get the results either for individual LLMs (each column in the data)...
print("Getting independent indep_intervals")
indep_intervals = evals.independent_intervals()
indep_intervals.plot('plots/indep_intervals.png')
indep_intervals.print()


# ... in which case you can then do independent LLM comparisons...
print("Getting independent comparisons")
indep_comparisons = evals.independent_comparisons()
indep_comparisons.plot('plots/indep_comparisons.png')
indep_comparisons.print()


# ... or get pairwise comparisons between LLMs
print("Getting paired comparisons")
paired_comparisons = evals.paired_comparisons()
paired_comparisons.plot('plots/paired_comparisons.png')
paired_comparisons.print()