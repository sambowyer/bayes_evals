import bayes_evals as be
import pandas as pd

# Load the data (should NOT contain an index column)
eval_data = pd.read_csv('data/langchain_data_tool_use_FULL.csv')
print(f"Data shape: {eval_data.shape}")
print(eval_data)

# Get the results either for individual LLMs (each column in the data)...
print("Getting independent indep_intervals")
indep_intervals = be.independent_intervals(eval_data)
print(indep_intervals)
be.plot_intervals(eval_data, indep_intervals, 'plots/indep_intervals.png')


# ... in which case you can also do independent LLM comparisons...
print("Getting independent comparisons")
indep_comparisons = be.independent_comparisons(eval_data)
print(indep_comparisons)
be.plot_comparisons(indep_comparisons, 'plots/indep_comparisons.png', title="Independent LLM comparisons")


# ... or get pairwise comparisons between LLMs
print("Getting paired comparisons")
paired_comparisons = be.paired_comparisons(eval_data)
print(paired_comparisons)
be.plot_comparisons(paired_comparisons, 'plots/paired_comparisons.png', title="Paired LLM comparisons")