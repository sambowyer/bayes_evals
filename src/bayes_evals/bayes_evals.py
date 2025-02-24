import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    'independent_intervals',
    'independent_comparisons',
    'paired_comparisons',
    'plot_intervals',
    'plot_comparisons',
]

def convert_to_df(data, model_names=None):
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, np.ndarray):
        if model_names is None:
            model_names = [f"Model {i}" for i in range(data.shape[0])]
        return pd.DataFrame(data, columns=model_names)
    
    raise ValueError(f"Data must be a numpy array or pandas DataFrame, found {type(data)}")

def extract_data_from_df(df):
    if isinstance(df, pd.DataFrame):
        return df.to_numpy().T, df.columns
    elif isinstance(df, np.ndarray):
        return df, None
    raise ValueError(f"Data must be a numpy array or pandas DataFrame, found {type(df)}")

def print_intervals(df):
    max_name_length = max([len(name) for name in df.columns])
    for llm in df.columns:
        print(f"{llm.rjust(max_name_length)}: {df[llm].values}")
        

################################################################
## Independent Analysis
def get_bayes_posterior(data, prior=(1, 1)):
    '''
    Compute the posterior distribution of the parameter of a Bernoulli distribution. 

    Parameters
    ----------
    data: np.ndarray of shape (..., n)
        Binary data, questions dimension is the last dimension
    prior: tuple
        Beta distribution parameters (default: (1, 1) i.e. uniform)

    Returns
    -------
    posterior: scipy.stats.beta
        Posterior distribution
    '''
    n = data.shape[-1]
    a = data.sum(axis=-1)
    b = n - a
    return stats.beta(a + prior[0], b + prior[1])

def independent_intervals(df, alpha=0.05, prior=(1, 1)):
    '''
    Compute the credible interval for the parameter of a Bernoulli distribution. 

    Parameters
    ----------
    df: pd.DataFrame
        Binary data, shape [Q,M] where Q is the number of questions and M is the number of models
    alpha: float, or list of floats
        Significance levels
    prior: tuple
        Beta distribution parameters (default: (1, 1) i.e. uniform)

    Returns
    -------
    credible_interval_df: pd.DataFrame (shape [2,M])
        Lower and upper bounds (rows) of the credible interval for each model (columns)
    '''
    assert isinstance(df, pd.DataFrame), f"Data must be a pandas DataFrame (found {type(df)})"
    assert isinstance(alpha, float), f"Alpha must be a float (found {type(alpha)})"
    
    data, model_names = extract_data_from_df(df)
    M = data.shape[0]

    posterior = stats.beta(data.sum(-1) + prior[0], data.shape[-1] - data.sum(-1) + prior[1])
    interval = np.column_stack(posterior.interval(1 - alpha))
    
    assert interval.shape == (M, 2)
    
    credible_interval_df = pd.DataFrame(interval.T, index=['lower', 'upper'], columns=model_names)
    return credible_interval_df

## Independent Comparison
def independent_comparisons(df, num_samples=10_000, prior=(1, 1)):
    '''
    Compute the credible interval for the parameter of a Bernoulli distribution. 

    Parameters
    ----------
    df: pd.DataFrame
        Binary data, shape [Q,M] where Q is the number of questions and M is the number of models
    num_samples: int
        Number of samples to draw from the posterior
    prior: tuple
        Beta distribution parameters (default: (1, 1) i.e. uniform)

    Returns
    -------
    comparison_matrix_df: pd.DataFrame (shape [M,M])
        Square pd.DataFrame containing entries of posterior probability of the difference in theta_A and theta_B being greater than 0
        for all pairs of models A (rows) and B (columns) (with nan on the diagonal)
    '''
    assert isinstance(df, pd.DataFrame), f"Data must be a pandas DataFrame (found {type(df)})"

    data, model_names = extract_data_from_df(df)
    M = data.shape[0]

    posterior_samples = get_bayes_posterior(data, prior).rvs(size=(num_samples,M))
    assert posterior_samples.shape == (num_samples,M)

    comparison_matrix = (posterior_samples[:, :, None] > posterior_samples[:, None, :]).mean(0)
    assert comparison_matrix.shape == (M,M)

    # set diagonal to nan
    np.fill_diagonal(comparison_matrix, np.nan)

    comparison_matrix_df = pd.DataFrame(comparison_matrix, index=model_names, columns=model_names)

    return comparison_matrix_df

################################################################
## Paired Comparison
def paired_comparisons(df, num_samples=10_000):
    '''
    Use Dirichlet-multiinomial model to compare all pairs of models present in the data

    df: pd.DataFrame
        Binary data, shape [Q,M] where Q is the number of questions and M is the number of models
    num_samples: int
        Number of samples to draw from the posterior

    Returns
    -------
    comparison_matrix_df: np.pd.DataFrame (shape [M,M])
        Square pd.DataFrame containing entries of posterior probability of the difference in theta_A and theta_B being greater than 0
        for all pairs of models A (rows) sand B (columns) (with nan on the diagonal)
    '''
    assert isinstance(df, pd.DataFrame), f"Data must be a pandas DataFrame (found {type(df)})"

    data, model_names = extract_data_from_df(df)
    M = data.shape[0]
    Q = data.shape[1]

    # get all possible pairs of models
    data_A = np.repeat(data, M, axis=0) # repeat each row M times
    data_B = np.tile(data, (M, 1))      # tile the data M times
                                        # so the first M rows are (A1, B1), (A1, B2), ..., (A1, BM)
                                        # and the second M rows are (A2, B1), (A2, B2), ..., (A2, BM)
    assert data_A.shape == data_B.shape == (M*M, Q)

    # create the 2x2 contingency table (flattened)
    #              | B correct | B incorrect
    # -------------|-----------|----------
    # A correct    | S         | T 
    # A incorrect  | U         | V 
    S = (data_A * data_B).sum(-1)             # S = A correct,   B correct
    T = (data_A * (1 - data_B)).sum(-1)       # T = A correct,   B incorrect
    U = ((1 - data_A) * data_B).sum(-1)       # U = A incorrect, B correct
    V = ((1 - data_A) * (1 - data_B)).sum(-1) # V = A incorrect, B incorrect

    table = np.column_stack([S, T, U, V])
    assert table.shape == (M*M, 4)

    dir_post = np.zeros((M*M, num_samples, 4))
    for comparison in range(M*M):
        dir_post[comparison] = stats.dirichlet(table[comparison] + 1).rvs(size=(num_samples,))

    assert np.allclose(dir_post.sum(-1), 1)

    # theta_As_post = dir_post[..., (0,1)].sum(-1)  # S + T
    # theta_Bs_post = dir_post[..., (0,2)].sum(-1)  # S + U
    # diff_post = theta_As_post - theta_Bs_post     # (S + T) - (S + U) = T - U
    diff_post = dir_post[..., 1] - dir_post[..., 2]
    assert diff_post.shape == (M*M, num_samples)

    posterior_prob = (diff_post > 0).mean(-1) 
    assert posterior_prob.shape == (M*M,)

    comparison_matrix = posterior_prob.reshape(M, M)

    # set diagonal to nan
    np.fill_diagonal(comparison_matrix, np.nan)

    comparison_matrix_df = pd.DataFrame(comparison_matrix, index=model_names, columns=model_names)

    return comparison_matrix_df

################################################################
## Plotting
def plot_intervals(eval_df, intervals_df, filename = None, title = "Model Accuracy"):
    assert isinstance(eval_df, pd.DataFrame)
    assert isinstance(intervals_df, pd.DataFrame)
    assert isinstance(title, str)

    intervals, model_names = extract_data_from_df(intervals_df)
    eval_data, _ = extract_data_from_df(eval_df)

    means = eval_data.mean(-1)

    fig, ax = plt.subplots()

    # plot means and error bars separately as they might not line up (e.g. mean is outside the error bars)
    ax.bar(model_names, means)

    positions = np.arange(len(model_names))
    for i, llm in enumerate(model_names):
        ax.bxp([{
                'med': means[i],
                'q1': means[i],
                'q3': means[i],
                'whislo': intervals[i,0],
                'whishi': intervals[i,1],
                'caps': intervals[i],
                'fliers': [],
                'mean': []
            }], 
            positions=[positions[i]], 
            widths=1, 
            showfliers=False, 
            boxprops=dict(color='#444'), 
    )
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

def plot_comparisons(comparison_df, filename = None, title = "Model Comparison"):
    assert isinstance(comparison_df, pd.DataFrame)
    assert isinstance(title, str)

    comparison_matrix, model_names = extract_data_from_df(comparison_df)

    fig, ax = plt.subplots()
    
    # transpose to have Model A on y-axis and Model B on x-axis (because data is in Model B cols x Model A rows)
    cax = ax.matshow(comparison_matrix.T, cmap='RdYlGn') 

    # add colorbar with title
    cbar = fig.colorbar(cax)
    cbar.set_label('P(Model A > Model B)')

    # set ticks and labels with model names
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticklabels(model_names)

    ax.set_title(title)
    ax.set_xlabel('Model B')
    ax.set_ylabel('Model A')

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()
