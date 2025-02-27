import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# for bivariate normal CDF
from scipy.special import erf
from scipy.stats import norm

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

    # Importance sampling based on Bivariate Gaussian model
    # sample a bunch of theta_As, theta_Bs and rhos from the proposal
    theta_As = np.random.beta(1, 1, size=(M*M, num_samples))
    theta_Bs = np.random.beta(1, 1, size=(M*M, num_samples))
    rhos     = np.random.uniform(-1, 1, size=(M*M, num_samples))

    diff = theta_As - theta_Bs

    # calculate the mus for the 2D Gaussian (using the pdf of the bivariate normal)
    mu_A = stats.norm(0,1).ppf(theta_As)
    mu_B = stats.norm(0,1).ppf(theta_Bs)
    assert mu_A.shape == mu_B.shape == (M*M, num_samples)

    # calculate the probabilities of each case
    theta_V = binorm_cdf(0, 0, mu_A, mu_B, 1, 1, rhos)
    theta_S = theta_As + theta_Bs + theta_V - 1
    theta_T = 1 - theta_Bs - theta_V
    theta_U = 1 - theta_As - theta_V

    # calculate the log likelihoods
    # (with np.errstate to ignore nan-based errors (if we have log(nan)=nan we don't care))
    with np.errstate(divide='ignore', invalid='ignore'):
        log_likelihoods = S[:,None] * np.log(theta_S) + T[:,None] * np.log(theta_T) + U[:,None] * np.log(theta_U) + V[:,None] * np.log(theta_V)
    assert log_likelihoods.shape == (M*M, num_samples)

    # (which are equal to the weights since proposal = prior)
    log_weights = log_likelihoods

    # normalise the weights (with nan being probabilty 0)
    max_log_weights = np.nanmax(log_weights, axis=-1, keepdims=True)
    weights = np.exp(log_weights - max_log_weights)
    weights[np.isnan(weights)] = 0
    weights /= weights.sum(axis=-1, keepdims=True)
    assert weights.shape == (M*M, num_samples)

    # Get posterior samples
    diff_post = np.zeros((M*M, num_samples))
        
    for r in range(M*M):
        diff_post[r] = diff[r, np.random.choice(num_samples, size=num_samples, replace=True, p=weights[r])]

    assert diff_post.shape == (M*M, num_samples)

    # calculate the posterior probability of the difference being greater than 0
    posterior_prob = (diff_post > 0).mean(-1) 
    assert posterior_prob.shape == (M*M,)

    comparison_matrix = posterior_prob.reshape(M, M)

    # set diagonal to nan
    np.fill_diagonal(comparison_matrix, np.nan)

    comparison_matrix_df = pd.DataFrame(comparison_matrix, index=model_names, columns=model_names)

    return comparison_matrix_df

## Bivariate Normal CDF
# using jax implementation: https://github.com/jax-ml/jax/issues/10562
# from the paper "A simple approximation for the bivariate normal integral", Tsay & Ke (2021)
# https://www.tandfonline.com/doi/full/10.1080/03610918.2021.1884718

cdf1d = norm.cdf

c1 = -1.0950081470333
c2 = -0.75651138383854
sqrt2 = 1.4142135623730951

def case1(p, q, rho, a, b):
    a2 = a * a
    b2 = b * b

    line11 = 0.5 * (erf(q / sqrt2) + erf(b / (sqrt2 * a)))
    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * np.sqrt(1 - a2 * c2))) * np.exp(aux1 / aux2)

    line21 = 1 - erf((sqrt2 * b - a2 * c1) / (2 * a * np.sqrt(1 - a2 * c2)))
    aux3 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * np.sqrt(1 - a2 * c2))) * np.exp(aux3 / aux2)

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1
    aux5 = 2 * np.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((a2 * c1 + sqrt2 * b) / (2 * a * np.sqrt(1 - a2 * c2)))

    return line11 + (line12 * line21) - (line22 * (line31 + line32))

def case2(p, q):
    return cdf1d(p) * cdf1d(q)

def case3(p, q, rho, a, b):
    a2 = a * a
    b2 = b * b

    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line11 = (1.0 / (4 * np.sqrt(1 - a2 * c2))) * np.exp(aux1 / aux2)

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1
    aux4 = 2 * np.sqrt(1 - a2 * c2)
    line12 = 1.0 + erf(aux3 / aux4)

    return line11 * line12

def case4(p, q, rho, a, b):
    a2 = a * a
    b2 = b * b

    line11 = 0.5 + 0.5 * erf(q / sqrt2)
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * np.sqrt(1 - a2 * c2))) * np.exp(aux1 / aux2)

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1
    aux4 = 2 * np.sqrt(1 - a2 * c2)
    line2 = 1.0 + erf(aux3 / aux4)

    return line11 - (line12 * line2)

def case5(p, q, rho, a, b):
    a2 = a * a
    b2 = b * b

    line11 = 0.5 - 0.5 * erf(b / (sqrt2 * a))
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * np.sqrt(1 - a2 * c2))) * np.exp(aux1 / aux2)

    line21 = 1 - erf((sqrt2 * b + a2 * c1) / (2 * a * np.sqrt(1 - a2 * c2)))
    aux3 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * np.sqrt(1 - a2 * c2))) * np.exp(aux3 / aux2)

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1
    aux5 = 2 * np.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((-a2 * c1 + sqrt2 * b) / (2 * a * np.sqrt(1 - a2 * c2)))

    return line11 - (line12 * line21) + line22 * (line31 + line32)

def binorm_cdf(x1, x2, mu1, mu2, sigma1, sigma2, rho):
    '''
    Compute the bivariate normal CDF.

    Parameters:
    ----------
    x1, x2: int, float, np.ndarray
        The values at which to compute the CDF.
    mu1, mu2: int, float, np.ndarray
        The margianal means of the normal distribution.
    sigma1, sigma2: int, float, np.ndarray
        The marginal standard deviations of the normal distribution.
    rho: int, float, np.ndarray
        The correlation coefficient of the normal distribution.
    '''
    # Make sure the inputs are floats, or if ints, convert them to floats
    # Make sure the inputs are either int, float or np.ndarray.
    # If they're float, convert them to singleton float np.ndarrays.
    # If they're np.ndarray, make sure they're of the same shape.
    shape = None
    dtype = None
    for input_ in [x1, x2, mu1, mu2, sigma1, sigma2, rho]:
        if isinstance(input_, np.ndarray):
            if shape is None:
                shape = input_.shape
            else:
                assert shape == input_.shape, f"All np.ndarrays must have the same shape. Got {input_.shape} and {shape}."
            if dtype is None:
                dtype = input_.dtype
            else:
                assert dtype == input_.dtype, f"All np.ndarrays must have the same dtype. Got {input_.dtype} and {dtype}."
        else:
            assert isinstance(input_, (int, float)), f"All inputs must be either int, float or np.ndarray. Got {type(input_)}."
    
    if shape is None:
        shape = (1,)
    if dtype is None:
        dtype = np.float64

    # Convert the inputs to np.ndarray if they're int or float
    x1     = np.broadcast_to(np.array([float(x1)],     dtype=dtype), shape) if isinstance(x1,     (int, float)) else x1
    x2     = np.broadcast_to(np.array([float(x2)],     dtype=dtype), shape) if isinstance(x2,     (int, float)) else x2
    mu1    = np.broadcast_to(np.array([float(mu1)],    dtype=dtype), shape) if isinstance(mu1,    (int, float)) else mu1
    mu2    = np.broadcast_to(np.array([float(mu2)],    dtype=dtype), shape) if isinstance(mu2,    (int, float)) else mu2
    sigma1 = np.broadcast_to(np.array([float(sigma1)], dtype=dtype), shape) if isinstance(sigma1, (int, float)) else sigma1
    sigma2 = np.broadcast_to(np.array([float(sigma2)], dtype=dtype), shape) if isinstance(sigma2, (int, float)) else sigma2
    rho    = np.broadcast_to(np.array([float(rho)],    dtype=dtype), shape) if isinstance(rho,    (int, float)) else rho

    p = (x1 - mu1) / sigma1
    q = (x2 - mu2) / sigma2

    a = -rho / np.sqrt(1 - rho * rho)
    b = p / np.sqrt(1 - rho * rho)

    assert a.shape == b.shape == p.shape == q.shape == rho.shape == shape, f"Shapes of a, b, p, q and rho must be the same. Got {a.shape}, {b.shape}, {p.shape}, {q.shape} and {rho.shape}."

    # find the indices where each case applies
    case1_indices = (a > 0) & (a * q + b >= 0)
    case2_indices = (a == 0)
    case3_indices = (a > 0) & (a * q + b < 0)
    case4_indices = (a < 0) & (a * q + b >= 0)
    case5_indices = (a < 0) & (a * q + b < 0)

    # compute the CDF for each case
    result = np.zeros_like(p)

    result[case1_indices] = case1(p[case1_indices], q[case1_indices], rho[case1_indices], a[case1_indices], b[case1_indices])
    result[case2_indices] = case2(p[case2_indices], q[case2_indices])
    result[case3_indices] = case3(p[case3_indices], q[case3_indices], rho[case3_indices], a[case3_indices], b[case3_indices])
    result[case4_indices] = case4(p[case4_indices], q[case4_indices], rho[case4_indices], a[case4_indices], b[case4_indices])
    result[case5_indices] = case5(p[case5_indices], q[case5_indices], rho[case5_indices], a[case5_indices], b[case5_indices])

    return result

    # if a > 0 and a * q + b >= 0:
    #     return case1(p, q, rho, a, b)
    # if a == 0:
    #     return case2(p, q)
    # if a > 0 and a * q + b < 0:
    #     return case3(p, q, rho, a, b)
    # if a < 0 and a * q + b >= 0:
    #     return case4(p, q, rho, a, b)
    # if a < 0 and a * q + b < 0:
    #     return case5(p, q, rho, a, b)


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
