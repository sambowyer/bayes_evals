import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Union

# from .plotting import plot_error_bars, plot_comparison_matrix
import matplotlib.pyplot as plt

__all__ = [
    'analyse',
]   

class EvalsData():
    def __init__(self, data : Union[np.ndarray, pd.DataFrame], model_names : list[str] = None):
        if isinstance(data, pd.DataFrame):
            self.data = data.to_numpy().T
            if model_names is None:
                self.model_names = data.columns
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError(f"Data must be a numpy array or pandas DataFrame, found {type(data)}")
        
        assert len(data.shape) in (1,2), f"Data must be a 1D or 2D array (found {len(data.shape)} dimensions)"
        if len(data.shape) == 1:
            self.data = self.data[:,None] # Add a singleton model dimension to make it 2D

        self.num_models = self.data.shape[0]        
        self.num_questions = self.data.shape[1]

        if model_names is not None:
            assert len(model_names) == self.num_models, f"Number of model names ({len(model_names)}) must match number of columns in data ({self.num_models})"
            self.model_names = model_names
        elif model_names is None and not isinstance(data, pd.DataFrame):
            self.model_names = [f"Model {i}" for i in range(self.num_models)]

    def __repr__(self):
        return f"EvalsData Obj: {self.num_models} models, {self.num_questions} questions"

    def independent_intervals(self, alpha : float = 0.05, prior : tuple = (1, 1)):
        '''
        Run beta-binomial analysis on the data and return the error bars for each model.

        Parameters
        ----------
        alpha: float, or list of floats
            Significance levels (default: 0.05)
        prior: tuple
            Beta distribution parameters (default: (1, 1) i.e. uniform)

        Returns
        -------
        IndependentIntervals object
        '''
        return IndependentIntervals(self, independent_credible_intervals(self.data, alpha=alpha, prior=prior))

    def independent_comparisons(self, num_samples : int = 10_000, prior : tuple = (1, 1)):
        '''
        Run beta-binomial independent comparison on the data and return the matrix of posterior probabilities.

        Parameters
        ----------
        num_samples: int
            Number of samples to draw from the posterior
        prior: tuple
            Beta distribution parameters (default: (1, 1) i.e. uniform)

        Returns
        -------
        IndependentComparison object
        '''
        return IndependentComparison(self, independent_comparison_probabilities(self.data, num_samples=num_samples, prior=prior))

    def paired_comparisons(self, num_samples : int = 10_000):
        '''
        Run paired Dirichlet analysis on all possilbe pairs of models in the data and return the matrix of posterior probabilities.

        Parameters
        ----------
        num_samples: int
            Number of samples to draw from the posterior

        Returns
        -------
        PairedComparison object
        '''
        return PairedComparison(self, paired_comparison_probabilities(self.data, num_samples=num_samples))
        



class Result():
    def __init__(self, result_type : str, evals_data : EvalsData):
        self.result_type = result_type
        self.evals_data = evals_data

    def __repr__(self):
        return f"Result Obj: {self.result_type} (on {self.evals_data})"
    

class IndependentIntervals(Result):
    def __init__(self, evals_data: EvalsData, error_bars : np.ndarray):
        super().__init__("Independent Error Bars", evals_data)
        self.error_bars = error_bars

    def print(self):
        # TODO: do some prettyprint table or something here
        print(f"Error Bars:")
        max_name_length = max([len(name) for name in self.evals_data.model_names])
        for i, llm in enumerate(self.evals_data.model_names):
            print(f"{llm.rjust(max_name_length)}: {self.error_bars[i]}")

    def plot(self, filename : str = None):
        plot_error_bars(self, title="Accuracy Error Bars", filename=filename)


class IndependentComparison(Result):
    def __init__(self, evals_data: EvalsData, comparison_matrix : np.ndarray):
        super().__init__("Independent Comparison", evals_data)
        self.comparison_matrix = comparison_matrix

    def print(self):
        # TODO: do some prettyprint table or something here
        print(f"Comparison Matrix:")
        print(self.to_dataframe())

    def to_dataframe(self):
        return pd.DataFrame(self.comparison_matrix, columns=self.evals_data.model_names, index=self.evals_data.model_names)

    def plot(self, filename : str = None):
        plot_comparison_matrix(self, title="Independent Comparison Matrix", filename=filename)
        


class PairedComparison(Result):
    def __init__(self, evals_data: EvalsData, comparison_matrix : np.ndarray):
        super().__init__("Paired Comparison", evals_data)
        self.comparison_matrix = comparison_matrix

    def print(self):
        # TODO: do some prettyprint table or something here
        print(f"Comparison Matrix:")
        print(self.to_dataframe())
        
    def to_dataframe(self):
        return pd.DataFrame(self.comparison_matrix, columns=self.evals_data.model_names, index=self.evals_data.model_names)

    def plot(self, filename : str = None):
        plot_comparison_matrix(self, title="Paired Comparison Matrix", filename=filename)

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

def independent_credible_intervals(data, alpha, prior=(1, 1)):
    '''
    Compute the credible interval for the parameter of a Bernoulli distribution. 

    Parameters
    ----------
    data: np.ndarray
        Binary data, shape [M,Q] where M is the number of models and Q is the number of questions
    alpha: float, or list of floats
        Significance levels
    prior: tuple
        Beta distribution parameters (default: (1, 1) i.e. uniform)

    Returns
    -------
    credible_interval: tuple
        Lower and upper bounds of the credible interval (with extra batch dimensions and potentially a significance level dimension prepended)
    '''
    if isinstance(alpha, float):
        alpha = np.array([alpha])
    else:
        alpha = np.array(alpha)
    
    output = np.zeros(alpha.shape + data.shape[:-1] + (2,))
    posterior = stats.beta(data.sum(-1) + prior[0], data.shape[-1] - data.sum(-1) + prior[1])
    for i, a in enumerate(alpha):
        interval = posterior.interval(1 - a)
        output[i, ..., :] = np.column_stack(interval)
    
    if len(alpha) == 1:
        return output[0]
    return output

## Independent Comparison
def independent_comparison_probabilities(data, num_samples=10_000, prior=(1, 1)):
    '''
    Compute the credible interval for the parameter of a Bernoulli distribution. 

    Parameters
    ----------
    data: np.ndarray
        Binary data, shape [M,Q] where M is the number of models and Q is the number of questions
    num_samples: int
        Number of samples to draw from the posterior
    prior: tuple
        Beta distribution parameters (default: (1, 1) i.e. uniform)

    Returns
    -------
    comparison_matrix: np.ndarray [M,M]
        Matrix containing entries of posterior probability of the difference in theta_A and theta_B being greater than 0
        for all pairs of models A and B
    '''
    assert len(data.shape) == 2, f"Data must be a 2D array (found {len(data.shape)} dimensions)"
    M = data.shape[0]
    Q = data.shape[1]

    posterior_samples = get_bayes_posterior(data, prior).rvs(size=(num_samples,M))
    assert posterior_samples.shape == (num_samples,M)

    comparison_matrix = (posterior_samples[:, :, None] > posterior_samples[:, None, :]).mean(0)
    assert comparison_matrix.shape == (M,M)

    # set diagonal to nan
    np.fill_diagonal(comparison_matrix, np.nan)

    return comparison_matrix

################################################################
## Paired Comparison
def paired_comparison_probabilities(data, num_samples=10_000):
    '''
    Use Dirichlet-multiinomial model to compare all pairs of models present in the data

    data: np.ndarray
        Binary data, shape [M,Q] where M is the number of models and Q is the number of questions
    num_samples: int
        Number of samples to draw from the posterior

    Returns
    -------
    comparison_matrix: np.ndarray [M,M]
        Matrix containing entries of posterior probability of the difference in theta_A and theta_B being greater than 0
        for all pairs of models A and B
    '''
    assert len(data.shape) == 2, f"Data must be a 2D array (found {len(data.shape)} dimensions)"
    M = data.shape[0]
    Q = data.shape[1]

    data_A = np.repeat(data, M, axis=0)
    data_B = np.tile(data, (M, 1))

    assert data_A.shape == data_B.shape == (M*M, Q)

    if len(data_A.shape) == 1:
        data_A = data_A[None,:]

    if len(data_B.shape) == 1:
        data_B = data_B[None,:]

    repeats = data_A.shape[0]
    Q = data_A.shape[-1]

    assert data_A.shape == data_B.shape == (repeats, Q)

    # create the 2x2 contingency table (flattened)
    table = np.zeros((repeats, 4))
    table[:, 0] = (data_A * data_B).sum(-1)             # S = A correct,   B correct
    table[:, 1] = (data_A * (1 - data_B)).sum(-1)       # T = A correct,   B incorrect
    table[:, 2] = ((1 - data_A) * data_B).sum(-1)       # U = A incorrect, B correct
    table[:, 3] = ((1 - data_A) * (1 - data_B)).sum(-1) # V = A incorrect, B incorrect

    dir_post = np.zeros((repeats, num_samples, 4))    
    for r in range(repeats):
        # closed form solution
        dir_post[r] = stats.dirichlet(table[r] + 1).rvs(size=(num_samples,))

    assert np.allclose(dir_post.sum(-1), 1)

    # theta_As_post = dir_post[..., (0,1)].sum(-1)  # S + T
    # theta_Bs_post = dir_post[..., (0,2)].sum(-1)  # S + U
    # diff_post = theta_As_post - theta_Bs_post     # (S + T) - (S + U) = T - U
    diff_post = dir_post[..., 1] - dir_post[..., 2]

    posterior_prob = (diff_post > 0).mean(-1) 
    assert posterior_prob.shape == (M*M,)

    comparison_matrix = posterior_prob.reshape(M, M)

    # set diagonal to nan
    np.fill_diagonal(comparison_matrix, np.nan)
    
    return comparison_matrix

################################################################

def analyse(data : Union[np.ndarray, pd.DataFrame], model_names : list[str] = None):
    '''
    Load in the data and model names and return an EvalsData object.

    Parameters
    ----------
    data: np.ndarray or pd.DataFrame
        Binary data, shape [M,Q] where M is the number of models and Q is the number of questions
    model_names: list of str (optional)
        Names of the models (default: None)
    '''
    return EvalsData(data, model_names)

################################################################
## Plotting


def plot_error_bars(independent_intervals: IndependentIntervals, title: str, filename: str = None):
    assert isinstance(independent_intervals, IndependentIntervals)
    assert isinstance(title, str)
    # assert isinstance(filename, str)

    llm_names = independent_intervals.evals_data.model_names 
    error_bars = independent_intervals.error_bars

    # only take first alpha value (significance level) if there are multiple
    if len(error_bars.shape) == 4:
        error_bars = error_bars[0]

    means = independent_intervals.evals_data.data.mean(-1)
    fig, ax = plt.subplots()
    # ax.errorbar(llm_names, means, yerr=np.stack([means - error_bars[:,0], error_bars[:,1] - means]), fmt='o')
    # plot means and error bars separately as they might not line up
    ax.bar(llm_names, means)
    positions = np.arange(len(llm_names))
    for i, llm in enumerate(llm_names):
        ax.bxp([{
                # 'med': error_bars_0_5[llm][method_idx].mean(),
                'med': means[i],
                'q1': means[i],
                'q3': means[i],
                'whislo': error_bars[i,0],
                'whishi': error_bars[i,1],
                'caps': error_bars[i],
                'fliers': [],
                'mean': []
            }], 
            positions=[positions[i]], 
            widths=1, 
            showfliers=False, 
            boxprops=dict(color='#444'), 
            # medianprops=dict(color=COLOURS[method], linewidth=0), 
            # whiskerprops=dict(color=COLOURS[method], zorder=50), 
            # capprops=dict(color=COLOURS[method])
    )
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(llm_names, rotation=45, ha='right')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

def plot_comparison_matrix(independent_comparison: Union[IndependentComparison, PairedComparison], title: str, filename: str = None):
    assert isinstance(independent_comparison, (IndependentComparison, PairedComparison))
    assert isinstance(title, str)
    # assert isinstance(filename, str)

    llm_names = independent_comparison.evals_data.model_names
    comparison_matrix = independent_comparison.comparison_matrix

    fig, ax = plt.subplots()
    cax = ax.matshow(comparison_matrix, cmap='RdYlGn')
    fig.colorbar(cax)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(range(len(llm_names)))
    ax.set_yticks(range(len(llm_names)))
    ax.set_xticklabels(llm_names, rotation=45, ha='right')
    ax.set_yticklabels(llm_names)
    # plt.xticks(rotation=45, ha='right')
    ax.set_title(title+"\n(Probability of Model A > Model B)")
    ax.set_xlabel('Model B')
    ax.set_ylabel('Model A')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()
