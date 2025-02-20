import numpy as np

def sample_bernoulli(n, theta, shape=None, seed=None):
    '''
    Sample n Bernoulli random variables with probability theta.

    Parameters
    ----------
    n: int
        Number of samples
    theta: float
        Probability of success
    shape: tuple
        Extra dimensions for the output

    Returns
    -------
    samples: np.ndarray
        shape == (shape[0], shape[1], ..., shape[-1], n)
    '''
    if seed is not None:
        np.random.seed(seed)

    if shape is None:
        shape = n
    else:
        shape = shape + (n,)
    return np.random.binomial(1, theta, shape)