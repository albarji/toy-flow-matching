"""Functions to compute distances between distributions."""

import numpy as np
import ot # POT library for optimal transport

def wasserstein_distance(X, Y, max_samples=1000):
    """Compute the Wasserstein distance between samples of two distributions X and Y.
    
    Parameters:
        X: np.ndarray of shape (n_samples_X, n_features)
        Y: np.ndarray of shape (n_samples_Y, n_features)
        max_samples: int, maximum number of samples to use for distance computation (for efficiency)

    Returns: float, the Wasserstein distance between the distributions represented by X and Y.
    """
    # Subsample to the requested value, ensuring both samples have the same size
    # Also randomize the subsampling to get a more representative estimate of the distance
    n_samples = min(len(X), len(Y), max_samples)
    X_subsample = X[np.random.permutation(len(X))[:n_samples]]
    Y_subsample = Y[np.random.permutation(len(Y))[:n_samples]]

    sol = ot.solve_sample(X_subsample, Y_subsample)
    return sol.value
    