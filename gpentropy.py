import numpy as np
import pandas as pd
from sys import exit
import itertools

class GPEntropy(object):

    """ An object for calculating the entropy of GPs. """

    def __init__(self, kernel, hypers, var_n=0, observations=None):
        self.kernel = kernel
        self.hypers = hypers
        self.var_n = var_n
        self.observed = pd.DataFrame()
        self.observe(observations)

    def entropy(self, X):
        K = self.posterior_covariance(X)
        D = len(X)
        return 0.5 * (np.log(np.linalg.det(K)) + D * np.log(2*np.pi*np.exp(1)))

    def expected_entropy(X, probabilities):
        total = 0
        for r in range(1,len(X)):
            for subset in itertools.combinations(X.index, r):
                H = entropy(X.loc[subset])
                prob_for = np.product([p for p in probabilities.loc[subset]])
                comp = list(set(X.index) - set(subset))
                prob_against = np.product([p for p in probabilities.loc[comp]])
                total += H * prob_for * prob_against
        return total


    def posterior_covariance(self, X):
        cov = self.kernel.make_K(X, hypers=self.hypers)
        k_off = np.matrix(self.k_star(X, self.observed))
        return cov - k_off * np.linalg.inv(K) * k_off.T


    def maximize_expected_entropy(self, sequences, probabilities, n):
        pass

    def observe(self, observations):
        self.observed = pd.concat([self.observed, observations])
        # this line might break
        K = self.kernel.make_K(self.observed, hypers=self.hypers)
        self._Ky = K+self.var_n*np.identity(len(self.observed))
        self._L = np.linalg.cholesky(self._Ky)

    def k_star(self, new_x, obs_x):
        return np.array([[self.kernel.calc_kernel(x, o, self.hypers) for o in obs_x]
                         for x in new_x])