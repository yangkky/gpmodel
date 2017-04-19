import numpy as np
import pandas as pd
from sys import exit
from itertools import chain, combinations


class GPEntropy(object):

    """ An object for calculating the entropy of Gaussian Processes.

    Attributes:
        kernel (GPKernel): kernel to use for calculating covariances
        var_n (float): measurement variance
        hypers (iterable): hyperparameters for the kernel
        observed (np.ndarray): observed inputs
        index (iterable): index for observed
        _Ky (np.matrix): noisy covariance matrix [K+var_n*I]
        _L (np.matrix): lower triangular Cholesky decomposition of Ky
    """

    def __init__(self, kernel, hypers, var_n=0,
                 observations=None):
        """ Create a new GPEntropy object.

        Create a new GPEntropy object either from a GPModel or by specifying
        the kernel, hyperparameters, and observations. If both are given, the
        model attributes will be used, and the others ignored.

        Optional keyword parameters:
            kernel (GPKernel)
            hypers (iterable)
            var_n (float)
            observations (pd.DataFrame)
        """
        self.kernel = kernel
        self.hypers = hypers
        self.var_n = var_n
        self.observed = None
        if observations is not None:
            self.observe(observations)

    def entropy(self, X):
        """ Calculate the entropy for a given set of points.

        The entropy is calculated from the posterior covariance of
        the given points conditioned on the observed points for the
        GPEntropy object, according to Equation A.20 of RW.

        Parameters:
            X (np.ndarray): new inputs at which to calculate the entropy.

        Returns:
            H (float)
        """
        K = self._posterior_covariance(X)
        L = np.linalg.cholesky(K)
        D = X.shape[0]
        return np.sum(np.log(np.diag(L))) + 0.5 * D * np.log(2*np.pi*np.exp(1))

    def expected_entropy(self, X, probabilities):
        """ Calculate the expected entropy for a given set of points.

        The expected entropy is the sum of the entropy of each subset
        in the power set of X, weighted by the probability that that
        is exactly the subset that is functional.

        Parameters:
            X (np.ndarray): new inputs at which to calculate the entropy.
            probabilities (np.ndarray): probability that each input
                is functional. Should be 1-dimensional.

        Returns:
            H (float)
        """
        total = 0
        for inds in chain.from_iterable(combinations(range(len(X)), r)
                                        for r in range(1, len(X) + 1)):
            this_X = X[np.array(inds), :]
            this_P = probabilities[np.array(inds)]
            H = self.entropy(this_X)
            prob_for = np.prod(this_P)
            not_inds = list(set(list(range(len(X)))) - set(inds))
            if len(not_inds) > 0:
                not_P = probabilities[np.array(not_inds)]
                prob_against = np.prod(1 - not_P)
            else:
                prob_against = 1.0
            total += H * prob_for * prob_against
        return total

    def _posterior_covariance(self, X):
        """ Calculate the covariance conditioned on observations.

        Parameters:
            X (np.ndarray): new inputs at which to calculate the entropy.

        Returns:
            K (np.ndarray): n_X x n_X
        """
        k_star_star = self.kernel.cov(X, X, hypers=self.hypers)
        k_star = self.kernel.cov(X, self.observed, hypers=self.hypers)
        v = np.linalg.lstsq(self._L, k_star.T)[0]
        return k_star_star - v.T @ v + self.var_n * np.eye(len(X))

    def maximize_entropy(self, X, n):
        """ Choose the subset of X that maximizes the entropy.

        Uses the lazy-greedy algorithm to choose a subset of X that
        maximizes the entropy, conditioned on the observed sequences.

        Parameters:
            X (np.ndarray): set of inputs
            n (int): number of inputs to select

        Returns:
            H (float): entropy of selected inputs
            selected (list): rows of X that were selected
        """
        return self._lazy_greedy(X, self.entropy, n)

    def maximize_expected_entropy(self, X, probabilities, n):
        """ Choose the subset of X that maximizes the expected entropy.

        Uses the lazy-greedy algorithm to choose a subset of X that
        maximizes the exptected entropy, conditioned on the observed
        sequences.

        Parameters:
            X (np.ndarray): set of inputs
            probabilities (np.ndarray): probability that each input
                is functional
            n (int): number of inputs to select

        Returns:
            H (float): expected entropy of selected inputs
            selected (list): rows of X that were selected
        """
        return self._lazy_greedy(X, self.expected_entropy, n,
                                 probabilities=probabilities)

    def _lazy_greedy(self, X, func, n, **kwargs):
        """ Implementation of lazy-greedy for optimizing set functions.

        Parameters:
            X (np.ndarray): set of inputs
            func (function): submodular set function to maximize
            n (int): number of inputs to select
            **kwargs: additional arguments for func. Must be indexed
                identically with X.

        Returns:
            H (float): function value of selected inputs
            selected (list): rows of X that were selected
        """
        # First column is upper bound on entropy gain by adding that row of X
        # Second column is the row of X
        UBs = np.array([[np.inf for _ in X],
                        list(range(len(X)))]).T
        # Indices that would sort UBs
        sort_inds = np.array(range(len(X)))
        # The rows of X chosen
        selected = []
        H = 0
        for i in range(n):
            found = False
            while not found:
                previous_best = sort_inds[-1]
                try_inds = selected + [UBs[sort_inds[-1], 1]]
                try_inds = [int(i) for i in try_inds]
                new_args = {k: kwargs[k][try_inds] for k in kwargs.keys()}
                del_H = func(X[try_inds], **new_args) - H
                UBs[sort_inds[-1], 0] = del_H
                sort_inds = np.argsort(UBs[:, 0])
                found = sort_inds[-1] == previous_best
            selected.append(UBs[sort_inds[-1], 1])
            H += UBs[sort_inds[-1], 0]
            UBs = np.delete(UBs, sort_inds[-1], axis=0)
            sort_inds = np.argsort(UBs[:, 0])
            selected = [int(i) for i in selected]
        return H, selected

    def observe(self, observations):
        """ Update the observations.

        Parameters:
            observations (pd.DataFrame): new observations

        Returns:
            None
        """
        if self.observed is not None:
            self.observed = np.concatenate([self.observed, observations])
        else:
            self.observed = observations
        K = self.kernel.cov(self.observed, self.observed, hypers=self.hypers)
        self._Ky = K + self.var_n*np.identity(len(K))
        self._L = np.linalg.cholesky(self._Ky)
