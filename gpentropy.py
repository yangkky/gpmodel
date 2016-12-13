import numpy as np
import pandas as pd
from sys import exit
import itertools


class GPEntropy(object):

    """ An object for calculating the entropy of Gaussian Processes.

    Attributes:
        kernel (GPKernel): kernel to use for calculating covariances
        var_n (float): measurement variance
        hypers (iterable): hyperparameters for the kernel
        observed (pd.DataFrame): observed inputs
        _Ky (np.matrix): noisy covariance matrix [K+var_n*I]
        _L (np.matrix): lower triangular Cholesky decomposition of Ky
    """

    def __init__(self, kernel=None, hypers=None, var_n=0,
                 observations=None, model=None):
        """ Create a new GPEntropy object.

        Create a new GPEntropy object either from a GPModel or by specifying
        the kernel, hyperparameters, and observations. If both are given, the
        model attributes will be used, and the others ignored.

        Optional keyword parameters:
            model (GPModel)
            kernel (GPKernel)
            hypers (iterable)
            var_n (float)
            observations (pd.DataFrame)
        """
        if model is not None:
            self.kernel = model.kern
            self.var_n = model.hypers[0]
            self.hypers = model.hypers[1::]
            self.observed = model.X_seqs
            self.observed.index = [str(_) for _ in range(len(self.observed))]
            self.kernel.train(self.observed)
            self._Ky = model._Ky
            self._L = model._L
        else:
            self.kernel = kernel
            self.hypers = hypers
            self.var_n = var_n
            self.observed = pd.DataFrame()
            self.observe(observations)

    def entropy(self, X):
        """ Calculate the entropy for a given set of points.

        The entropy is calculated from the posterior covariance of
        the given points conditioned on the observed points for the
        GPEntropy object.

        Parameters:
            X (np.ndarray): new inputs at which to calculate the entropy.

        Returns:
            H (float)
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, index=[str(i) for i in range(len(X))])
        K = self._posterior_covariance(X)
        L = np.linalg.cholesky(K)
        D = len(X)
        return np.sum(np.log(np.diag(L))) + 0.5 * D * np.log(2*np.pi*np.exp(1))

    def expected_entropy(self, X, probabilities):
        """ Calculate the expected entropy for a given set of points.

        The expected entropy is the sum of theentropy of each subset
        in the power set of X, weighted by the probability that that
        is exactly the subset that is functional.

        Parameters:
            X (np.ndarray): new inputs at which to calculate the entropy.
            probabilities (np.ndarray): probability that each input
                is functional

        Returns:
            H (float)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(probabilities, pd.DataFrame):
            probabilities = probabilities.values
        total = 0
        data = np.concatenate((X, probabilities), axis=1)
        for r in range(1, len(X)+1):
            for subset_inds in itertools.combinations(range(len(data)), r):
                subset = data[subset_inds, :]
                sub_X = subset[:, 0:-1]
                sub_probs = subset[:, -1]
                H = self.entropy(sub_X)
                prob_for = np.prod(sub_probs)
                comp = list(set(range(len(data))) - set(subset_inds))
                prob_against = np.prod([1-p for p in data[comp, -1]])
                total += H * prob_for * prob_against
        return total

    def _posterior_covariance(self, X):
        """ Calculate the covariance conditioned on observations.

        Parameters:
            X (np.ndarray): new inputs at which to calculate the entropy.

        Returns:
            K (np.matrix)
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, index=[str(i) for i in range(len(X))])
        cov = self.kernel.make_K(X, hypers=self.hypers)
        k_off = np.matrix(self._k_star(X))
        v = np.linalg.lstsq(self._L, k_off.T)[0]
        return cov - v.T * v

    def maximize_entropy(self, X, n):
        """ Choose the subset of X that maximizes the entropy.

        Uses the lazy-greedy algorithm to choose a subset of X that
        maximizes the entropy, conditioned on the observed sequences.

        Parameters:
            X (np.ndarray): set of inputs
            n (int): number of inputs to select

        Returns:
            selected (np.ndarray): selected inputs
            H (float): entropy of selected inputs
            selected: positional indices of selected inputs relative
                to original X
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
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
            selected (np.ndarray): selected inputs
            H (float): expected entropy of selected inputs
            selected: positional indices of selected inputs relative
                to original X
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
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
            selected (np.ndarray): selected inputs
            H (float): function value of selected inputs
            selected: positional indices of selected inputs relative
                to original X
        """
        UBs = [np.inf for _ in X]
        UBs = pd.DataFrame(UBs, columns=['UB'])
        selected = []
        H = 0
        for i in range(n):
            found = False
            while not found:
                try_inds = selected + [UBs.index[0]]
                new_args = {k: kwargs[k][try_inds] for k in kwargs.keys()}
                del_H = func(X[try_inds], **new_args) - H
                UBs.iloc[0, 0] = del_H
                found = (del_H >= UBs.iloc[1, 0])
                if found:
                    break
                less_than = (del_H < UBs.iloc[1::]).values.T[0]
                if less_than.all():
                    new_inds = list(UBs.index[1::]) + [UBs.index[0]]
                else:
                    first = np.nonzero(~less_than)[0][0]+1
                    new_inds = list(UBs.index[1:first]) + [UBs.index[0]]
                    new_inds += list(UBs.index[first::])
                UBs = UBs.reindex(new_inds)
            selected.append(UBs.index[0])
            H += UBs.iloc[0, 0]
            UBs = UBs.drop(UBs.index[[0]])
        return X[selected], H, selected

    def observe(self, observations):
        """ Update the observations.

        Parameters:
            observations (pd.DataFrame): new observations

        Returns:
            None
        """
        self.observed = pd.concat([self.observed, observations])
        # reindex everything
        self.observed.index = [str(_) for _ in range(len(self.observed))]
        self.kernel.train(self.observed)
        K = self.kernel.make_K(self.observed, hypers=self.hypers)
        self._Ky = K+self.var_n*np.identity(len(self.observed))
        self._L = np.linalg.cholesky(self._Ky)

    def _k_star(self, new_x):
        """ Calculate covariance of new inputs with observed inputs.

        Parameters:
            new_x (np.ndarray): new inputs

        Returns:
            k (np.ndarray)
        """
        observed = self.observed
        if isinstance(new_x, pd.DataFrame):
            new_x = new_x.values
        if len(np.shape(new_x)) == 2:
            return np.array([[self.kernel.calc_kernel(x, o, self.hypers)
                              for o in observed.index] for x in new_x])
        elif len(np.shape(new_x)) == 1:
            return np.array([[self.kernel.calc_kernel(new_x, o, self.hypers)
                              for o in observed.index]])
