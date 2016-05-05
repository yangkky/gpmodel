import numpy as np
import pandas as pd
from sys import exit
import itertools

class GPEntropy(object):

    """ An object for calculating the entropy of GPs. """

    def __init__(self, kernel=None, hypers=None, var_n=0, observations=None, model=None):
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
        K = self.posterior_covariance(X)
        L = np.linalg.cholesky(K)
        D = len(X)
        return np.sum(np.log(np.diag(L))) + 0.5 * D * np.log(2*np.pi*np.exp(1))

    def expected_entropy(self, X, probabilities):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(probabilities, pd.DataFrame):
            probabilities = probabilities.values
        total = 0
        data = np.concatenate((X, probabilities), axis=1)
        for r in range(1,len(X)+1):
            for subset_inds in itertools.combinations(xrange(len(data)), r):
                subset = data[subset_inds,:]
                sub_X = subset[:,0:-1]
                sub_probs = subset[:,-1]
                H = self.entropy(sub_X)
                prob_for = np.prod(sub_probs)
                comp = list(set(xrange(len(data))) - set(subset_inds))
                prob_against = np.prod([1-p for p in data[comp,-1]])
                total += H * prob_for * prob_against
        return total

    def posterior_covariance(self, X):
        cov = self.kernel.make_K(X, hypers=self.hypers)
        k_off = np.matrix(self.k_star(X))
        v = np.linalg.lstsq(self._L,k_off.T)[0]
        return cov - v.T*v

    def maximize_expected_entropy(self, X, probabilities, n):
        if isinstance(X, pd.DataFrame):
            X = X.values
        UBs = [np.inf for _ in X]
        UBs = pd.DataFrame(UBs, columns=['UB'])
        selected = []
        H = 0
        for i in range(n):
            found = False
            while not found:
                try_inds = selected + [UBs.index[0]]
                del_H = self.expected_entropy(X[try_inds],
                                              probabilities[try_inds]) - H
                UBs.iloc[0,0] = del_H
                found = (del_H > UBs.iloc[1,0])
                if found:
                    break
                less_than = (del_H < UBs.iloc[1::]).values.T[0]
                if less_than.all():
                    new_inds = list(UBs.index[1::]) + [UBs.index[0]]
                else:
                    first = np.nonzero(~less_than)[0][0]+1
                    new_inds = list(UBs.index[1:first]) + \
                    [UBs.index[0]] + list(UBs.index[first::])
                UBs = UBs.reindex(new_inds)
            selected.append(UBs.index[0])
            H += UBs.iloc[0,0]
            UBs = UBs.drop(UBs.index[[0]])
        return X[selected], H, selected

    def observe(self, observations):
        self.observed = pd.concat([self.observed, observations])
        # reindex everything
        self.observed.index = [str(_) for _ in range(len(self.observed))]
        self.kernel.train(self.observed)
        K = self.kernel.make_K(self.observed, hypers=self.hypers)
        self._Ky = K+self.var_n*np.identity(len(self.observed))
        self._L = np.linalg.cholesky(self._Ky)

    def k_star(self, new_x):
        observed = self.observed
        if isinstance(new_x, pd.DataFrame):
            new_x = new_x.values
        if len(np.shape(new_x)) == 2:
            return np.array([[self.kernel.calc_kernel(x, o, self.hypers)
                              for o in observed.index] for x in new_x])
        elif len(np.shape(new_x)) == 1:
            return np.array([[self.kernel.calc_kernel(new_x, o, self.hypers)
                              for o in observed.index]])