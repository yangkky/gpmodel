import numpy as np
import pandas as pd
from sys import exit
import chimera_tools

class GPMean(object):

    """ A Gaussian process mean function.

    Attributes:
    """

    def __init__(self, clf=None, **kwargs):
        """ Create a new GPMean object."""
        if clf is not None:
            self._clf = clf(**kwargs)
        else:
            self._clf = clf

    def fit(self, X, Y):
        """ Fit the mean function."""
        if self._clf is not None:
            self._clf.fit(X, Y)
        else:
            return

    def mean(self, X):
        """ Calculate the mean function. """
        if self._clf is not None:
            return self._clf.predict(X)
        else:
            return np.zeros(len(X))

class StructureSequenceMean(GPMean):
    def __init__(self, sample_space, contacts, clf, **kwargs):
        self._sample_space = sample_space
        self._contacts = contacts
        self._terms = None
        super(StructureSequenceMean, self).__init__(clf=clf, **kwargs)

    def fit(self, X_seqs, Y):
        self._terms = None
        if isinstance(X_seqs, pd.DataFrame):
            X_seqs = [''.join(row) for _, row in X_seqs.iterrows()]
        X, self._terms = self._make_X(X_seqs)
        super(StructureSequenceMean, self).fit(X, Y)

    def mean(self, X_seqs):
        X, _ = self._make_X(X_seqs)
        return super(StructureSequenceMean, self).mean(X)

    def _make_X(self, X_seqs):
        if isinstance(X_seqs, pd.DataFrame):
            X_seqs = [''.join(row) for _, row in X_seqs.iterrows()]
        X, terms = chimera_tools.make_X(X_seqs,
                                    self._sample_space,
                                    self._contacts,
                                    terms = self._terms)
        return X, terms






