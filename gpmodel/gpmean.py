import numpy as np
import pandas as pd
from sys import exit
from gpmodel import chimera_tools


class GPMean(object):

    """ A Gaussian process mean function.

    Attributes:
        clf
    """

    def __init__(self, clf=None, **kwargs):
        """ Create a new GPMean object."""
        if clf is not None:
            self._clf = clf(**kwargs)
        else:
            self._clf = clf
        self.means = None

    def fit(self, X, Y):
        """ Fit the mean function."""
        if self._clf is not None:
            self._clf.fit(X, Y)
        self.means = self.mean(X)

    def mean(self, X):
        """ Calculate the mean function. """
        if self._clf is not None:
            return self._clf.predict(X)
        else:
            return np.zeros((len(X), 1))


class StructureSequenceMean(GPMean):

    """ A Gaussian process mean function for proteins.

    Calculates X for the protein as a binary indicator vector for single
    sequence elements and binary elements that are in contact.

    Attributes:
        _sample_space (list)
        _contacts (list)
        _terms (list)
        clf
        means (np.ndarray)
    """

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
        self._clf.fit(X, Y)
        self.means = self._clf.predict(X)

    def mean(self, X_seqs):
        X, _ = self._make_X(X_seqs)
        return super(StructureSequenceMean, self).mean(X)

    def _make_X(self, X_seqs):
        if isinstance(X_seqs, pd.DataFrame):
            X_seqs = [''.join(row) for _, row in X_seqs.iterrows()]
        return chimera_tools.make_X(X_seqs,
                                    self._sample_space,
                                    self._contacts,
                                    terms=self._terms,
                                    collapse=False)
