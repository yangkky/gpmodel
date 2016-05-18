import numpy as np
import pandas as pd
from sys import exit
#import chimera_tools

class GPMean(object):

    """ A Gaussian process mean function.

    Attributes:
    """

    def __init__(self, clf, **kwargs):
        """ Create a new GPMean object."""
        self._clf = clf(**kwargs)

    def fit(self, X, Y):
        """ Fit the mean function."""
        self._clf.fit(X, Y)

    def mean(self, X):
        """ Calculate the mean function. """
        return self._clf.predict(X)





