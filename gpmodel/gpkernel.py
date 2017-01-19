"""Kernel functions that calculate the covariance between two inputs."""

import numpy as np
import pandas as pd
from sys import exit
import abc


class BaseKernel(abc.ABC):

    """ A Gaussian Process kernel.

       Attributes:
           hypers (list)
    """

    @abc.abstractmethod
    def __init__(self):
        """ Create a GPKernel. """
        self.hypers = []

    @abc.abstractmethod
    def cov(self, X1, X2, hypers=None):
        """ Calculate the covariance. """
        return np.zeros((len(X1), len(X2)))


class PolynomialKernel(BaseKernel):

    """ A Polynomial kernel of the form (s0 + sp * x.T*x)^d

    Attributes:
    hypers (list): names of the hyperparameters required
    _deg (integer): degree of polynomial
    """

    def __init__(self, d):
        """ Initiate a polynomial kernel.

        Parameters:
            d (integer): degree of the polynomial
        """
        if not isinstance(d, int):
            raise TypeError('d must be an integer.')
        if d < 1:
            raise ValueError('d must be greater than or equal to 1.')
        self.hypers = ['sigma_0', 'sigma_p']
        self._deg = d

    def cov(self, X1, X2, hypers=None):
        """ Calculate the polynomial covariance matrix between X1 and X2.

        Parameters:
            X1 (np.ndarray):
            X2 (np.ndarray)
            hypers (iterable):

        Returns:
            K (np.ndarray)
        """
        if hypers is None:
            sigma_0, sigma_p = (1.0, 1.0)
        else:
            sigma_0, sigma_p = hypers
        return np.power(sigma_0 ** 2 + sigma_p ** 2 * X1.T @ X2, self._deg)


class BaseRadialKernel(BaseKernel):

    """ Base class for radial kernel functions. """

    def _distance(self, X1, X2):
        """ Calculates the squared distances between rows of X1 and X2.

        Each row of X1 and X2 represents one measurement. Each column
        represents a dimension.

        Parameters:
            X1 (np.ndarray): n x d
            X2 (np.ndarray): m x d

        Returns:
            D (np.ndarray): n x m
        """
        xs = np.array(xs)
        A = np.sum(xs ** 2, axis=1).reshape((len(X1), 1))
        B = np.sum(xs ** 2, axis=1).reshape((len(X2), 1)).T
        C = 2 * X1 @ X2.T
        return A + B - C


class MaternKernel(BaseRadialKernel):

    """ A Matern kernel with nu = 5/2 or 3/2.

    Attributes:
        hypers (list): names of the hyperparameters required
        _saved_X (dict): dict of saved index:X pairs
        nu (string): '3/2' or '5/2'
    """

    def __init__(self, nu):
        """ Initiate a Matern kernel.

        Parameters:
            nu (string): '3/2' or '5/2'
        """
        if nu not in ['3/2', '5/2']:
            raise ValueError("nu must be '3/2' or '5/2'")
        self.hypers = ['ell']
        self.nu = nu

    def cov(self, X1, X2, hypers=None):
        """ Calculate the Matern kernel between X1 and X2.

        Parameters:
            X1 (np.ndarray):
            X2 (np.ndarray)
            hypers (iterable): default is ell=1.0.

        Returns:
            K (np.ndarray)
        """
        if hypers = None:
            ell = 1.0
        else:
            ell = hypers[0]
        D = np.sqrt(self._distance(X1, X2))
        if self.nu == '3/2':
            M = (1.0 + np.sqrt(3.0) * D / ell) * np.exp(-np.sqrt(3) * D / ell)
        elif self.nu == '5/2':
            first = (1.0 + np.sqrt(5.0)*D/ell) + 5.0*np.power(D, 2)/3.0/ell**2
            second = np.exp(-np.sqrt(5.0) * D / ell)
            M = first * second
        return M


class SEKernel(BaseRadialKernel):

    """ A squared exponential kernel.

    Attributes:
        hypers (list)
        _d_squared (np.ndarray)
        _saved_X (dict)
    """

    def __init__(self):
        """ Initiate a SEKernel. """
        self.hypers = ['sigma_f', 'ell']

    def cov(self, X1, X2, hypers=None):
        """ Calculate the squared exponential kernel between x1 and x2.

        Parameters:
            X1 (np.ndarray):
            X2 (np.ndarray)
            hypers (iterable): default is (1.0, 1.0)

        Returns:
            K (np.ndarray)
        """
        if hypers = None:
            sigma_f, ell = 1.0, 1.0
        else:
            sigma_f, ell = hypers
        D = self._distance(X1, X2)
        return sigma_f**2 * np.exp(-0.5/np.power(ell, 2) * D)


class SumKernel(BaseKernel):

    """
    A kernel that sums over other kernels.

    Attributes:
        _kernels (list): list of member kernels
        hypers (list): the names of the hyperparameters
    """

    def __init__(self, kernels):
        '''
        Initiate a SumKernel containing a list of other kernels.

        Parameters:
            kernels(list): list of member kernels
            hypers (list): list of hyperparameter names
        '''
        self._kernels = kernels
        hypers = []
        for k in self._kernels:
            hypers += k.hypers
        self.hypers = [hypers[i] +
                       str(hypers[0:i].count(hypers[i]))
                       for i in range(len(hypers))]
        hypers_inds = [len(k.hypers) for k in self._kernels]
        hypers_inds = np.cumsum(np.array(hypers_inds))
        hypers_inds = np.insert(hypers_inds, 0, 0)
        self.hypers_inds = hypers_inds.astype(int)

    def calc_kernel(self, X1, X2, hypers=None):
        """ Calculate the sum kernel between two inputs.

        Parameters:
            X1 (np.ndarray):
            X2
            hypers (iterable): the hyperparameters. Default is to use
                the defaults for each kernel.

        Returns:
            K (np.ndarray)
        """
        if hypers is None:
            Ks = [kern.cov(X1, X2) for kern in self._kernels]
        else:
            hypers_inds = self.hypers_inds
            Ks = [kern.cov(X1, X2, hypers[hypers_inds[i]:
                                          hypers_inds[i+1]])
                  for i, kern in enumerate(self._kernels)]
        return np.sum(Ks)


class LinearKernel(BaseKernel):

    """ The linear (dot product) kernel for two inputs. """

    def __init__(self):
        """ Initiates a LinearKernel. """
        self.hypers = ['var_p']

    def calc_kernel(self, X1, X2, hypers=None):
        """ Calculate the linear kernel between x1 and x2.

        The linear kernel is the dot product of x1 and x2 multiplied
        by a scaling variable var_p.

        Parameters:
            x1 (np.ndarray)
            x2 (np.ndarray)
            hypers (iterable): default is var_p=1.0.

        Returns:
            K (np.ndarray)
        """
        vp = hypers[0]
        return vp * X1 @ X2.T
