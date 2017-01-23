import pytest
import os

import pandas as pd
import numpy as np
import math
import scipy
from scipy import stats
from sklearn import metrics, linear_model

from gpmodel import gpkernel
from gpmodel import gpmodel
from gpmodel import gpmean
from gpmodel import chimera_tools
from cholesky import chol

n = 100
d = 10
X = np.random.random(size=(n, d))
xa = X[[0]]
xb = X[[1]]
Xc = X[[2]]
X_test = np.random.random(size=(5, d))
kernel = gpkernel.SEKernel()
cov = kernel.cov(X, X, hypers=(1.0, 0.5))
variances = np.random.random(size=(n, ))
Y = np.random.multivariate_normal(np.zeros(n), cov=cov)
Y += np.random.normal(0, 0.2, n)


def test_init():
    model = gpmodel.GPClassifer(kernel)
    assert model.objective == model._log_ML
    assert model.kernel == kernel
    model = gpmodel.GPClassifer(kernel, guesses=(0.1, 0.1))
    assert model.guesses == (0.1, 0.1)


if __name__ == "__main__":
    test_init()
    test_normalize()
    test_K()
    test_ML()
    test_fit()
    test_LOO()
    test_predict()
