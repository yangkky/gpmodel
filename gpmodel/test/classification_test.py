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
hypers = np.random.random(size=(2,)) + 1.0
cov = kernel.cov(X, X, hypers)
variances = np.random.random(size=(n, ))
F = np.random.multivariate_normal(np.zeros(n), cov=cov)
P = 1.0 / (1.0 + np.exp(-F))
Y = np.array([np.random.choice([-1, 1], p=[1 - p, p]) for p in P])


def test_init():
    model = gpmodel.GPClassifer(kernel)
    assert model.objective == model._log_ML
    assert model.kernel == kernel
    model = gpmodel.GPClassifer(kernel, guesses=(0.1, 0.1))
    assert model.guesses == (0.1, 0.1)


def test_probs():
    model = gpmodel.GPClassifer(kernel)
    actual_P = 1.0 / (1.0 + np.exp(-Y * F))
    assert np.allclose(model._logistic_likelihood(Y, F), actual_P)
    assert np.isclose(model._logistic_likelihood(Y[0], F[0]), actual_P[0])
    actual_log_P = np.log(np.prod(actual_P))
    assert np.allclose(model._log_logistic_likelihood(Y, F), actual_log_P)
    actual_grad = np.diag((Y + 1.0) / 2.0 - P)
    assert np.allclose(model._grad_log_logistic_likelihood(Y, F), actual_grad)
    actual_hess = np.diag(-P * (1 - P))
    assert np.allclose(model._hess(F), -actual_hess)


def test_find_F():
    model = gpmodel.GPClassifer(kernel)
    model.Y = Y
    model.kernel.fit(X)
    f_hat = model._find_F(hypers)
    check_f_hat = cov @ ((Y + 1) / 2 - 1.0 / (1.0 + np.exp(-f_hat)))
    assert np.allclose(f_hat, check_f_hat)


def test_ML():
    model = gpmodel.GPClassifer(kernel)
    model.Y = Y
    model._ell = len(Y)
    model.kernel.fit(X)
    f_hat = model._find_F(hypers)
    q = model._logq(f_hat, hypers)
    K = kernel.cov(X, X, hypers)
    this_P = 1.0 / (1.0 + np.exp(-Y * f_hat))
    W = -np.diag(-this_P * (1 - this_P))
    det_B = np.linalg.det(K) * np.linalg.det(np.linalg.inv(K) + W)
    f_hat = f_hat.reshape((len(f_hat), 1))
    first = -0.5 * f_hat.T @ np.linalg.inv(K) @ f_hat
    second = np.log(np.prod(this_P))
    third = -0.5 * np.log(det_B)
    actual_q = first + second + third
    assert np.isclose(actual_q, -q)
    assert np.isclose(q, model._log_ML(hypers))


if __name__ == "__main__":
    test_init()
    test_probs()
    test_find_F()
    test_ML()
