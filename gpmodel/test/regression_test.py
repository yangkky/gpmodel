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
class_Y = np.random.choice((1, -1), size=(n,))
alpha = 1e-1
func = gpmean.GPMean(linear_model.Lasso, alpha=alpha)
X_test = np.random.random(size=(5, d))
kernel = gpkernel.SEKernel()
cov = kernel.cov(X, X, hypers=(1.0, 0.5))
variances = np.random.random(size=(n, ))
Y = np.random.multivariate_normal(np.zeros(n), cov=cov)
Y += np.random.normal(0, 0.2, n)


def test_init():
    model = gpmodel.GPRegressor(kernel)
    assert np.allclose(model.mean_func.mean(X), np.zeros((len(X), )))
    assert model.objective == model._log_ML
    assert model.kernel == kernel
    assert model.guesses is None
    model = gpmodel.GPRegressor(kernel, objective='LOO_log_p')
    assert model.objective == model._LOO_log_p
    model = gpmodel.GPRegressor(kernel, guesses=(0.1, 0.1, 0.1))
    assert model.guesses == (0.1, 0.1, 0.1)


def test_normalize():
    model = gpmodel.GPRegressor(kernel)
    m, s, normed = model._normalize(Y)
    assert np.isclose(m, Y.mean())
    assert np.isclose(s, Y.std())
    assert np.allclose(normed, (Y - m) / s)
    model.std = s
    model.mean = m
    assert np.allclose(Y, model.unnormalize(normed))


def test_K():
    model = gpmodel.GPRegressor(kernel)
    model.kernel.fit(X)
    K, Ky = model._make_Ks((1, 1, 1))
    assert np.allclose(K, kernel.cov(X, X))
    assert np.allclose(Ky, K + np.diag(np.ones(len(K))))
    model.variances = variances
    K, Ky = model._make_Ks((1, 1))
    assert np.allclose(K, kernel.cov(X, X))
    assert np.allclose(Ky, K + np.diag(variances))


def test_ML():
    model = gpmodel.GPRegressor(kernel)
    model.kernel.fit(X)
    model.normed_Y = model._normalize(Y)[2]
    model._ell = len(Y)
    hypers = np.random.random(size=(3,))
    y_mat = model.normed_Y.reshape((n, 1))
    K, Ky = model._make_Ks(hypers)
    first = 0.5 * y_mat.T @ np.linalg.inv(Ky) @ y_mat
    second = 0.5 * np.log(np.linalg.det(Ky))
    third = model._ell / 2.0 * np.log(2 * np.pi)
    actual = first + second + third
    assert np.isclose(actual, model._log_ML(hypers))


def test_LOO():
    model = gpmodel.GPRegressor(kernel)
    model.fit(X, Y)
    hypers = np.random.random(size=(3,))
    LOOs = model.LOO_res(hypers)
    LP = model._LOO_log_p(hypers)


def test_fit():
    model = gpmodel.GPRegressor(kernel)
    model.fit(X, Y)
    assert np.allclose(model.X, X)
    assert np.allclose(model.Y, Y)
    m, s, normed = model._normalize(Y)
    assert np.allclose(model.normed_Y, normed)
    assert np.isclose(m, model.mean)
    assert np.isclose(s, model.std)
    vn, s0, ell = model.hypers
    K = kernel.cov(X, X, (s0, ell))
    Ky = K + np.diag(vn * np.ones(len(K)))
    ML = model._log_ML((vn, s0, ell))
    L, p, _ = chol.modified_cholesky(Ky)
    alpha = np.linalg.inv(Ky) @ normed.reshape((n, 1))
    assert np.isclose(model.ML, ML)
    assert np.allclose(model._K, K)
    assert np.allclose(model._Ky, Ky)
    assert np.allclose(model._L, L)
    assert np.allclose(model._p, p)
    assert np.allclose(model._alpha, alpha)


def test_predict():
    model = gpmodel.GPRegressor(kernel)
    model.fit(X, Y)
    h = model.hypers[1::]
    m, s, normed = model._normalize(Y)
    k_star = model.kernel.cov(X_test, X, hypers=h)
    k_star_star = model.kernel.cov(X_test, X_test, hypers=h)
    K = kernel.cov(X, X, h)
    Ky = K + np.diag(model.hypers[0] * np.ones(len(K)))
    means = k_star @ np.linalg.inv(Ky) @ normed.reshape(len(Y), 1)
    means = means * s + m
    var = k_star_star - k_star @ np.linalg.inv(Ky) @ k_star.T
    var *= s ** 2
    m, v = model.predict(X_test)
    assert np.allclose(v, var)
    assert np.allclose(means, m)


if __name__ == "__main__":
    test_init()
    test_normalize()
    test_K()
    test_ML()
    test_fit()
    test_LOO()
    test_predict()
    # To Do:
    # Test LOO_res and LOO_log_p and fitting with LOO_log_p
    # Test with mean functions
    # Test with given variances
