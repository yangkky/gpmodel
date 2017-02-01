import pytest
import os

import pandas as pd
import numpy as np
from scipy import stats
from scipy import integrate
from sklearn import metrics, linear_model

from gpmodel import gpkernel
from gpmodel import gpmodel
from gpmodel import gpmean
from cholesky import chol

n = 200
d = 10
X = np.random.random(size=(n, d))
xa = X[[0]]
xb = X[[1]]
Xc = X[[2]]
X_test = np.random.random(size=(5, d))
kernel = gpkernel.SEKernel()
hypers = np.array([1.87081465, 1.4759388])
cov = kernel.cov(X, X, hypers)
F = np.random.multivariate_normal(np.zeros(n), cov=cov)
P = 1.0 / (1.0 + np.exp(-F))
Y = np.array([np.random.choice([-1, 1], p=[1 - p, p]) for p in P])


def test_init():
    model = gpmodel.GPClassifier(kernel)
    assert model.objective == model._log_ML
    assert model.kernel == kernel
    model = gpmodel.GPClassifier(kernel, guesses=(0.1, 0.1))
    assert model.guesses == (0.1, 0.1)


def test_probs():
    model = gpmodel.GPClassifier(kernel)
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
    model = gpmodel.GPClassifier(kernel)
    model.Y = Y
    model.kernel.fit(X)
    f_hat = model._find_F(hypers)
    check_f_hat = cov @ ((Y + 1) / 2 - 1.0 / (1.0 + np.exp(-f_hat)))
    assert np.allclose(f_hat, check_f_hat)


def test_ML():
    model = gpmodel.GPClassifier(kernel)
    model.Y = Y
    model._ell = len(Y)
    model.kernel.fit(X)
    f_hat = model._find_F(hypers)
    q = model._logq(f_hat)
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


def test_fit():
    model = gpmodel.GPClassifier(kernel)
    model.fit(X, Y)
    assert model._n_hypers == kernel._n_hypers
    assert np.allclose(model.X, X)
    assert np.allclose(model.Y, Y)
    s0, ell = model.hypers
    K = kernel.cov(X, X, (s0, ell))
    ML = model._log_ML((s0, ell))
    f_hat = model._find_F((s0, ell))
    this_P = 1.0 / (1.0 + np.exp(-Y * f_hat))
    W = -np.diag(-this_P * (1 - this_P))
    W_root = np.sqrt(W)
    trip_dot = W_root @ K @ W_root
    L, p, _ = chol.modified_cholesky(np.eye(len(Y)) + trip_dot)
    grad = np.diag(model._grad_log_logistic_likelihood(Y, f_hat))
    assert np.isclose(model.ML, ML)
    assert np.allclose(model._K, K)
    assert np.allclose(model._W_root, W_root)
    assert np.allclose(model._L, L)
    assert np.allclose(model._p, p)
    assert np.allclose(model._grad, grad)


def test_predict():
    model = gpmodel.GPClassifier(kernel)
    model.fit(X, Y)
    p, m, v = model.predict(X_test)
    h = model.hypers
    k_star = model.kernel.cov(X_test, X, hypers=h)
    k_star_star = model.kernel.cov(X_test, X_test, hypers=h)
    K = kernel.cov(X, X, h)
    f_hat = model._find_F(h)
    this_P = 1.0 / (1.0 + np.exp(-Y * f_hat))
    W = -np.diag(-this_P * (1 - this_P))
    means = k_star @ np.linalg.inv(K) @ f_hat.reshape(len(Y), 1)
    var = k_star_star - k_star @ np.linalg.inv(K + np.linalg.inv(W)) @ k_star.T
    assert np.allclose(v, var)
    assert np.allclose(means[:, 0], m)
    pi_star = np.zeros(len(X_test))
    span = 20.0
    for i, preds in enumerate(zip(means, np.diag(var))):
        f, va = preds
        pi_star[i] = integrate.quad(model._p_integral,
                                    -span * va + f,
                                    span * va + f,
                                    args=(f, va))[0]
    assert np.allclose(p, pi_star)


def test_score():
    model = gpmodel.GPClassifier(kernel)
    n_train = int(0.8 * n)
    X_train = X[0:n_train]
    X_test = X[n_train::]
    Y_train = Y[0:n_train]
    Y_test = Y[n_train::]
    model.fit(X_train, Y_train)
    assert 0 < model.score(X_test, Y_test) < 1.0


def test_pickles():
    model = gpmodel.GPClassifier(kernel)
    model.fit(X, Y)
    p1, m1, v1 = model.predict(X_test)
    model.dump('test.pkl')
    new_model = gpmodel.GPClassifier.load('test.pkl')
    os.remove('test.pkl')
    p2, m2, v2 = new_model.predict(X_test)
    assert np.allclose(m1, m2)
    assert np.allclose(v1, v2)
    assert np.allclose(p1, p2)


if __name__ == "__main__":
    test_init()
    test_probs()
    test_find_F()
    test_ML()
    test_fit()
    test_predict()
    test_score()
    test_pickles()
