import pytest
import os

import pandas as pd
import numpy as np
from scipy import stats
from scipy import integrate
from scipy import linalg
from sklearn import metrics, linear_model
import matplotlib.pyplot as plt

from gpmodel import gpkernel
from gpmodel import gpmodel
from gpmodel import gpmean
from cholesky import chol

np.random.seed(0)
n = 50
d = 1
n_classes = 3
# X = np.random.random(size=(n, d))
# X = np.sort(X)
X = np.linspace(0, 1, n).reshape((n, 1))
xa = X[[0]]
xb = X[[1]]
Xc = X[[2]]
X_test = np.random.random(size=(4, d))
k1 = gpkernel.MaternKernel('3/2')
k2 = gpkernel.MaternKernel('3/2')
k3 = gpkernel.MaternKernel('5/2')
h1 = np.array([1.0])
h2 = np.array([2.0])
h3 = np.array([0.5])
all_h = np.concatenate((h1, h2, h3))
cov1 = k1.cov(X, X, h1)
cov2 = k2.cov(X, X, h2)
cov3 = k3.cov(X, X, h3)
cov = np.stack((cov1, cov2, cov3), axis=2)
f1 = np.random.multivariate_normal(np.zeros(n), cov=cov1)
f2 = np.random.multivariate_normal(np.zeros(n), cov=cov2)
f3 = np.random.multivariate_normal(np.zeros(n), cov=cov3)
f = np.empty((n, n_classes))
f[:, 0] = f1
f[:, 1] = f2
f[:, 2] = f3
P = np.exp(f)
P /= np.sum(P, axis=1).reshape((n, 1))
Y_inds = np.array([np.random.choice(n_classes, p=p) for p in P])
Y = np.zeros_like(f)
for i, ind in enumerate(Y_inds):
    Y[i, ind] = 1
PI = np.concatenate([np.diag(p) for p in P.T], axis=0)


def test_soft_max():
    model = gpmodel.GPMultiClassifier([k1, k2, k3])
    assert np.allclose(P, model._softmax(f))
    assert np.allclose(PI, model._stack(P))


def test_aux_functions():
    model = gpmodel.GPMultiClassifier([k1, k2, k3])
    model.X = X
    model.Y = Y
    model._n_hypers = [k.fit(X) for k in model._kernels]
    hypers = model._split_hypers(all_h)
    for h, i in zip(hypers, [h1, h2, h3]):
        assert np.allclose(h, i)
    assert np.allclose(cov, model._make_K(all_h))

    A = np.random.random(size=(4, 2, 3))
    A_exp = model._expand(A)
    for i in range(3):
        assert np.allclose(A[:, :, i], A_exp[4*i:4*i+4, 2*i:2*i+2])


def test_find_F():
    model = gpmodel.GPMultiClassifier([k1, k2, k3])
    model.X = X
    model.Y = Y
    model._n_hypers = [k.fit(X) for k in model._kernels]
    f_hat = model._find_F(all_h)
    K_expanded = model._expand(cov)
    Y_vector = (model.Y.T).reshape((n * n_classes, 1))
    p_hat = np.exp(f_hat)
    p_hat /= np.sum(p_hat, axis=1).reshape((n, 1))
    p_hat_vector = p_hat.T.reshape((n * n_classes, 1))
    actual = K_expanded @ (Y_vector - p_hat_vector)
    actual = actual.reshape((n_classes, n)).T
    assert np.allclose(actual, f_hat)


def test_ML():
    model = gpmodel.GPMultiClassifier([k1, k2, k3])
    model.X = X
    model.Y = Y
    model._n_hypers = [k.fit(X) for k in model._kernels]
    f_hat = model._find_F(all_h)
    f_hat_vector = (f_hat.T).reshape((n * n_classes, 1))
    K_expanded = model._expand(cov)
    Y_vector = (model.Y.T).reshape((n * n_classes, 1))
    p_hat = np.exp(f_hat)
    p_hat /= np.sum(p_hat, axis=1).reshape((n, 1))
    p_hat_vector = p_hat.T.reshape((n * n_classes, 1))
    PI = model._stack(p_hat)
    W = np.diag(p_hat_vector[:, 0]) - PI @ PI.T
    print(p_hat)
    print(np.diag(p_hat_vector[:, 0]))
    ML = model._log_ML(all_h)
    first = 0.5 * f_hat_vector.T @ np.linalg.inv(K_expanded) @ f_hat_vector
    second = -Y_vector.T @ f_hat_vector
    third = np.sum(np.log(np.sum(np.exp(f_hat), axis=1)))
    W_root = linalg.sqrtm(W)
    Icn = np.eye(n * n_classes)
    fourth = 0.5 * np.log(np.linalg.det(Icn + W_root @ K_expanded @ W_root))
    actual = first + second + third + fourth
    # print(first, second, third, fourth)
    print(ML)
    print(actual)


def test_fit():
    model1 = gpmodel.GPMultiClassifier([k1, k2, k3])
    model1.X = X
    model1.Y = Y
    model1._n_hypers = [k.fit(X) for k in model1._kernels]
    print(model1._log_ML(all_h))
    m = gpmodel.GPMultiClassifier([k1, k2, k3], guesses=all_h)
    m.fit(X, Y)
    print(m.hypers)
    print(m.ML)
    p_hat = m._softmax(m._f_hat)
    plt.plot(X, f[:, 0], 'b')
    plt.plot(X, f[:, 1], 'g')
    plt.plot(X, f[:, 2], 'r')
    plt.plot(X, m._f_hat[:, 0], 'b.')
    plt.plot(X, m._f_hat[:, 1], 'g.')
    plt.plot(X, m._f_hat[:, 2], 'r.')
    for x, y in zip(X, Y_inds):
        if y == 0:
            plt.plot(x, 3, 'b.')
        elif y == 1:
            plt.plot(x, 3, 'g.')
        elif y == 2:
            plt.plot(x, 3, 'r.')
    plt.plot(X, p_hat[:, 0], 'b-.')
    plt.show()
    # print(model.Y)
    # print(model._softmax(model._f_hat))


def test_predict():
    m = gpmodel.GPMultiClassifier([k1, k2, k3], guesses=all_h)
    m.fit(X, Y)
    pi_star, mu, sigma = m.predict(X_test)
    print(m.hypers)
    print(np.sum(Y, axis=0))
    print(pi_star)
    print(mu)
    print(sigma[0])


if __name__ == "__main__":
    # test_soft_max()
    # test_aux_functions()
    # test_find_F()
    # test_ML()
    # test_fit()
    test_predict()
