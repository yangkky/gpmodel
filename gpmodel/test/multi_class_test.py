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

n = 5
d = 10
n_classes = 3
X = np.random.random(size=(n, d))
xa = X[[0]]
xb = X[[1]]
Xc = X[[2]]
X_test = np.random.random(size=(5, d))
k1 = gpkernel.SEKernel()
k2 = gpkernel.MaternKernel('5/2')
k3 = gpkernel.PolynomialKernel(2)
h1 = np.array([1.8, 1.5])
h2 = np.array([1.4])
h3 = np.array([0.5, 1.1])
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





if __name__ == "__main__":
    # test_soft_max()
    # test_aux_functions()
    test_find_F()
