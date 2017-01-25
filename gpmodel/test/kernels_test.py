import sys
import pytest

import pandas as pd
import numpy as np

from gpmodel import gpkernel

d = 10
xa = np.random.random(size=(1, d))
xb = np.random.random(size=(1, d))
xc = np.random.random(size=(1, d))
X = np.concatenate((xa, xb, xc), axis=0)
actual_ds = np.array([[np.linalg.norm(i-j)**2 for i in [xa, xb, xc]] for
                      j in [xa, xb, xc]])
A = np.random.random(size=(d, d))
L = A @ A.T  # PD Matrix
actual_ARD_ds = np.array([[((i - j) @ L @ (i - j).T).item()
                           for i in [xa, xb, xc]]
                          for j in [xa, xb, xc]])


def test_radial_kernel():
    kernel = gpkernel.BaseRadialKernel()
    assert np.allclose(kernel._distance(X, X), actual_ds)
    assert np.allclose(kernel._distance(X[0:2], X[2:]),
                       actual_ds[0:2, 2:])
    kernel.fit(X)
    assert np.allclose(kernel._saved, actual_ds)


def test_ARD_radial_kernel():
    kernel = gpkernel.BaseRadialARDKernel()
    assert np.allclose(kernel._distance(X, X, np.eye(d)), actual_ds)
    assert np.allclose(kernel._distance(X, X, L), actual_ARD_ds)
    kernel.fit(X)
    assert np.allclose(kernel._saved, X)
    params = np.random.random(size=(d, ))
    actual_diag_ds = np.array([[((i - j) @ np.diag(params) @ (i - j).T).item()
                                for i in [xa, xb, xc]]
                               for j in [xa, xb, xc]])
    assert np.allclose(kernel._distance(X, X, params), actual_diag_ds)


def test_se_kernel():
    # Test __init__
    kern = gpkernel.SEKernel()
    assert kern.hypers == ['sigma_f', 'ell']
    sigma_f = 0.3
    ell = 0.2
    params = (sigma_f, ell)
    actual = sigma_f**2 * np.exp(-0.5 * actual_ds / ell**2)
    assert np.allclose(kern.cov(X, X, hypers=params), actual)
    assert np.allclose(kern.cov(xa, xb, params), actual[0, 1])
    assert np.allclose(kern.cov(X[0:2], X[2:], params),
                       actual[0:2, 2:])
    assert np.allclose(kern.cov(X, X), np.exp(-0.5 * actual_ds))
    kern.fit(X)
    assert np.allclose(kern.cov(hypers=params), actual)


def test_ARD_se_kernel():
    kern = gpkernel.ARDSEKernel()
    assert kern.hypers == ['sigma_f', 'ell']
    r = actual_ARD_ds
    sigma_f = np.random.random()
    params = (sigma_f, L)
    actual = sigma_f**2 * np.exp(-0.5 * r)
    assert np.allclose(kern.cov(X, X, hypers=params), actual)
    assert np.allclose(kern.cov(xa, xb, params), actual[0, 1])
    assert np.allclose(kern.cov(X[0:2], X[2:], params),
                       actual[0:2, 2:])
    r = actual_ds
    assert np.allclose(kern.cov(X, X),
                       np.exp(-0.5 * r))
    kern.fit(X)
    assert np.allclose(kern.cov(hypers=params), actual)


def test_polynomial_kernel():
    kern1 = gpkernel.PolynomialKernel(1)
    kern3 = gpkernel.PolynomialKernel(3)
    assert kern1.hypers == ['sigma_0', 'sigma_p']
    assert kern1._deg == 1
    assert kern3._deg == 3
    pytest.raises(ValueError, 'gpkernel.PolynomialKernel(0)')
    pytest.raises(TypeError, 'gpkernel.PolynomialKernel(1.2)')

    params = [0.9, 0.1]
    sigma_0, sigma_p = params
    inside = sigma_0 ** 2 + sigma_p ** 2 * X @ X.T
    assert np.allclose(kern1.cov(X, X, params), inside)
    assert np.allclose(kern3.cov(X, X, params), inside ** 3)
    assert np.allclose(kern1.cov(xa, xb, params), inside[0, 1])
    assert np.allclose(kern3.cov(X[0:], X[1:], params), inside[0:, 1:] ** 3)

    assert np.allclose(kern1.cov(X, X), 1 + X @ X.T)
    kern3.fit(X)
    assert np.allclose(kern3.cov(hypers=params), inside ** 3)


def test_matern_kernel():
    kern1 = gpkernel.MaternKernel('3/2')
    kern2 = gpkernel.MaternKernel('5/2')
    assert kern1.hypers == ['ell']
    assert kern1.nu == '3/2'
    assert kern2.nu == '5/2'
    r = np.sqrt((actual_ds))
    params = np.random.random(size=(1, ))
    ell = params[0]
    actual1 = (1 + np.sqrt(3.0) / ell * r) * np.exp(-np.sqrt(3.0) * r / ell)
    actual2 = (1 + np.sqrt(5.0) / ell * r + 5.0*r**2/3/ell**2) *\
        np.exp(-np.sqrt(5.0) * r / ell)
    assert np.allclose(kern1.cov(X, X, hypers=params), actual1)
    assert np.allclose(kern1.cov(xa, xb, params), actual1[0, 1])
    assert np.allclose(kern1.cov(X[0:2], X[2:], params),
                       actual1[0:2, 2:])
    assert np.allclose(kern1.cov(X, X),
                       (1 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r))
    kern2.fit(X)
    assert np.allclose(kern2.cov(hypers=params), actual2)


def test_ARD_matern_kernel():
    kern1 = gpkernel.ARDMaternKernel('3/2')
    kern2 = gpkernel.ARDMaternKernel('5/2')
    assert kern1.hypers == ['ell']
    assert kern1.nu == '3/2'
    assert kern2.nu == '5/2'
    r = np.sqrt((actual_ARD_ds))
    params = L
    actual1 = (1 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
    actual2 = (1 + np.sqrt(5.0) * r + 5.0*r**2/3) *\
        np.exp(-np.sqrt(5.0) * r)
    assert np.allclose(kern1.cov(X, X, hypers=params), actual1)
    assert np.allclose(kern1.cov(xa, xb, params), actual1[0, 1])
    assert np.allclose(kern1.cov(X[0:2], X[2:], params),
                       actual1[0:2, 2:])
    r = np.sqrt(actual_ds)
    assert np.allclose(kern1.cov(X, X),
                       (1 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r))
    kern2.fit(X)
    assert np.allclose(kern2.cov(hypers=params), actual2)


def test_linear_kernel():
    kern = gpkernel.LinearKernel()
    assert kern.hypers == ['var_p']
    params = np.random.random(size=(1, ))
    vp = params[0]
    actual = X @ X.T * vp
    assert np.allclose(kern.cov(X, X, params), actual)
    assert np.allclose(kern.cov(xa, xb, params), actual[0, 1])
    assert np.allclose(kern.cov(X[0:2], X[2:], params),
                       actual[0:2, 2:])
    assert np.allclose(kern.cov(X, X), X @ X.T)
    kern.fit(X)
    assert np.allclose(kern.cov(hypers=params), actual)


def test_sum_kernel():
    kern52 = gpkernel.MaternKernel('5/2')
    kernSE = gpkernel.SEKernel()
    kern32 = gpkernel.MaternKernel('3/2')
    kernels = [kern52, kernSE, kern32]
    params = np.random.random((4, ))
    actual = kern52.cov(X, X, params[[0]]) + kernSE.cov(X, X, params[1:3]) + \
        kern32.cov(X, X, params[[3]])
    kernel = gpkernel.SumKernel(kernels)
    assert kernel._kernels == [kern52, kernSE, kern32]
    assert kernel.hypers == ['ell0', 'sigma_f0', 'ell1', 'ell2']
    assert np.allclose(kernel.cov(X, X, params), actual)
    assert np.allclose(kernel.cov(xa, xb, params), actual[0, 1])
    assert np.allclose(kernel.cov(X[0:2], X[2:], params),
                       actual[0:2, 2:])
    actual2 = kern52.cov(X, X) + kernSE.cov(X, X) + kern32.cov(X, X)
    assert np.allclose(kernel.cov(X, X), actual2)
    kernel.fit(X)
    assert np.allclose(kernel.cov(hypers=params), actual)

if __name__=="__main__":
    test_radial_kernel()
    test_ARD_radial_kernel()
    test_matern_kernel()
    test_ARD_matern_kernel()
    test_linear_kernel()
    test_polynomial_kernel()
    test_se_kernel()
    test_ARD_se_kernel()
    test_sum_kernel()
