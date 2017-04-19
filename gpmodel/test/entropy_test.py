import sys

import pandas as pd
import numpy as np

from gpmodel import gpkernel
from gpmodel import gpentropy


def entropy(new_X, old_X, kernel, hypers):
    k_star = kernel.cov(new_X, old_X, hypers=hypers)
    kss = kernel.cov(new_X, new_X, hypers=hypers)
    K = kernel.cov(X1=old_X, X2=old_X, hypers=hypers)
    Ky = K + np.eye(len(old_X)) * vn
    kss -= k_star @ np.linalg.inv(Ky) @ k_star.T

d = 20
xa = np.random.random(size=(1, d))
xb = np.random.random(size=(1, d))
xc = np.random.random(size=(1, d))
X = np.concatenate((xa, xb, xc), axis=0)

kernel = gpkernel.MaternKernel('5/2')
out_kernel = gpkernel.MaternKernel('5/2')
ell = 0.5
vn = 0.01

# Test the constructor
ent = gpentropy.GPEntropy(kernel, [ell], var_n=vn, observations=X)
K = out_kernel.cov(X1=X, X2=X, hypers=(ell, ))
Ky = K + np.eye(len(X)) * vn
L = np.linalg.cholesky(Ky)
assert np.allclose(Ky, ent._Ky)
assert np.allclose(L, ent._L)
assert ent.hypers[0] == ell
assert ent.var_n == vn

# add observations and check Ky again
X_new = np.random.random(size=(2, d))
ent.observe(X_new)
all_X = np.concatenate((X, X_new))
assert np.allclose(all_X, ent.observed)
K = out_kernel.cov(X1=all_X, X2=all_X, hypers=(ell, ))
Ky = K + np.eye(len(all_X)) * vn
L = np.linalg.cholesky(Ky)
assert np.allclose(Ky, ent._Ky)
assert np.allclose(L, ent._L)

# k_star
X_test = np.random.random(size=(3, d))
k_star = out_kernel.cov(X_test, all_X, hypers=(ell, ))

# posterior
kss = out_kernel.cov(X_test, X_test, hypers=(ell, ))
kss -= k_star @ np.linalg.inv(Ky) @ k_star.T
kss += vn * np.eye(len(kss))
assert np.allclose(ent._posterior_covariance(X_test), kss)

# entropy
entropy = 0.5 * np.log(np.linalg.det(kss))
entropy += 3 / 2 * np.log(2 * np.pi * np.exp(1.0))
assert np.isclose(ent.entropy(X_test), entropy)


# probabilistic entropy
P = np.ones(3)
assert np.isclose(ent.expected_entropy(X_test, P), entropy)

# lazy-greedy
# X_test = np.random.random(size=(25, d))
# P = np.random.random(size=25)
# print(ent.maximize_entropy(X_test, 20))
# print(ent.maximize_expected_entropy(X_test, P, 12))
