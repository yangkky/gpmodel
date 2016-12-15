import sys
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

SE_kern = gpkernel.SEKernel()
np.random.seed(0)
n = 30
n_dims = 1000

X = np.random.random(size=n*n_dims)
X = X.reshape((n, n_dims))
w1 = np.random.random(size=(1, 450)) * 10.0
w2 = np.random.random(size=(1, n_dims-450)) * 0.01
w = np.concatenate((w1, w2), axis=1)
Y = np.dot(w, X.T)[0]
X_df = pd.DataFrame(X, index=[str(i) for i in range(len(X))])
Y += np.random.normal(size=len(X), scale=1.0)
Y = pd.Series(Y, index=X_df.index)


def test_init():
    model = gpmodel.LassoGPModel(gpkernel.LinearKernel(), gamma=1)
    assert model._gamma_0 == 1
    model = gpmodel.LassoGPModel(gpkernel.LinearKernel())
    assert model._gamma_0 == 0


def test_regularize():
    model = gpmodel.LassoGPModel(gpkernel.LinearKernel())
    pytest.raises(ValueError, model._regularize, X_df, gamma=0)
    X, mask = model._regularize(X_df, y=Y, gamma=0)
    assert np.isclose(X.values,
                      X_df.transpose()[mask].transpose().values).all()
    clf = linear_model.Lasso(alpha=np.exp(0))
    clf.fit(X_df, Y)
    not_zero = ~np.isclose(clf.coef_, 0)
    assert np.array_equal(not_zero, mask)
    new_mask = np.random.choice([True, False], size=len(X_df.columns))
    X, mask = model._regularize(X_df, mask=new_mask)
    assert np.array_equal(mask, new_mask)
    assert np.isclose(X.values,
                      X_df.transpose()[mask].transpose().values).all()


def test_log_ML_from_lambda():
    g = 0.0
    model = gpmodel.LassoGPModel(gpkernel.LinearKernel())
    model2 = gpmodel.GPModel(gpkernel.LinearKernel())
    X, mask = model._regularize(X_df, gamma=g, y=Y)
    model2.fit(X, Y)
    neg_ML = model._log_ML_from_gamma(g, X, Y)
    assert np.isclose(neg_ML, model2.ML)


def test_fit():
    model = gpmodel.LassoGPModel(gpkernel.LinearKernel(), gamma=0.1)
    np.random.seed(1)
    model.fit(X_df, Y)
    assert len(model.X_seqs.columns) == 19
    assert np.isclose(model.ML, 26.189324722766216)
    # need a test for kernel remembering correct X


def test_predict():
    model = gpmodel.LassoGPModel(gpkernel.LinearKernel(), gamma=-2)
    model.fit(X_df, Y)
    np.random.seed(1)
    X_test = np.random.random(size=(1, n_dims))
    X_test = pd.DataFrame(X_test, index=['A'])
    X_masked = X_test.transpose()[model._mask].transpose()
    Y_test = model.predict(X_test)
    check_model = gpmodel.GPModel(gpkernel.LinearKernel())
    check_model.fit(model.X_seqs, Y)
    Y_check = check_model.predict(X_masked)
    assert np.isclose(Y_test, Y_check).all()


def test_dump_and_load():
    model = gpmodel.LassoGPModel(gpkernel.LinearKernel(), gamma=-2)
    model.fit(X_df, Y)
    model.dump('test.pkl')
    model_2 = gpmodel.GPModel.load('test.pkl')
    assert model.gamma == model_2.gamma


if __name__ == "__main__":
    test_init()
    test_regularize()
    test_log_ML_from_lambda()
    test_fit()
    test_predict()
