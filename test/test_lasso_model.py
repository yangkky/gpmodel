import sys
sys.path.append('/Users/kevinyang/Documents/Projects/GPModel')
sys.path.append ('/Users/seinchin/Documents/Caltech/Arnold Lab/Programming tools/GPModel')
import pytest
import os

import pandas as pd
import numpy as np
import math
import scipy
from scipy import stats
from sklearn import metrics, linear_model

import gpkernel, gpmodel, gpmean, chimera_tools

SE_kern = gpkernel.SEKernel()
# func = gpmean.StructureSequenceMean(space, contacts, linear_model.Lasso,
#
np.random.seed(0)
X = np.random.random(size=30000)
X = X.reshape((30,1000))
w1 = np.random.random(size=(1, 450)) * 10.0
w2 = np.random.random(size=(1, 550)) * 0.01
w = np.concatenate((w1, w2), axis=1)
Y = np.dot(w, X.T)[0]
X_df = pd.DataFrame(X)
Y += np.random.normal(size=len(X), scale=1.0)
Y = pd.Series(Y, index=X_df.index)
#variances = pd.Series([0.11, 0.18, 0.13, 0.,14], index=X_df.index)


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
    assert np.isclose(neg_ML, -model2.ML)


def test_fit():
    model = gpmodel.LassoGPModel(gpkernel.LinearKernel(), gamma=0.1)
    model.fit(X_df, Y)
    assert len(model.X_seqs.columns) == 29
    assert np.isclose(model.ML, 30.585317139044697)


if __name__=="__main__":
    #test_creation()
    test_init()
    test_regularize()
    test_log_ML_from_lambda()
    test_fit()
    #test_classification()
    #test_score
