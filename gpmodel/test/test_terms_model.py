import sys
sys.path.append ('/Users/seinchin/Documents/Caltech/Arnold Lab/Programming tools/GPModel')

import pandas as pd
import numpy as np
import pickle

import gpmodel
import gpkernel

contact_terms = [((0, 'A'), (1, 'A')),
                 ((0, 'A'), (1, 'C')),
                 ((0, 'B'), (1, 'A')),
                 ((0, 'B'), (1, 'C')),
                 ((0, 'C'), (1, 'A')),
                 ((0, 'C'), (1, 'C')),
                 ((0, 'A'), (2, 'A')),
                 ((0, 'A'), (2, 'B')),
                 ((0, 'A'), (2, 'D')),
                 ((0, 'B'), (2, 'A')),
                 ((0, 'B'), (2, 'B')),
                 ((0, 'B'), (2, 'D')),
                 ((0, 'C'), (2, 'A')),
                 ((0, 'C'), (2, 'B')),
                 ((0, 'C'), (2, 'D')),
                 ((1, 'A'), (2, 'A')),
                 ((1, 'A'), (2, 'B')),
                 ((1, 'A'), (2, 'D')),
                 ((1, 'C'), (2, 'A')),
                 ((1, 'C'), (2, 'B')),
                 ((1, 'C'), (2, 'D'))]

sequence_terms = [(0, 'A'), (0, 'B'), (0, 'C'), (1, 'A'),
                  (1, 'C'), (2, 'A'), (2, 'B'), (2, 'D')]

all_X = np.array([[1, 0, 0, 1, 0, 0],
                  [0, 1, 0, 1, 0, 1],
                  [0, 1, 0, 0, 1, 0]])
complete_X = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
terms = sequence_terms + contact_terms
seqs = ['AAB', 'BAD', 'BCA']
seqs_df = np.array([[s for s in seq] for seq in seqs])
seqs_df = pd.DataFrame(seqs_df, index=['A', 'B', 'C'])
model = gpmodel.TermedLassoGPModel(gpkernel.SEKernel(), gamma=-5)
Y = pd.Series([1.0, 0.1, 0.2], index=seqs_df.index)

def test_make_X():
    X = model._make_X(seqs_df, terms)
    X = X.values
    assert np.array_equal(complete_X, X)

def test_regularize():
    X, mask, reg_terms = model._regularize(seqs_df, terms, y=Y, gamma=-4)
    assert np.isclose(X.values,
                      complete_X.transpose()[mask].transpose()).all()
    assert reg_terms == [t for t, m in zip(terms, mask) if m]
    test_mask = np.random.choice([True, False], size=(len(mask)))
    X, reg_mask, reg_terms = model._regularize(seqs_df, terms, mask=test_mask)
    assert reg_terms == [t for t, m in zip(terms, test_mask) if m]
    assert (reg_mask == test_mask).all()

def test_log_ML_from_gamma():
    g = -2.0
    model2 = gpmodel.LassoGPModel(gpkernel.SEKernel())
    X = model._make_X(seqs_df, terms)
    true_ML = model2._log_ML_from_gamma(g, X, Y)
    test_ML = model._log_ML_from_gamma(g, seqs_df, Y, terms)
    assert np.isclose(true_ML, test_ML)


def test_fit():
    np.random.seed(1)
    model = gpmodel.TermedLassoGPModel(gpkernel.LinearKernel(), gamma=-5)
    model.fit(seqs_df, Y, terms)
    assert(np.isclose(model.ML, 3.3895503170039034))
    assert len(model.X_seqs.columns == 2)



def test_predict():
    model = gpmodel.TermedLassoGPModel(gpkernel.LinearKernel(), gamma=-5)
    model.fit(seqs_df, Y, terms)
    preds = model.predict(seqs_df)
    assert np.isclose(preds[0][0], 0.85724109890691791)
    assert np.isclose(preds[0][1], 0.04072834275708116)


if __name__=="__main__":
    test_make_X()
    test_regularize()
    test_log_ML_from_gamma()
    test_fit()
    test_predict()
