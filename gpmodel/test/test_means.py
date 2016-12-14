import pandas as pd
import numpy as np
from sklearn import linear_model

from gpmodel import gpmean
from gpmodel import chimera_tools


seqs = pd.DataFrame([['R','Y','M','A'],['C','T','I','A'], ['R','T','M','B']],
                    index=['A','B','C'], columns=[0,1,2,3])
space = [('R', 'T', 'C'), ('Y', 'T', 'B'), ('M', 'H', 'I'), ('A', 'A', 'B')]
contacts = [(0,1),(0,3),(2,3)]
Y = pd.Series([-1,1,0.5],index=seqs.index)
alpha = 0.1
new_seqs = pd.DataFrame([['C','Y','M','A'],['T','T','I','A']],
                    index=['D', 'E'], columns=[0,1,2,3])


def test_constructor():
    this = gpmean.StructureSequenceMean(space, contacts, linear_model.Lasso,
                                        alpha=alpha)
    assert this._sample_space == space
    assert this._contacts == contacts
    assert type(this._clf) == linear_model.coordinate_descent.Lasso
    assert this._clf.alpha == alpha

def test_fit():
    this = gpmean.StructureSequenceMean(space, contacts, linear_model.Lasso,
                                        alpha=alpha)
    this.fit(seqs, Y)
    X, terms = chimera_tools.make_X([''.join(row) for _, row in seqs.iterrows()],
                                    space, contacts, collapse=False)
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X, Y)
    assert this._terms == terms
    assert np.array_equal(this._clf.coef_, clf.coef_)
    assert np.array_equal(this.means, clf.predict(X))

def test_X():
    this = gpmean.StructureSequenceMean(space, contacts, linear_model.Lasso,
                                        alpha=alpha)
    X, terms = this._make_X(seqs)
    actual_X, actual_terms = chimera_tools.make_X([''.join(row) for _, row
                                                   in seqs.iterrows()],
                                                  space, contacts,
                                                  collapse=False)
    assert np.array_equal(X, actual_X)
    assert terms == actual_terms
    this.fit(seqs, Y)
    X, terms = this._make_X(seqs)
    assert np.array_equal(X, actual_X)
    X, terms = this._make_X(new_seqs)
    assert terms == this._terms
    actual_X, actual_terms = chimera_tools.make_X([''.join(row) for _, row
                                                   in new_seqs.iterrows()],
                                                  space, contacts,
                                                  collapse=False)
    assert np.array_equal(X, actual_X)

def test_mean():
    this = gpmean.GPMean()
    assert np.array_equal(np.zeros(3), this.mean(seqs))
    this = gpmean.StructureSequenceMean(space, contacts, linear_model.Lasso,
                                        alpha=alpha)
    this.fit(seqs, Y)
    X, terms = chimera_tools.make_X([''.join(row) for _, row
                                     in seqs.iterrows()],
                                    space, contacts, collapse=False)
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X, Y)
    new_X, _ = chimera_tools.make_X([''.join(row) for
                                     _, row in new_seqs.iterrows()],
                                    space, contacts, terms=terms,
                                    collapse=False)
    preds = clf.predict(new_X)
    assert np.array_equal(this.mean(new_seqs), preds)


if __name__=="__main__":
    test_constructor()
    test_fit()
    test_X()
    test_mean()
