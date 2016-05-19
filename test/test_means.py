import sys
sys.path.append ('/Users/seinchin/Documents/Caltech/Arnold Lab/Programming tools/GPModel')
import gpmean
import pandas as pd
import numpy as np
import chimera_tools
from sklearn import linear_model


seqs = pd.DataFrame([['R','Y','M','A'],['C','T','I','A'], ['R','T','M','B']],
                    index=['A','B','C'], columns=[0,1,2,3])
space = [('R', 'T', 'C'), ('Y', 'T', 'B'), ('M', 'H', 'I'), ('A', 'A', 'B')]
contacts = [(0,1),(0,3),(2,3)]
Y = pd.Series([-1,1,0.5],index=seqs.index)
alpha = 0.1


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
                                    space, contacts)
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X, Y)
    assert this._terms == terms
    assert np.array_equal(this._clf.coef_, clf.coef_)

if __name__=="__main__":
    test_constructor()
    test_fit()
