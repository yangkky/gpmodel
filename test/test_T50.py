import sys
sys.path.append ('/Users/seinchin/Documents/Caltech/Arnold Lab/Programming tools/GPModel')
import gpkernel,gpmodel
import pandas as pd
import cPickle as pickle
import os
import pytest

def main():
    dir = os.path.dirname(__file__)
    print 'Loading T50 data...'
    a_and_c = os.path.join(dir, 'alignment_and_contacts.pkl')
    seq_file = os.path.join(dir, 'X_seqs.pkl')
    Y_file = os.path.join(dir, 'Ys.pkl')
    with open (a_and_c) as f:
        cons, ss = pickle.load(f)
    with open (seq_file) as f:
        X_seqs = pickle.load(f)
    with open (Y_file) as f:
        Ys = pickle.load(f)

    print 'Making the structure kernel...'
    kern = gpkernel.StructureKernel(contacts=cons,
                                    sample_space=ss)
    print 'Building the model...'
    model = gpmodel.GPModel(X_seqs, Ys, kern, guesses=(2.4899,0.107),train=False)

    print model.var_p
    print model.var_n
    print model.ML

#     print 'Pickling the model...'
#     with open('model.pkl','w') as f:
#         pickle.dump(model, f)

if __name__ == "__main__":
    main()

