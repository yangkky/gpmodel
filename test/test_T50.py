import sys
sys.path.append ('/Users/seinchin/Documents/Caltech/Arnold Lab/Programming tools/GPModel')
import gpkernel,gpmodel, gptools
import pandas as pd
import cPickle as pickle
import os
import pytest
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def main():
    dir = os.path.dirname(__file__)
#     print 'Loading T50 data...'
#     a_and_c = os.path.join(dir, 'alignment_and_contacts.pkl')
#     seq_file = os.path.join(dir, 'X_seqs.pkl')
#     Y_file = os.path.join(dir, 'Ys.pkl')
#     with open (a_and_c) as f:
#         cons, ss = pickle.load(f)
#     with open (seq_file) as f:
#         X_seqs = pickle.load(f)
#     with open (Y_file) as f:
#         Ys = pickle.load(f)

#     print 'Making the structure kernel...'
#     kern = gpkernel.StructureKernel(contacts=cons,
#                                     sample_space=ss)
#     print 'Building the model...'
#     model = gpmodel.GPModel(X_seqs, Ys, kern, guesses=(2.4899,0.107),train=True)


#     print 'Pickling the model...'
#     with open('2015-11-12_model.pkl','w') as f:
#         pickle.dump(model, f)

    with open ('2015-11-12_model.pkl') as f:
        model = pickle.load(f)

    LOO1 = model.LOO_res((model.var_n, model.var_p))
    gptools.plot_predictions(model.normed_Y, LOO1['mu'],label='T50',title='ML')
    plt.savefig('2015-11-12_ML.pdf')
    plt.close('all')
    res = minimize(model.LOO_log_p,
                   (model.var_p, model.var_n),
                   bounds=[(1e-5, None), (1e-5, None)],
                   method='L-BFGS-B')
    LOO1 = model.LOO_res(res['x'])
    gptools.plot_predictions(model.normed_Y, LOO1['mu'],label='T50',title='log_p')
    plt.savefig('2015-11-12_LOO_log_p.pdf')
    plt.close('all')
    res = minimize(model.LOO_MSE,
                   (model.var_p, model.var_n),
                   bounds=[(1e-5, None), (1e-5, None)],
                   method='L-BFGS-B')
    LOO1 = model.LOO_res(res['x'])
    gptools.plot_predictions(model.normed_Y, LOO1['mu'],label='T50',title='MSE')
    plt.savefig('2015-11-12_LOO_MSE.pdf')


if __name__ == "__main__":
    main()

