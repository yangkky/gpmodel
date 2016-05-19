import sys
sys.path.append ('/Users/seinchin/Documents/Caltech/Arnold Lab/Programming tools/GPModel')
import gpkernel,gpmodel, gptools
import cPickle as pickle
import os
import argparse
import pytest
import datetime
import numpy as np
import matplotlib.pyplot as plt

def main():
    print 'Loading T50 data...'
    a_and_c = os.path.join('data/alignment_and_contacts.pkl')
    seq_file = os.path.join('data/X_seqs.pkl')
    Y_file = os.path.join('data/Ys.pkl')
    with open (a_and_c) as f:
        cons, ss = pickle.load(f)
    with open (seq_file) as f:
        X_seqs = pickle.load(f)
    with open (Y_file) as f:
        Ys = pickle.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kernel',required=True)
    parser.add_argument('-n', '--name',required=False)
    #parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-g', '--guess', nargs='*', required=False)
    parser.add_argument('-o', '--objective', required=False, default='log_ML')


    args = parser.parse_args()


    print 'Creating kernel...'
    if args.kernel == 'hamming':
        kern = gpkernel.HammingKernel()
    elif args.kernel == 'structure':
        kern = gpkernel.StructureKernel(cons)
    elif args.kernel == 'SEStructure':
        kern = gpkernel.StructureSEKernel(cons)
    elif args.kernel == 'SEHamming':
        kern = gpkernel.HammingSEKernel()
    else:
        sys.exit ('Invalid kernel type')

    print 'Building the model...'
    model = gpmodel.GPModel(X_seqs, Ys, kern, guesses=args.guess,
                           objective=args.objective)
    print model.hypers
    print '-log_ML = %f' %model.ML
    try:
        print '-log_LOO_P = %f' %model.log_p
    except:
        pass
    names = model.hypers._fields
    hypers = {n:h for n,h in zip(names, model.hypers)}

    print 'Pickling the model...'
    dt = datetime.date.today()
    name = str(dt) + '_' + args.kernel + '_model.pkl'
    with open(name,'w') as f:
        pickle.dump(hypers, f)

    print 'Plotting...'
    LOOs = model.LOO_res(model.hypers)
    predicted = model.unnormalize(LOOs['mu'])
    actual = model.Y
    gptools.plot_predictions(actual, predicted)
    plt.title ('LOO predictions for T50 with ' + args.kernel +  ' kernel')
    print np.corrcoef(model.normed_Y, LOOs['mu'])[0,1]
    plt.savefig(str(dt) + '_' + args.kernel+'_LOO.pdf')
    plt.show()


if __name__ == "__main__":
    main()

