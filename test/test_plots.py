import sys
sys.path.append ('/Users/seinchin/Documents/Caltech/Arnold Lab/Programming tools/GPModel')
import gpkernel, gpmodel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--kernel',required=True)
parser.add_argument('-b', '--binary', action='store_true')
parser.add_argument('-g', '--hypers', nargs='*', required=False, type=float)

args = parser.parse_args()
if args.kernel == '3/2':
    kern = gpkernel.MaternKernel(nu='3/2')
elif args.kernel == '5/2':
    kern = gpkernel.MaternKernel(nu='5/2')
elif args.kernel == 'se':
    kern = gpkernel.SEKernel()
else:
    raise ValueError('Invalid kernel.')

if args.binary:
    X = pd.DataFrame(np.array([[-5.0, -3.0], [1.0, -9.5], [6.5, 2.5],
                               [0.5, 0.6], [2.0, 0.0]]))
    X.columns = ['x1','x2']
    X.index = ['A', 'B', 'C', 'D', 'E']
    Y = pd.Series([-1, 1, 1, -1, -1], index=X.index)


else:
    X = pd.DataFrame(np.array([[-5.0, -3.0, 1.0, -9.5, 6.5, 2.5, 0.5, 0.6, 2.0]]).T)
    X.index = ['A', 'B', 'C', 'D', 'E','F','G','H', 'I']
    Y = pd.Series ([-0.5, -1.0, -1.3, -1.3, -1.3, 0.1, -1.1, -1.15, 0], index=X.index)


model = gpmodel.GPModel(kern)

if args.hypers is None:
    model.fit(X, Y)
    print model.hypers
    print 'log(ML) = ', -model.ML

else:
    model.set_params(X=X, Y=Y, hypers=args.hypers)

X_new = np.arange(-10, 10, 0.1)

if args.binary:
    xx, yy = np.meshgrid(X_new, X_new, sparse=False)
    X_df = []
    for i,x in enumerate(X_new):
        for j, y in enumerate(X_new):
            X_df.append([x,y])
    X_df = pd.DataFrame(np.array(X_df))
    preds = model.predicts(X_df)
    preds = [p[0] for p in preds]
    preds = np.array(preds)
    preds = np.reshape(preds, (len(X_new), len(X_new)))
    CS = plt.contour(xx, yy, preds)
    plt.clabel(CS, inline=1, fontsize=10)
    pos = Y==1
    neg = Y==-1
    plt.plot(X[pos]['x1'], X[pos]['x2'], 'go')
    plt.plot(X[neg]['x1'], X[neg]['x2'], 'ro')



else:
    X_new = np.arange(-10, 10, 0.1)
    X_new = pd.DataFrame(X_new.T)
    preds = model.predicts(X_new)
    preds = pd.DataFrame(preds, columns=['mean', 'var'])
    preds['x'] = X_new
    plt.plot(preds['x'], preds['mean'])
    plt.plot(preds['x'], preds['mean']+preds['var'], 'k--')
    plt.plot(preds['x'], preds['mean']-preds['var'], 'k--')
    plt.plot(X, Y, 'o')
plt.show()
