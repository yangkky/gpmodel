import sys
import string
sys.path.append('/Users/kevinyang/Documents/Projects/GPModel')
sys.path.append ('/Users/seinchin/Documents/Caltech/Arnold Lab/Programming tools/GPModel')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

import gpkernel, gpmodel

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--kernel',required=True)
parser.add_argument('-type', '--type')
parser.add_argument('-g', '--hypers', nargs='*', required=False, type=float)

args = parser.parse_args()
if args.kernel == '3/2':
    kern = gpkernel.MaternKernel(nu='3/2')
elif args.kernel == '5/2':
    kern = gpkernel.MaternKernel(nu='5/2')
elif args.kernel == 'se':
    kern = gpkernel.SEKernel()
elif args.kernel == 'linear':
    kern = gpkernel.LinearKernel()
else:
    raise ValueError('Invalid kernel.')

if args.type == 'b2':
    np.random.seed(0)
    cov =  np.array([[1.2, 0.1],[0.1, 1.2]])
    x1 = np.random.multivariate_normal(np.array([-3.0, -3.0]),
                                      cov, size=20)
    x2 = np.random.multivariate_normal(np.array([1.0, 1.0]),
                                       cov, size=20)
    x3 = np.random.multivariate_normal(np.array([0.0, 3.0]),
                                       cov, size=20)
    x4 = np.random.multivariate_normal(np.array([4.0, -4.0]),
                                       cov, size=20)
    X = np.concatenate((x1, x2, x3, x4))
    X = pd.DataFrame(X)
    X.index = [chr(i) for i in range(len(X))]
    X.columns = ['x1','x2']
    y1 = [1 for _ in range(40)]
    y2 = [-1 for _ in range(40)]
    Y = y1 + y2
    Y = pd.Series(Y, index=X.index)

elif args.type == 'b1':
    np.random.seed(0)
    x1 = np.random.normal(-6, 0.8, 20)
    x2 = np.random.normal(0, 0.8, 30)
    x3 = np.random.normal(2, 0.8, 10)
    X = np.concatenate((x1, x2, x3))
    X = pd.DataFrame(X)
    X.index = [chr(i) for i in range(len(X))]
    y1 = [1 for _ in range(20)]
    y2 = [-1 for _ in range(30)]
    y3 = [1 for _ in range(10)]
    Y = y1 + y2 + y3
    Y = pd.Series(Y, index=X.index)


else:
    X = pd.DataFrame(np.array([[-5.0, -3.0, 1.0, -9.5, 6.5, 2.5, 0.5, 0.6, 2.0]]).T)
    X.index = ['A', 'B', 'C', 'D', 'E','F','G','H', 'I']
    Y = pd.Series ([-0.5, -1.0, -1.3, -1.3, -1.3, 0.1, -1.1, -1.15, 0], index=X.index)


model = gpmodel.GPModel(kern)

if args.hypers is None:
    model.fit(X, Y)
    print(model.hypers)
    print('log(ML) = ', -model.ML)

else:
    model.set_params(X=X, Y=Y, hypers=args.hypers)


if args.type == 'b2':
    X_new = np.arange(-7, 7, 0.1)
    xx, yy = np.meshgrid(X_new, X_new, sparse=False)
    X_df = []
    for i,x in enumerate(X_new):
        for j, y in enumerate(X_new):
            X_df.append([x,y])
    X_df = pd.DataFrame(np.array(X_df))
    preds = model.predicts(X_df)
    preds = [p[0] for p in preds]
    preds = np.array(preds)
    preds = np.reshape(preds, (len(X_new), len(X_new))).T
    CS = plt.contour(xx, yy, preds)
    plt.clabel(CS, inline=1, fontsize=10)
    pos = Y==1
    neg = Y==-1
    plt.plot(X[pos]['x1'], X[pos]['x2'], 'go')
    plt.plot(X[neg]['x1'], X[neg]['x2'], 'ro')

elif args.type == 'b1':
    X_new = np.arange(5, -8, -0.1)
    preds = model.predicts(pd.DataFrame(X_new.T))
    pis = [p[0] for p in preds]
    f_bars = np.array([p[1] for p in preds])
    print(f_bars)
    var = np.array([p[2] for p in preds])
    plt.plot(X_new, pis, 'g.', alpha=0.7)
    plt.plot (X, (Y+1.0)/2.0, 'ko', alpha=0.7)

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
plt.title(args.kernel)
plt.margins(0.02)
plt.show()
