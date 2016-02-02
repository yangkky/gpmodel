import sys
sys.path.append ('/Users/seinchin/Documents/Caltech/Arnold Lab/Programming tools/GPModel')
import gpkernel, gpmodel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.DataFrame(np.array([[-5.0, -3.0, 1.0, -9.5, 6.5, 2.5, 0.5, 0.6, 2.0]]).T)
X.index = ['A', 'B', 'C', 'D', 'E','F','G','H', 'I']
Y = pd.Series ([-0.5, -1.0, -1.3, -1.3, -1.3, 0.1, -1.1, -1.15, 0], index=X.index)

kern = gpkernel.SEKernel()
model = gpmodel.GPModel(kern)
model.fit(X, Y)
X_new = np.arange(-10, 10, 0.1)
X_new = pd.DataFrame(X_new.T)
preds = model.predicts(X_new)
preds = pd.DataFrame(preds, columns=['mean', 'var'])
preds['x'] = X_new
print model.hypers
plt.plot(preds['x'], preds['mean'])
plt.plot(preds['x'], preds['mean']+preds['var'], 'k--')
plt.plot(preds['x'], preds['mean']-preds['var'], 'k--')
plt.plot(X, Y, 'o')
plt.show()