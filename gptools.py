import matplotlib.pyplot as plt
import gpmodel
import math
import numpy as np
import gpkernel
import pandas as pd

######################################################################
# Here are some plotting tools that are generally useful
######################################################################
def plot_predictions (real_Ys, predicted_Ys,stds=None,file_name=None,title='',label='', line=True):
    if stds is None:
        plt.plot (real_Ys, predicted_Ys, 'g.')
    else:
        plt.errorbar (real_Ys, predicted_Ys, yerr = [stds, stds], fmt = 'g.')
    small = min(set(real_Ys) | set(predicted_Ys))*1.1
    large = max(set(real_Ys) | set(predicted_Ys))*1.1
    if line:
        plt.plot ([small, large], [small, large], 'b--')
    plt.xlabel ('Actual ' + label)
    plt.ylabel ('Predicted ' + label)
    plt.title (title)
    plt.xlim (small, large)
    plt.text(small*.9, large*.7, 'R = %.3f' %np.corrcoef(real_Ys, predicted_Ys)[0,1])
    if not file_name is None:
        plt.savefig (file_name)

def plot_ROC (real_Ys, pis, file_name=None,title='',label=''):
    fps = np.empty(np.shape(pis))
    tps = np.empty(np.shape(pis))
    n_false = sum([1 if y == -1 else 0 for y in real_Ys])
    n_true = len(real_Ys) - n_false
    for i,cut in enumerate(sorted(pis)):
        predictions = [1 if p > cut else -1 for p in pis]
        fp = 0
        tp = 0
        for y,p in zip(real_Ys, predictions):
            if y == 1 and p == 1:
                tp += 1
            elif y==-1 and p == 1:
                fp +=1
        fps[i] = float(fp)/n_false
        tps[i] = float(tp)/n_true
    plt.plot (fps,tps,'.-')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    if not file_name is None:
        plt.savefig (file_name)

def plot_LOO(Xs, Ys,kernel,save_as=None, lab=''):
    std = []
    predicted_Ys = []
    for i in Xs.index:
        train_inds = list(set(Xs.index) - set(i))
        train_Xs = Xs.loc[train_inds]
        train_Ys = Ys.loc[train_inds]
        verify = Xs.loc[[i]]
        print 'Building model for ' + str(i)
        model = gpmodel.GPModel(train_Xs,train_Ys, kernel)
        print 'Making prediction for ' + str(i)
        predicted = model.predicts(verify)
        if model.is_class():
            E = predicted[0]
        else:
            [(E,v)] = predicted
            std.append(math.pow(v,0.5))
        predicted_Ys.append (E)
    plot_predictions (Ys.tolist(), predicted_Ys, stds=None,label=lab, line=True, file_name=save_as)
    return predicted_Ys,std

if __name__ == "__main__":
    import random
    num = 100
    real_Ys = pd.Series([random.choice([-1,1]) for i in range(num)])
    pis = [random.random() for i in range(num)]
    plot_ROC(real_Ys,pis)
    plt.show()
