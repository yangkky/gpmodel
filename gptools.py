import matplotlib.pyplot as plt
import gpmodel
import math
import numpy as np
import gpkernel

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
        
def plot_LOO(Xs, Ys, kernel,save_as=None, lab=''):
    std = []
    predicted_Ys = []
    for i in Xs.index:
        train_inds = list(set(Xs.index) - set(i))
        train_Xs = Xs.loc[train_inds]
        train_Ys = Ys.loc[train_inds]
        verify = Xs.loc[[i]]
       
        model = gpmodel.GPModel(train_Xs,train_Ys, kernel)
        predicted = model.predicts(verify)
        if model.is_class():
            E = predicted[0]
        else:
            [(E,v)] = predicted
            std.append(math.pow(v,0.5))
        predicted_Ys.append (E)
    plot_predictions (Ys.tolist(), predicted_Ys, label=lab, line=False, file_name=save_as)