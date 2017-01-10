from sys import exit
import math

import numpy as np
import pandas as pd
from sklearn import metrics
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from gpmodel import gpmodel
from gpmodel import gpkernel


rc = {'lines.linewidth': 3,
      'axes.labelsize': 30,
      'axes.titlesize': 40,
      'axes.edgecolor': 'black'}
sns.set_context('talk', rc=rc)
sns.set_style('whitegrid', rc=rc)

######################################################################
# Here are some plotting tools that are generally useful
######################################################################


def cv(Xs, Ys, model, n_train, replicates=50, keep_inds=[]):
    ''' Returns cross-validation predictions.

    Parameters:
        Xs (pd.DataFrame)
        Ys (pd.DataFrame)
        model (GPModel)
        n_train (int)
        replicates (int)
        keep_inds (iterable)

    Returns:
        predicted (list)
        actual (list)
        R (float)
    '''
    # check that n_train is less than the number of observations
    if n_train + len(keep_inds) >= len(Xs):
        raise ValueError('n_train must be less than len(Xs) - len(keep_inds)')
    if not np.array_equal(Xs.index, Ys.index):
        raise ValueError('Xs and Ys must have same index.')
    changed_index = list(set(Xs.index) - set(keep_inds))
    actual = []
    predicted = []
    if all(y in [-1, 1] for y in Ys):
        regr = False
    else:
        regr = True
    if n_train == len(Xs) - 1 - len(keep_inds):
        for test_inds in changed_index:
            train_inds = list(set(Xs.index) - set(test_inds))
            model.fit(Xs.loc[train_inds], Ys.loc[train_inds])
            preds = model.predict(Xs.loc[[test_inds]])
            predicted += [p[0] for p in preds]
            actual += list(Ys.loc[[test_inds]])
        if not regr:
            fpr, tpr, _ = metrics.roc_curve(actual, predicted)
            metric = metrics.auc(fpr, tpr)
        else:
            metric = np.corrcoef(predicted, actual)[0, 1]
        return predicted, actual, metric
    Rs = []
    for r in range(replicates):
        # pick indices for train and test sets
        train_inds = np.random.choice(changed_index, n_train, replace=False)
        train_inds = list(train_inds) + keep_inds
        test_inds = list(set(Xs.index) - set(train_inds))
        if all(Ys.loc[test_inds] == 1) or all(Ys.loc[test_inds] == -1):
            continue
        # fit the model
        model.fit(Xs.loc[train_inds], Ys.loc[train_inds])
        # make predictions
        preds = model.predict(Xs.loc[test_inds])
        predictions = [p[0] for p in preds]
        truth = list(Ys.loc[test_inds])
        predicted += predictions
        actual += truth
        if regr:
            Rs.append(np.corrcoef(predictions, truth)[0, 1])
        else:
            fpr, tpr, _ = metrics.roc_curve(truth, predictions)
            Rs.append(metrics.auc(fpr, tpr))

    return predicted, actual, np.mean(Rs)


def plot_predictions(real_Ys, predicted_Ys, stds=None,
                     file_name=None, title='', label='', line=False):
    if stds is None:
        plt.plot(real_Ys, predicted_Ys, '.', color='black')
    else:
        plt.errorbar(real_Ys, predicted_Ys, yerr=[stds, stds], fmt='k.')
    small = min(set(real_Ys) | set(predicted_Ys))*1.1
    if small == 0:
        small = real_Ys.mean()/10.0
    large = max(set(real_Ys) | set(predicted_Ys))*1.1
    if line:
        plt.plot([small, large], [small, large], 'k--', alpha=0.3)
    plt.xlabel('Actual ' + label)
    plt.ylabel('Predicted ' + label)
    plt.title(title)
    plt.xlim(small, large)

    if small <= 0:
        left = small*0.8
    else:
        left = small*1.2
    if large <= 0:
        right = large*1.3
    else:
        right = large*0.7
    if file_name is not None:
        plt.savefig(file_name)


def plot_ROC(real_Ys, pis, file_name=None, title=''):
    fpr, tpr, _ = metrics.roc_curve(real_Ys, pis)
    plt.plot(fpr, tpr, 'k.-')
    plt.xlim([-.1, 1.1])
    plt.ylim([-.1, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    auc = metrics.auc(fpr, tpr)
    if file_name is not None:
        plt.savefig(file_name)
    return auc


def plot_LOO(Xs, Ys, kernel, save_as=None, lab=''):
    std = []
    predicted_Ys = []
    count = 0
    for i in Xs.index:
        train_inds = list(set(Xs.index) - set(i))
        train_Xs = Xs.loc[train_inds]
        train_Ys = Ys.loc[train_inds]
        verify = Xs.loc[[i]]
        print('Building model for ' + str(i))
        model = gpmodel.GPModel(train_Xs, train_Ys, kernel,
                                guesses=[1, 10], remember=(count < 1))
        count += 1
        print('Making prediction for ' + str(i))
        predicted = model.predicts(verify, delete=False)
        if model.is_class():
            E = predicted[0]
        else:
            try:
                [(E, v)] = predicted
            except ValueError:
                print('ValueError', i, predicted)
            std.append(math.pow(v, 0.5))
        predicted_Ys.append(E)
    return predicted_Ys


def log_marginal_likelihood(variances, model):
    """ Returns the negative log marginal likelihood.

        Parameters:
            model: a GPmodel regression model
            variances (iterable): var_n and var_p

        Uses RW Equation 5.8
    """
    if model.regr:
        Y_mat = np.matrix(model.normed_Y)
        var_n, var_p = variances
        K_mat = np.matrix(model.K)
        Ky = K_mat*var_p+np.identity(len(K_mat))*var_n
        L = cholesky(Ky)
        alpha = np.linalg.lstsq(L.T,
                                np.linalg.lstsq(L, np.matrix(Y_mat).T)[0])[0]
        first = -0.5*Y_mat*alpha
        second = -sum([math.log(l) for l in np.diag(L)])
        third = -len(K_mat)/2.*math.log(2*math.pi)
        ML = (first+second+third).item()
        return (first, second, third, ML)
    else:
        Y_mat = np.matrix(model.Y)
        vp = variances
        f_hat = model.find_F(var_p=vp)
        K_mat = vp*np.matrix(model.K)
        W = model.hess(f_hat)
        W_root = scipy.linalg.sqrtm(W)
        F_mat = np.matrix(f_hat)
        ell = len(model.Y)
        L = cholesky(np.matrix(np.eye(ell))+W_root*K_mat*W_root)
        b = W*F_mat.T + np.matrix(np.diag(model.grad_log_logistic_likelihood
                                          (model.Y, f_hat))).T
        a = b - W_root*np.linalg.lstsq(L.T,
                                       np.linalg.lstsq(L,
                                                       W_root*K_mat*b)[0])[0]
        fit = -0.5*a.T*F_mat.T + model.log_logistic_likelihood(model.Y, f_hat)
        complexity = -sum(np.log(np.diag(L)))
        return (fit, complexity, fit+complexity)


def plot_ML_contour(model, ranges, save_as=None, lab='', n=100, n_levels=10):
    """
    Make a plot of how ML varies with the hyperparameters for a given model
    """
    if model.regr:
        vns = np.linspace(ranges[0][0], ranges[0][1], n)
        vps = np.linspace(ranges[1][0], ranges[1][1], n)
        nn, pp = np.meshgrid(vns, vps)
        log_ML = np.empty_like(nn)
        for j in range(len(vns)):
            for i in range(len(vps)):
                log_ML[j, i] = -model.log_ML((nn[i, j], pp[i, j]))
        levels = np.linspace(log_ML.min(), log_ML.max(), n_levels)
        print(levels)
        cs = plt.contour(nn, pp, log_ML, alpha=0.7, levels=levels)
        plt.clabel(cs)
        plt.xlabel(r'$\sigma_n^2$')
        plt.ylabel(r'$\sigma_p^2$')
        plt.title(lab)
        return log_ML

    else:
        vps = np.linspace(ranges[0], ranges[1], n)
        log_ML = np.empty_like(vps)
        for i, p in enumerate(vps):
            log_ML[i] = -model.logistic_log_ML([p])

        plt.plot(vps, log_ML, '.-')
        plt.xlabel(r'$\sigma_p^2$')
        plt.ylabel('log marginal likelihood')
        plt.title(lab)
        return log_ML


def plot_ML_parts(model, ranges, lab='', n=100,
                  plots=['log_ML', 'fit', 'complexity']):
    if model.regr:
        if len(ranges[0]) == 1:
            indpt = 'var_p**2'
            held = 'var_n**2'
            cons = ranges[0][0]
            lower = ranges[1][0]
            upper = ranges[1][1]
        elif len(ranges[1]) == 1:
            indpt = 'var_n**2'
            held = 'var_p**2'
            cons = ranges[1][0]
            lower = ranges[0][0]
            upper = ranges[0][1]
        varied = np.linspace(lower, upper, n)
        ML = np.empty_like(varied)
        fit = np.empty_like(varied)
        complexity = np.empty_like(varied)
        norm = np.empty_like(varied)
        for i, v in enumerate(varied):
            if indpt == 'var_p**2':
                fit[i], complexity[i], norm[i], ML[i] = \
                    log_marginal_likelihood((cons, v), model)
            else:
                fit[i], complexity[i], norm[i], ML[i] = \
                   log_marginal_likelihood((v, cons), model)
    else:
        indpt = 'var_p**2'
        varied = np.linspace(ranges[0], ranges[1], n)
        ML = np.empty_like(varied)
        fit = np.empty_like(varied)
        norm = np.empty_like(varied)
        complexity = np.empty_like(varied)
        for i, v in enumerate(varied):
            fit[i], complexity[i], ML[i] = log_marginal_likelihood(v, model)

    plot_dict = {'log_ML': ML, 'fit': fit,
                 'complexity': complexity, 'norm': norm}
    for pl in plots:
        plt.plot(varied, plot_dict[pl])
    plt.legend(plots)
    plt.xlabel(indpt)
    if model.regr:
        plt.title(lab + ' ' + held + ' = %f' %cons)
    else:
        plt.title(lab)
    return (fit, complexity, ML)


if __name__ == "__main__":
    pass
