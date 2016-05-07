import sys
sys.path.append ('/Users/seinchin/Documents/Caltech/Arnold Lab/Programming tools/GPModel')
import gpkernel,gpmodel
import pandas as pd
import numpy as np
import math
import pytest
import scipy
from scipy import stats
import os
from sklearn import metrics

seqs = pd.DataFrame([['R','Y','M','A'],['R','T','H','A'], ['R','T','M','A']],
                    index=['A','B','C'], columns=[0,1,2,3])
seqs = seqs.append(seqs.iloc[0])
seqs.index = ['A','B','C','A']

space = [('R'), ('Y', 'T'), ('M', 'H'), ('A')]
contacts = [(0,1),(2,3)]

class_Ys = pd.Series([-1,1,1,-1],index=seqs.index)
reg_Ys = pd.Series([-1,1,0.5,-.4],index=seqs.index)
struct = gpkernel.StructureKernel (contacts)
SE_kern = gpkernel.StructureSEKernel(contacts)

test_seqs = pd.DataFrame([['R','Y','M','A'],['R','T','H','A']],index=['A','D'])

def test_creation():
    print 'Testing constructors, fits, and pickling method...'

    model = gpmodel.GPModel(struct, objective='LOO_log_p', guesses=(1.0,))
    assert model.objective == model._LOO_log_p
    pytest.raises(AttributeError, 'model.fit(seqs, reg_Ys)')
    model._set_params(objective='log_ML')
    assert model.objective == model._log_ML

    # create a model
    model = gpmodel.GPModel(struct, guesses=(2,2), objective='LOO_log_p')
    # pickle the model
    model.dump('test_creation.pkl')
    # reload the model
    model = gpmodel.GPModel.load('test_creation.pkl')
    # delete the pickle
    os.remove('test_creation.pkl')
    # test the model
    assert model.objective == model._LOO_log_p
    assert model.guesses == (2,2)

    # fit the model
    model.fit(seqs, reg_Ys)
    ML = model.ML
    lp = model.log_p
    hypers = model.hypers
    # pickle the model
    model.dump('test_creation.pkl')
    # reload the model
    model = gpmodel.GPModel.load('test_creation.pkl')
    # delete the pickle
    os.remove('test_creation.pkl')
    # test the model
    assert model.ML == ML
    assert model.log_p == lp
    assert model.hypers == hypers

    model = gpmodel.GPModel(struct, guesses=(2,2), objective='LOO_log_p')
    model._set_params(X=seqs, Y=reg_Ys)
    # pickle the model
    model.dump('test_creation.pkl')
    # reload the model
    model = gpmodel.GPModel.load('test_creation.pkl')
    # delete the pickle
    os.remove('test_creation.pkl')
    # test the model
    assert model.ML == ML
    assert model.log_p == lp
    assert model.hypers == hypers

def test_score():
    print 'Testing score ...'
    model = gpmodel.GPModel(struct)
    model.fit(seqs, reg_Ys)
    preds = model.predicts(seqs)
    pred_Y = [p[0] for p in preds]
    r1 = stats.rankdata(reg_Ys)
    r2 = stats.rankdata(pred_Y)
    kendall = stats.kendalltau(r1, r2).correlation
    R = np.corrcoef(reg_Ys, pred_Y)[0,1]
    u = sum((y1-y2)**2 for y1, y2 in zip(reg_Ys, pred_Y))
    m = sum(reg_Ys)/len(reg_Ys)
    v = sum((y1-m)**2 for y1 in reg_Ys)
    R2 = 1 - u/v
    scores = model.score(seqs, reg_Ys, 'R', 'kendalltau', 'R2')
    assert R == scores['R']
    assert kendall == scores['kendalltau']
    assert R2 == scores['R2']

    scores = model.score(seqs, reg_Ys)
    assert kendall == scores

    scores = model.score(seqs, reg_Ys, 'R')
    assert R == scores

    scores = model.score(seqs, reg_Ys, 'R', 'R2')
    assert R == scores['R']
    assert R2 == scores['R2']

    pytest.raises(ValueError, 'model.score(seqs, reg_Ys, "R3")')

    model.fit(seqs, class_Ys)
    preds = [p[0] for p in model.predicts(seqs)]
    score = model.score(seqs, class_Ys)
    fpr, tpr, _ = metrics.roc_curve(class_Ys, preds)
    AUC = metrics.auc(fpr, tpr)
    assert AUC == score

    print 'score passes all tests. '

def test_regression ():
    model = gpmodel.GPModel(struct)
    model.fit(seqs, reg_Ys)
    assert close_enough(model.hypers.var_p, 0.63016924576335664),\
    'Regression model.hypers.var_p is incorrect'
    assert close_enough(model.hypers.var_n, 0.18044635639161319),\
    'Regression model.hypers.var_n is incorrect'
    assert close_enough(model.ML, 4.59859002013),\
    'Regression model.ML is incorrect'
    assert close_enough(model.log_p, 3.79099390643),\
    'Regression model.log_p is incorrect'
    assert np.isclose(model._K,
                      struct.make_K(seqs,
                                    [model.hypers.var_p],
                                    normalize=True)).all()
    assert model._ell ==  len(seqs.index)

    m = reg_Ys.mean()
    s = reg_Ys.std()
    assert model.mean == m
    assert model.std == s

    normed_Ys = (reg_Ys - m) / s
    assert (y1==y2 for y1, y2 in zip (normed_Ys, model.normed_Y)), \
    'Model does not normalize Y-values correctly.'


    Y_mat = np.matrix(normed_Ys)

    print 'Testing objective functions for regression models...'
    # test marginal likelihood
    vp = 1.0
    vn = model.hypers.var_n
    K_mat = np.matrix(struct.make_K(seqs, normalize=True))
    Ky = vp*K_mat + np.eye(len(reg_Ys))*vn
    first = 0.5*Y_mat*np.linalg.inv(Ky)*Y_mat.T
    second = math.log(np.linalg.det(Ky))*0.5
    third = len(reg_Ys)*math.log(2*math.pi)*.5
    ML = first + second + third

    # because floating point precision
    assert close_enough(model._log_ML((vn,vp)), ML.item()), \
    'log_ML fails: ' + ' '.join([str(first),str(second),str(third)])

    # test LOO log predictive probability
    K_inv = np.linalg.inv(Ky)
    mus = np.diag(Y_mat.T - K_inv*Y_mat.T/K_inv)
    vs = np.diag(1/K_inv)

    res1 = model.LOO_res((vn,vp))
    res2 = pd.DataFrame(zip(mus, vs), index=normed_Ys.index,
                                                  columns=['mu','v'])
    assert res1.equals(res2), 'Regression model does not correctly predict LOO values'

    log_p_1 = model._LOO_log_p((vn,vp))
    log_p_2 = 0.5*np.sum(np.log(res2['v']) + np.power(normed_Ys-res2['mu'],2)/res2['v'] \
                          + np.log(2*np.pi))
    assert close_enough(log_p_1, log_p_2), \
    'Regression model does not correctly calculate LOO log predictive probability'



    print 'Testing regression ... '
    # test predictions
    kA = np.matrix([model.kern.calc_kernel(test_seqs.loc['A'],
                                           seq1, [model.hypers.var_p],
                                           normalize=True) for seq1 \
                    in [seqs.iloc[i] for i in range(len(seqs.index))]])
    kD = np.matrix([model.kern.calc_kernel(test_seqs.loc['D'],
                                           seq1, [model.hypers.var_p],
                                           normalize=True) for seq1 \
                    in [seqs.iloc[i] for i in range(len(seqs.index))]])
    EA = (kA*np.linalg.inv(model._Ky)*Y_mat.T) * s + m
    ED = (kD*np.linalg.inv(model._Ky)*Y_mat.T) * s + m
    k_star_A = model.kern.calc_kernel(test_seqs.loc['A'],
                                      test_seqs.loc['A'],
                                      normalize=True)*model.hypers.var_p\
               + model.hypers.var_n
    k_star_D = model.kern.calc_kernel(test_seqs.loc['D'],
                                      test_seqs.loc['D'],
                                      normalize=True)*model.hypers.var_p\
                   + model.hypers.var_n

    var_A = (k_star_A - kA*np.linalg.inv(model._Ky)*kA.T) * s**2
    var_D = (k_star_D - kD*np.linalg.inv(model._Ky)*kD.T) * s**2
    predictions = model.predicts(test_seqs,delete=False)

    assert close_enough(EA, predictions[0][0])
    assert close_enough(ED, predictions[1][0])
    assert close_enough(var_A, predictions[0][1])
    assert close_enough(var_D, predictions[1][1])

    [(E,v)] = model.predicts(test_seqs.loc[['D']])
    assert close_enough(E,ED)
    assert close_enough(v,var_D)

    # test regression with StructureSEKernel
    model = gpmodel.GPModel(SE_kern)
    model.fit(seqs, reg_Ys)
    kA = np.matrix([model.kern.calc_kernel(test_seqs.loc['A'],
                                           seq1, [model.hypers.sigma_f,
                                                  model.hypers.ell]) for seq1 \
                    in [seqs.iloc[i] for i in range(len(seqs.index))]])
    kD = np.matrix([model.kern.calc_kernel(test_seqs.loc['D'],
                                           seq1, [model.hypers.sigma_f,
                                                  model.hypers.ell]) for seq1 \
                    in [seqs.iloc[i] for i in range(len(seqs.index))]])
    EA = (kA*np.linalg.inv(model._Ky)*Y_mat.T) * s + m
    ED = (kD*np.linalg.inv(model._Ky)*Y_mat.T) * s + m
    k_star_A = model.kern.calc_kernel(test_seqs.loc['A'],
                                      test_seqs.loc['A'],
                                      [model.hypers.sigma_f,
                                                  model.hypers.ell])\
                   + model.hypers.var_n

    k_star_D = model.kern.calc_kernel(test_seqs.loc['D'],
                                      test_seqs.loc['D'],
                                      [model.hypers.sigma_f,
                                                  model.hypers.ell])\
                   + model.hypers.var_n

    var_A = (k_star_A - kA*np.linalg.inv(model._Ky)*kA.T) * s**2
    var_D = (k_star_D - kD*np.linalg.inv(model._Ky)*kD.T) * s**2
    predictions = model.predicts(test_seqs,delete=False)
    assert close_enough(EA, predictions[0][0])
    assert close_enough(ED, predictions[1][0])
    assert close_enough(var_A, predictions[0][1])
    assert close_enough(var_D, predictions[1][1])

    [(E,v)] = model.predicts(test_seqs.loc[['D']])
    assert close_enough(E,ED)
    assert close_enough(v,var_D)

    h = model.hypers
    ML = model.ML
    model._set_params(hypers=(1,1,10))
    assert np.isclose([1,1,10], model.hypers).all()

    model._set_params(hypers=h)
    assert np.isclose(h, model.hypers).all()
    assert model.ML == ML

    print 'Regression model passes all tests.\n'

def test_classification ():
    print 'Testing constructors for classification models...'
    model = gpmodel.GPModel(struct, objective='LOO_log_p')
    pytest.raises(AttributeError, 'model.fit( seqs, class_Ys)')
    model = gpmodel.GPModel(struct)
    model.fit(seqs, class_Ys)
    test_F = pd.Series([-.5,.5,.6,.1])
    assert close_enough(model.hypers.var_p, 43.814739881967405),\
    'Classification model.hypers.var_p is incorrect'
    assert close_enough(model.ML, 2.45520196), \
    'Classification model.ML is incorrect'


    #assert model.K.equals(struct.make_K(seqs, normalize=True))
    assert model._ell ==  len(seqs.index)



    print 'Testing marginal likelihood for classification model...'
    assert model._logistic_likelihood(1,0) == 0.5
    assert model._logistic_likelihood(-1,0) == 0.5
    pytest.raises(RuntimeError, "model._logistic_likelihood(-2,1)")

    assert model._log_logistic_likelihood(class_Ys,
                                         pd.Series([0,0,0,0])) \
    == math.log(0.125/2)
    pytest.raises(RuntimeError,
                  "model._log_logistic_likelihood(class_Ys,pd.Series([0,0,0]))")

    # Test the gradient and hessian functions
    glll = model._grad_log_logistic_likelihood(class_Ys, test_F)
    for i in range(model._ell):
        for j in range(model._ell):
            if i != j:
                assert glll[i,j] == 0, 'Non-zero non-diagonal element of glll'
            else:
                assert close_enough(glll[i,j], (class_Ys.iloc[i]+1)/2. -
                                    model._logistic_likelihood(1,test_F.iloc[i]))

    hess = model._hess (test_F)
    for i in range(model._ell):
        for j in range(model._ell):
            if i != j:
                assert hess[i,j] == 0, 'Non-zero non-diagonal element of W'
            else:
                pi_i = model._logistic_likelihood(1,test_F.iloc[i])
                assert close_enough (hess[i,j], pi_i*(1-pi_i))

    # test find_F (Algorithm 3.1) by seeing if the result satisfies Eq 3.17 from RW
    f_hat = model._find_F(hypers=(1,))
    K = model.kern.make_K(seqs, hypers=(1,), normalize=True)
    K_mat = np.matrix(K)
    glll = np.matrix(np.diag(model._grad_log_logistic_likelihood
                             (class_Ys,f_hat))).T
    f_check = K_mat*glll
    for fh, fc in zip(f_hat, f_check):
        assert close_enough(fh,fc), 'find_F fails for var_p = 1.'

    vp = 0.1
    f_hat = model._find_F(hypers=(vp,))
    K_mat = K_mat*vp
    glll = np.matrix(np.diag(model._grad_log_logistic_likelihood
                             (class_Ys,f_hat))).T
    f_check = K_mat*glll
    for fh, fc in zip(f_hat, f_check):
        assert close_enough(fh,fc), 'find_F fails for var_p ~= 1.'

    # Test the functions that calculate marginal likelihood
    logq = model._logq(f_hat, hypers=(vp,))
    W = model._hess (f_hat)
    W_root = scipy.linalg.sqrtm(W)
    F_mat = np.matrix (f_hat)
    l = len(f_hat)
    L = np.linalg.cholesky (np.matrix(np.eye(l))+W_root*K_mat*W_root)
    b = W*F_mat.T + glll
    a = b - W_root*np.linalg.lstsq(L.T,np.linalg.lstsq(L,W_root*K_mat*b)[0])[0]
    check_q = 0.5*a.T*F_mat.T - model._log_logistic_likelihood(class_Ys, f_hat) \
    + sum(np.log(np.diag(L)))
    assert close_enough(check_q, logq)
    assert close_enough(model._log_ML([vp]), check_q)

    # Test the function that is integrated
    v = 0.4
    m = 0.5
    for z in [0.5, 0, 10000, -10000]:
        val = 1./(1+np.exp(-z))/math.sqrt(2*math.pi*v)*math.exp(-(z-m)**2/2/v)
        assert close_enough(val, model._p_integral(z,m,v))
    for z in [np.inf, -np.inf]:
        assert close_enough(0., model._p_integral(z,m,v))

    # test predictions
    preds = model.predicts(test_seqs)
    for p1, p2 in zip(preds, [0.19135281113445562, 0.7792366872177071]):
        assert close_enough(p1[0], p2), 'Predictions failed.'




    print 'Classification models pass all tests.'

def test_UB():
    model = gpmodel.GPModel(struct)
    model.fit(seqs, reg_Ys)
    new_seqs = pd.DataFrame([['R','Y','H','A'],
                             ['N','T','H','A'],
                             ['G','T','M','A'],
                             ['N', 'T', 'M', 'A']],
                            index=['1', '2', '3', '4'], columns=[0,1,2,3])
    print model.batch_UB_bandit(new_seqs, n=3)

def close_enough(f1,f2):
    return (f1-f2)**2 < 1e-7


if __name__=="__main__":
#     test_creation()
#     test_regression()
#     test_classification()
#     test_score
    test_UB()
