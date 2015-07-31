''' Classes for doing Gaussian process models of proteins'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import math
from sys import exit


      
class GPModel(object):
    """A Gaussian process model. 

    Attributes:
        X_seqs (DataFrame): The sequences in the training set
        Y (Series): The outputs for the training set
        kern (kernel): a kernel for calculating covariances
        K (DataFrame): Covariance matrix
        Ky (np.matrix): noisy covariance matrix [var_p*K+var_n*I]
        L (np.matrix): lower triangular Cholesky decomposition of Ky
        alpha (np.matrix): L.T\(L\Y)
        ML (float): The negative log marginal likelihood
    """
        
    def __init__ (self, X_seqs, Y, kern, guesses=[1.,1.]):
        self.X_seqs = X_seqs
        self.Y = Y
        self.kern = kern
        self.K = self.kern.make_K(X_seqs)
        minimize_res = minimize(self.log_ML,(guesses), bounds=[(1e-3,None),(1e-5,None)], method='L-BFGS-B')
        self.var_n,self.var_p = minimize_res['x']
        self.Ky = self.var_p*self.K+self.var_n*np.identity(len(X_seqs))
        self.L = np.linalg.cholesky(self.Ky)
        self.ML = minimize_res['fun']
        self.alpha = np.linalg.lstsq(self.L.T,np.linalg.lstsq (self.L, np.matrix(self.Y).T)[0])[0]
        
            
    def predict (self, k, k_star):
        """ Predicts the mean and variance of the output for each of new_seqs
        
        Uses Equations 2.23 and 2.24 of RW
        
        Parameters: 
            k (np.matrix): k in equations 2.23 and 2.24
            k_star (float): k* in equation 2.24
            
        Returns: 
            res (tuple): (E,v) as floats
        """
        E = k*self.alpha
        v = np.linalg.lstsq(self.L,k.T)[0]
        var = k_star - v.T*v
        return (E.item(),var.item())
    
    def log_ML (self,variances):
        """ Returns the negative log marginal likelihood.  
        
        Parameters: 
            variances (iterable): var_n and var_p
    
        Uses RW Equation 5.8
        """
        Y_mat = np.matrix(self.Y)
        var_n,var_p = variances
        K_mat = np.matrix (self.K)
        Ky = K_mat*var_p+np.identity(len(K_mat))*var_n
        L = np.linalg.cholesky (Ky)
        alpha = np.linalg.lstsq(L.T,np.linalg.lstsq (L, np.matrix(Y_mat).T)[0])[0]
        try:
            ML = (0.5*Y_mat*alpha + 0.5*math.log(np.linalg.det(L)**2) + len(Y_mat)/2*math.log(2*math.pi)).item()
        except:
            print np.linalg.det(L)
            exit ('')
        return ML
        
    def predicts (self, new_seqs):
        """ Calculates predicted (mean, variance) for each sequence in new_seqs
    
        Uses Equations 2.23 and 2.24 of RW
        Parameters: 
            new_seqs (DataFrame): sequences to predict
            
         Returns: 
            predictions (list): (E,v) as floats
        """
        predictions = []
        for ns in [new_seqs.loc[i] for i in new_seqs.index]:
            k = np.matrix([self.kern.calc_kernel(ns,seq1,self.var_p) for seq1 in [self.X_seqs.loc[i] for i in self.X_seqs.index]])
            k_star = self.kern.calc_kernel(ns,ns,self.var_p)
            predictions.append(self.predict(k, k_star))
        return predictions   


        