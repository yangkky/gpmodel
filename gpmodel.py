''' Classes for doing Gaussian process models of proteins'''

import numpy as np
from scipy.optimize import minimize
import math
from sys import exit
import scipy
import pandas as pd



class GPModel(object):
    """A Gaussian process model.

    Attributes:
        X_seqs (DataFrame): The sequences in the training set
        Y (Series): The outputs for the training set
        kern (kernel): a kernel for calculating covariances
        regr (Boolean): classification or regression
        K (DataFrame): Covariance matrix
        Ky (np.matrix): noisy covariance matrix [var_p*K+var_n*I]
        L (np.matrix): lower triangular Cholesky decomposition of Ky
        alpha (np.matrix): L.T\(L\Y)
        ML (float): The negative log marginal likelihood
        l (int): number of training samples
    """
    def __init__ (self, X_seqs, Y, kern, guesses=[1.,100.]):
        self.X_seqs = X_seqs
        self.Y = Y
        self.l = len(Y)
        self.kern = kern
        self.K = self.kern.make_K(X_seqs)
        self.kern.train(X_seqs)
        # check if regression or classification
        self.regr = not self.is_class()
        if self.regr:
            minimize_res = minimize(self.log_ML,
                                    (guesses),
                                    bounds=[(1e-4,None),
                                            (1e-5,None)],
                                    method='L-BFGS-B')
            self.set_hypers(minimize_res['x'])
            self.ML = minimize_res['fun']

        else:
            minimize_res = minimize(self.logistic_log_ML,
                                    10.,
                                    bounds=[(1e-4, None)])
            self.set_hypers(minimize_res['x'])

    def set_hypers(self,hypers):
        '''
        Set the hyperparameters to hypers and update all dependent values
        '''
        if self.regr:
            self.var_n,self.var_p = hypers
            self.Ky = self.var_p*self.K+self.var_n*np.identity(len(self.X_seqs))
            self.L = np.linalg.cholesky(self.Ky)
            self.alpha = np.linalg.lstsq(self.L.T,
                                         np.linalg.lstsq (self.L,
                                                          np.matrix(self.Y).T)[0])[0]
        else:
            self.var_p = hypers[0]
            self.f_hat = self.find_F(var_p=self.var_p)
            self.W = self.hess (self.f_hat)
            self.W_root = scipy.linalg.sqrtm(self.W)
            self.Ky = np.matrix (self.K*self.var_p)
            self.L = np.linalg.cholesky (np.matrix(np.eye(self.l))+self.W_root*self.Ky*self.W_root)
            self.grad = np.matrix(np.diag(self.grad_log_logistic_likelihood (self.Y, self.f_hat)))

    def predict (self, k, k_star):
        """ Predicts the mean and variance for one new sequence given its
        covariance vector.

        Uses Equations 2.23 and 2.24 of RW

        Parameters:
            k (np.matrix): k in equations 2.23 and 2.24
            k_star (float): k* in equation 2.24

        Returns:
            res (tuple): (E,v) as floats for regression, pi_star for
                classification
        """
        if self.regr:
            E = k*self.alpha
            v = np.linalg.lstsq(self.L,k.T)[0]
            var = k_star - v.T*v
            return (E.item(),var.item())
        else:
            f_bar = k*self.grad.T
            v = np.linalg.lstsq(self.L, self.W_root*k.T)[0]
            var = k_star - v.T*v
            i = 10
            pi_star = scipy.integrate.quad(self.p_integral,
                                           -i*var+f_bar,
                                           f_bar+i*var,
                                           args=(f_bar.item(), var.item()))[0]
            return (pi_star)

    def p_integral (self, z, mean, variance):
        '''Equation to integrate when calculating pi_star for classification'''
        try:
            first = 1./(1+math.exp(-z))
        except OverflowError:
            first=0.
        second = 1/math.sqrt(2*math.pi*variance)
        third = math.exp(-(z-mean)**2/(2*variance))
        return first*second*third


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
        first = 0.5*Y_mat*alpha
        second = sum([math.log(l) for l in np.diag(L)])
        third = len(K_mat)/2.*math.log(2*math.pi)
        ML = (first+second+third).item()
        # log[det(Ky)] = 2*sum(log(diag(L))) is a property of the Cholesky decomposition
        # Y.T*Ky^-1*Y = L.T\(L\Y.T) (another property of the Cholesky)
        return ML

    def predicts (self, new_seqs):
        """ Calculates predicted (mean, variance) for each sequence in new_seqs

        Uses Equations 2.23 and 2.24 of RW
        Parameters:
            new_seqs (DataFrame): sequences to predict. They must have unique indices.

         Returns:
            predictions (list): (E,v) as floats
        """
        predictions = []
        self.kern.train(new_seqs)
        for ns in new_seqs.index:
            k = np.matrix([self.kern.calc_kernel(ns,seq1,self.var_p) \
                           for seq1 in self.X_seqs.index])
            k_star = self.kern.calc_kernel(ns,ns,self.var_p)
            predictions.append(self.predict(k, k_star))
        self.kern.delete(new_seqs)
        return predictions

    def is_class (self):
        '''True if Y only contains values 1 and -1, otherwise False'''
        return all (y in [-1,1] for y in self.Y)

    def logistic_likelihood (self,y,f):
        '''Calculates the logistic probability of the outcome y given the latent
        variable f according to Equation 3.2 in RW

        Parameters:
            y (float): +/- 1
            f (float): value of latent function

        Returns:
            float
        '''
        if int(y) not in [-1,1]:
            raise RuntimeError ('y must be -1 or 1')
        return 1./(1+math.exp(-y*f))

    def log_logistic_likelihood (self,Y, F):
        """ Calculates the log logistic likelihood of the outcomes Y given the
        latent variables F. log[p(Y|f)]
        Uses Equation 3.15 of RW

        Parameters:
            Y (Series): outputs, +/-1
            F (Series): values for the latent function

        Returns:
            lll (float): log logistic likelihood
        """
        if len(Y) != len(F):
            raise RuntimeError ('Y and F must be the same length')
        lll = sum(np.log([self.logistic_likelihood(y,f) for y,f in zip(Y,F)]))
        return lll

    def grad_log_logistic_likelihood (self,Y,F):
        """ Calculates the gradient of the logistic likelihood of the outcomes
        Y given the latent variables F.
        Uses Equation 3.15 of RW

        Parameters:
            Y (Series): outputs, +/-1
            F (Series): values for the latent function

        Returns:
            glll (np.matrix): diagonal log likelihood matrix
        """
        glll = np.matrix(np.zeros([self.l,self.l]))
        for i in range(self.l):
            glll[i,i] = (Y[i]+1.)/2. - self.logistic_likelihood(1.,F[i])
        return glll

    def hess (self,F):
        """ Calculates the negative Hessian of the logistic likelihood according to
        Equation 3.15 of RW
        Parameters:
            F (Series): values for the latent function

        Returns:
            W (np.matrix): diagonal negtive Hessian of the log likelihood matrix
        """
        W = np.matrix(np.zeros([self.l,self.l]))
        for i in range(self.l):
            pi_i = self.logistic_likelihood(1., F[i])
            W[i,i] = pi_i*(1-pi_i)
        return W

    def find_F (self, var_p=1, guess=None, threshold=.0001, evals=1000):
        """Calculates f_hat according to Algorithm 3.1 in RW

        Returns:
            f_hat (pd.Series)
        """
        l = len(self.Y)
        if guess == None:
            f_hat = pd.Series(np.zeros(l))
        elif len(guess) == l:
            f_hat = guess
        else:
            exit ('Initial guess must have same dimensions as Y')


        K_mat = var_p*np.matrix (self.K)
#         r = minimize(self.logq, f_hat, args=var_p)
#         F = r['x']
#         F_test = K_mat*np.matrix(np.diag(self.grad_log_logistic_likelihood(self.Y,F))).T
#         for f,ft in zip(F, F_test):
#             print f, ft
#         exit('')
        n_below = 0
        for i in range (evals):
            # find new f_hat
            W = self.hess (f_hat)
            W_root = scipy.linalg.sqrtm(W)
            f_hat_mat = np.matrix (f_hat)
            L = np.linalg.cholesky (np.matrix(np.eye(l))+W_root*K_mat*W_root)
            b = W*f_hat_mat.T + np.matrix(np.diag(self.grad_log_logistic_likelihood (self.Y,f_hat))).T
            a = b - W_root*np.linalg.lstsq(L.T,np.linalg.lstsq(L,W_root*K_mat*b)[0])[0]
            f_new = K_mat*a
            f_new = pd.Series([f.item() for f in np.nditer(f_new)])
            # find error between new and old f_hat
            sq_error = sum([(fh-fn)**2 for fh,fn in zip (f_hat, f_new)])
            if sq_error/sum([fn**2 for fn in f_new]) < threshold:
                n_below += 1
            else:
                n_below = 0
            if n_below > 9:
                f_new.index = self.X_seqs.index
                return f_new
            f_hat = f_new
        exit ('Maximum number of evaluations reached without convergence')


    def logq(self, F, var_p=1):
        '''
        Finds the negative log marginal likelihood for Laplace's approximation
        Equation 5.20 or 3.12 from RW, as described in Algorithm 5.1
        Parameters:
            var_p (float)
            F (Series): values for the latent function

        Returns:
            logq (float)
        '''
        l = self.l
        K_mat = var_p*np.matrix (self.K)
        W = self.hess (F)
        W_root = scipy.linalg.sqrtm(W)
        F_mat = np.matrix (F)
        L = np.linalg.cholesky (np.matrix(np.eye(l))+W_root*K_mat*W_root)
        b = W*F_mat.T + np.matrix(np.diag(self.grad_log_logistic_likelihood (self.Y,F))).T
        a = b - W_root*np.linalg.lstsq(L.T,np.linalg.lstsq(L,W_root*K_mat*b)[0])[0]
        logq = 0.5*a.T*F_mat.T - self.log_logistic_likelihood(self.Y, F) + sum(np.log(np.diag(L)))
        return logq

    def logistic_log_ML (self, var_p):
        '''
        Finds the negative log marginal likelihood for Laplace's approximation
        Equation 5.20 or 3.12 from RW, as described in Algorithm 5.1
        Parameters:
            var_p (list) of size 1
        '''
        var_p = var_p[0]
        f_hat = self.find_F(var_p=var_p) # use Algorithm 3.1 to find mode
        ML = self.logq(f_hat, var_p=var_p)
        #print 'var_p = %f, ML = %f' %(var_p,ML)
        return ML



if __name__=="__main__":
    Y = pd.Series([1,-1,1,1])
    F = pd.Series([-1.0, 4.0, 4.0, 0.1])
    print GPModel.grad_log_logistic_likelihood(Y,F)

