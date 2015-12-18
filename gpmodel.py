''' Classes for doing Gaussian process models of proteins'''

import numpy as np
from scipy.optimize import minimize
import math
from sys import exit
import scipy
import pandas as pd
from collections import namedtuple
import cPickle as pickle



class GPModel(object):
    """A Gaussian process model for proteins.

    Attributes:
        X_seqs (DataFrame): The sequences in the training set
        Y (Series): The outputs for the training set
        normed_Y (Series): Normalized outputs for the training set
        mean (float): mean of unnormed Ys
        std (float): standard deviation of unnormed Ys
        kern (GPKernel): a kernel for calculating covariances
        Hypers (namedtuple): the hyperparameters
        regr (Boolean): classification or regression
        K (pdDataFrame): Covariance matrix
        Ky (np.matrix): noisy covariance matrix [K+var_n*I]
        L (np.matrix): lower triangular Cholesky decomposition of Ky for
            regression models. Lower triangular Cholesky decomposition of
            (I + W_root*Ky*W_root.T) for classification models.
        alpha (np.matrix): L.T\(L\Y)
        ML (float): The negative log marginal likelihood
        log_p (float): the negative LOO log likelihood
        l (int): number of training samples
        f_hat (Series): MAP values of the latent function for training set
        W (np.matrix): negative Hessian of the log likelihood
        W_root (np.matrix): Square root of W
        grad (np.matrix): gradient of the log logistic likelihood
    """

    def __init__ (self, kern, **kwargs):
        """
        Create a new GPModel.

        Parameters:

            kern (GPKernel): kernel to use

        Optional keyword parameters:
            guesses (iterable): initial guesses for the hyperparameters.
                Default is [1 for _ in range(len(hypers))].
            objective (String): objective function to use in training model. Choices
                are 'log_ML' and 'LOO_log_p'. Classification must be trained
                on 'log_ML.' Default is 'log_ML'.
        """
        self.guesses = kwargs.get('guesses', None)
        objective = kwargs.get('objective', 'log_ML')
        hypers = kwargs.get('hypers', None)

        self.kern = kern


        if objective == 'log_ML':
            self.objective = self.log_ML
        elif objective == 'LOO_log_p':
            self.objective = self.LOO_log_p


    def fit(self, X_seqs, Y):
        '''
        Set the hyperparameters by training on the given data.
        Update all dependent values.

        Parameters:
            X_seqs (pandas.DataFrame): Sequences in training set
            Y (pandas.Series): measurements in training set
            guesses (iterable): initial guesses for the hyperparameters.
                Default is [1 for _ in range(len(hypers))].
            objective (function): objective function to use in training model. Choices
                are 'log_ML' and 'LOO_log_p'. Classification must be trained
                on 'log_ML'.
            hypers (iterable): hyperparameters to set. This overrides the other
                optional parameters.
        '''
        self.X_seqs = X_seqs
        self.Y = Y
        self.l = len(Y)
        self.kern.set_X(X_seqs)

        self.regr = not self.is_class()
        if self.regr:
            self.mean, self.std, self.normed_Y = self.normalize (self.Y)
            n_guesses = 1 + len(self.kern.hypers)
        else:
            n_guesses = len(self.kern.hypers)
            if self.objective == self.LOO_log_p:
                raise AttributeError\
                ('Classification models must be trained on marginal likelihood')

        if self.guesses == None:
            guesses = [0.9 for _ in range(n_guesses)]
        else:
            guesses = self.guesses
            if len(guesses) != n_guesses:
                raise AttributeError\
                ('Length of guesses does not match number of hyperparameters')

        if self.regr:
            hypers_list = ['var_n'] + self.kern.hypers
            Hypers = namedtuple('Hypers', hypers_list)
        else:
            Hypers = namedtuple('Hypers', self.kern.hypers)

        bounds = [(1e-5,None) for _ in guesses]
        minimize_res = minimize(self.objective,
                                (guesses),
                                bounds=bounds,
                                method='L-BFGS-B')

        self.hypers = Hypers._make(minimize_res['x'])


        if self.regr:
            self.K = self.kern.make_K(hypers=self.hypers[1:])
            self.Ky = self.K+self.hypers.var_n*np.identity(len(self.X_seqs))
            self.L = np.linalg.cholesky(self.Ky)
            self.alpha = np.linalg.lstsq(self.L.T,
                                         np.linalg.lstsq (self.L,
                                                          np.matrix(self.normed_Y).T)[0])[0]
            self.ML = self.log_ML(self.hypers)
            self.log_p = self.LOO_log_p(self.hypers)
        else:
            self.f_hat = self.find_F(hypers=self.hypers)
            self.W = self.hess (self.f_hat)
            self.W_root = scipy.linalg.sqrtm(self.W)
            self.Ky = np.matrix(self.kern.make_K(hypers=self.hypers))
            self.L = np.linalg.cholesky (np.matrix(np.eye(self.l))+self.W_root\
                                         *self.Ky*self.W_root)
            self.grad = np.matrix(np.diag(self.grad_log_logistic_likelihood\
                                          (self.Y,
                                           self.f_hat)))
            self.ML = self.log_ML(self.hypers)


    def normalize(self, data):
        """
        Normalizes the elements in data by subtracting the mean and dividing
        by the standard deviation.

        Parameters:
            data (pd.Series)

        Returns:
            mean, standard_deviation, normed
        """
        m = data.mean()
        s = data.std()
        return m, s, (data-m) / s

    def unnormalize(self, normed):
        """
        Inverse of normalize, but works on single values or arrays.

        Parameters:
            normed

        Returns:
            normed*self.std * self.mean
        """
        return normed*self.std + self.mean

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
            E = self.unnormalize(k*self.alpha)
            v = np.linalg.lstsq(self.L,k.T)[0]
            var = (k_star - v.T*v) * self.std**2
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
            return (pi_star,)

    def p_integral (self, z, mean, variance):
        '''Equation to integrate when calculating pi_star for classification.
        Equation 3.25 from RW with a sigmoid likelihood.

        Parameters:
            z (float): value at which to evaluate the function.
            mean (float): mean of the Gaussian
            variance (float): variance of the Gaussian

        Returns:
            res (float)
        '''
        try:
            first = 1./(1+math.exp(-z))
        except OverflowError:
            first = 0.
        second = 1/math.sqrt(2*math.pi*variance)
        third = math.exp(-(z-mean)**2/(2*variance))
        return first*second*third

    def log_ML (self, hypers):
        """ Returns the negative log marginal likelihood for the model.
        Uses RW Equation 5.8 for regression models and Equation 3.32 for
        classification models.

        Parameters:
            hypers (iterable): the hyperparameters

        Returns:
            log_ML (float)
        """
        if self.regr:
            Y_mat = np.matrix(self.normed_Y)
            K = self.kern.make_K(hypers=hypers[1::])
            K_mat = np.matrix (K)
            Ky = K_mat + np.identity(len(K_mat))*hypers[0]
            try:
                L = np.linalg.cholesky (Ky)
            except:
                print hypers
                exit('')
            alpha = np.linalg.lstsq(L.T,np.linalg.lstsq (L, np.matrix(Y_mat).T)[0])[0]
            first = 0.5*Y_mat*alpha
            second = sum([math.log(l) for l in np.diag(L)])
            third = len(K_mat)/2.*math.log(2*math.pi)
            ML = (first+second+third).item()
            # log[det(Ky)] = 2*sum(log(diag(L))) is a property
            # of the Cholesky decomposition
            # Y.T*Ky^-1*Y = L.T\(L\Y.T) (another property of the Cholesky)
            if np.isnan(ML):
                print "nan!"
                print hypers
                print Ky, L, alpha, Y_mat
            return ML
        else:
            f_hat = self.find_F(hypers=hypers) # use Algorithm 3.1 to find mode
            ML = self.logq(f_hat, hypers=hypers)
            return ML.item()

    def predicts (self, new_seqs, delete=True):
        """ Calculates predicted (mean, variance) for each sequence in new_seqs

        Uses Equations 2.23 and 2.24 of RW
        Parameters:
            new_seqs (DataFrame): sequences to predict. They must have unique indices.

         Returns:
            predictions (list): (E,v) as floats
        """
        predictions = []
        self.kern.train(new_seqs)
        if self.regr:
            h = self.hypers[1::]
        else:
            h = self.hypers
        for ns in new_seqs.index:
            k = np.matrix([self.kern.calc_kernel(ns, seq1,
                                                hypers=h) \
                           for seq1 in self.X_seqs.index])
            k_star = self.kern.calc_kernel(ns, ns,
                                           hypers=h)
            if self.regr:
                k_star += self.hypers.var_n
            predictions.append(self.predict(k, k_star))
        if delete:
            inds = list(set(new_seqs.index) - set(self.X_seqs.index))
            self.kern.delete(new_seqs.loc[inds, :])
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
            W (np.matrix): diagonal negative Hessian of the log likelihood matrix
        """
        W = np.matrix(np.zeros([self.l,self.l]))
        for i in range(self.l):
            pi_i = self.logistic_likelihood(1., F[i])
            W[i,i] = pi_i*(1-pi_i)
        return W

    def find_F (self, hypers, guess=None, threshold=.0001, evals=1000):
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


        K = self.kern.make_K(hypers=hypers)
        K_mat = np.matrix(K)
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


    def logq(self, F, hypers):
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
        K = self.kern.make_K(hypers=hypers)
        K_mat = np.matrix(K)
        W = self.hess (F)
        W_root = scipy.linalg.sqrtm(W)
        F_mat = np.matrix (F)
        L = np.linalg.cholesky (np.matrix(np.eye(l))+W_root*K_mat*W_root)
        b = W*F_mat.T + np.matrix(np.diag(self.grad_log_logistic_likelihood (self.Y,F))).T
        a = b - W_root*np.linalg.lstsq(L.T,np.linalg.lstsq(L,W_root*K_mat*b)[0])[0]
        logq = 0.5*a.T*F_mat.T - self.log_logistic_likelihood(self.Y, F) \
        + sum(np.log(np.diag(L)))
        return logq

    def LOO_log_p (self, hypers):
        """
        Calculates the negative LOO log predictive probability
        For now, only for regression
        Equation 5.10 and 5.11 from RW
        Parameters:
            variances (iterable)
        Returns:
            log_p
        """
        LOO = self.LOO_res(hypers)
        vs = LOO['v']
        mus = LOO['mu']
        log_ps = -0.5*np.log(vs) - (self.normed_Y-mus)**2 / 2 / vs - 0.5*np.log(2*np.pi)
        return_me = -sum (log_ps)
        return return_me

    def LOO_MSE (self, hypers):
        LOO = self.LOO_res(hypers)
        mus = LOO['mu']
        return sum((self.normed_Y - mus)**2) / len(self.normed_Y)

    def LOO_res (self, hypers):
        """
        Calculates the LOO predictions according to Equation 5.12 from RW
        Parameters:
            hypers (iterable)
        Returns:
            res (pandas.DataFrame): columns are 'mu' and 'v'
        """
        K = self.kern.make_K(hypers=hypers[1::])
        Ky = K + hypers[0]*np.identity(len(self.X_seqs))
        K_inv = np.linalg.inv(Ky)
        Y_mat = np.matrix (self.normed_Y)
        mus = np.diag(Y_mat.T - K_inv*Y_mat.T/K_inv)
        vs = np.diag(1/K_inv)
        return pd.DataFrame(zip(mus, vs), index=self.normed_Y.index,
                           columns=['mu', 'v'])

    @staticmethod
    def load(model):
        '''
        Use cPickle to load the saved model.
        '''
        with open(model,'r') as m_file:
            attributes = pickle.load(m_file)
        if all([y==1 or y==-1 for y in attributes['Y']]):
            hypers = [attributes['hypers'][k] for k in attributes['kern'].hypers]
        else:
            hypers = ['var_n']
            hypers = [attributes['hypers'][k] for k in hypers+attributes['kern'].hypers]

        return GPModel(attributes['X_seqs'],
                       attributes['Y'],
                       attributes['kern'],
                       hypers=hypers)

    def dump(self, f):
        '''
        Use cPickle to save a dict containing the model's X, Y, kern, and hypers.
        '''
        save_me = {}
        save_me['X_seqs'] = self.X_seqs
        save_me['Y'] = self.Y
        names = self.hypers._fields
        hypers = {n:h for n,h in zip(names, self.hypers)}
        save_me['hypers'] = hypers
        save_me['kern'] = self.kern
        with open(f, 'wb') as f:
            pickle.dump(save_me, f)

