''' Classes for doing Gaussian process models of proteins.'''

from collections import namedtuple
import pickle
import abc

import numpy as np
from scipy.optimize import minimize
from scipy import stats, integrate, linalg
from scipy.special import expit
import pandas as pd
from sklearn import linear_model
from sklearn import metrics

from gpmodel import gpmean
from gpmodel import gpkernel
from gpmodel import gptools
from gpmodel import chimera_tools


class BaseGPModel(abc.ABC):

    """ Base class for Gaussian process models. """

    @abc.abstractmethod
    def __init__(self, kernel):
        self.kernel = kernel

    @abc.abstractmethod
    def predict(self, X):
        return

    @abc.abstractmethod
    def fit(self, X, Y):
        return

    def _set_params(self, **kwargs):
        ''' Sets parameters for the model.

        This function can be used to set the value of any or all
        attributes for the model. However, it does not necessarily
        update dependencies, so use with caution.
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def load(cls, model):
        ''' Load a saved model.

        Use pickle to load the saved model.

        Parameters:
            model (string): path to saved model
        '''
        with open(model, 'rb') as m_file:
            attributes = pickle.load(m_file, encoding='latin1')
        model = cls(attributes['kernel'])
        del attributes['kernel']
        if attributes['objective'] == 'LOO_log_p':
            model.objective = model._LOO_log_p
        else:
            model.objective = model._log_ML
        del attributes['objective']
        model._set_params(**attributes)
        return model

    def dump(self, f):
        ''' Save the model.

        Use pickle to save a dict containing the model's
        attributes.

        Parameters:
            f (string): path to where model should be saved
        '''
        save_me = {k: self.__dict__[k] for k in list(self.__dict__.keys())}
        if self.objective == self._log_ML:
            save_me['objective'] = 'log_ML'
        else:
            save_me['objective'] = 'LOO_log_p'
        save_me['guesses'] = self.guesses
        try:
            save_me['hypers'] = list(self.hypers)
            # names = self.hypers._fields
            # hypers = {n: h for n, h in zip(names, self.hypers)}
            # save_me['hypers'] = hypers
        except AttributeError:
            pass
        with open(f, 'wb') as f:
            pickle.dump(save_me, f)


class GPRegressor(BaseGPModel):

    """ A Gaussian process regression model for proteins. """

    def __init__(self, kernel, **kwargs):
        BaseGPModel.__init__(self, kernel)
        self.guesses = None
        if 'objective' not in list(kwargs.keys()):
            kwargs['objective'] = 'log_ML'
        if 'mean_func' not in list(kwargs.keys()):
            self.mean_func = gpmean.GPMean()
        self.variances = None
        self._set_objective(kwargs['objective'])
        del kwargs['objective']
        self._set_params(**kwargs)

    def _set_objective(self, objective):
        """ Set objective function for model. """
        if objective is not None:
            if objective == 'log_ML':
                self.objective = self._log_ML
            else:
                raise AttributeError(objective + ' is not a valid objective')
        else:
            self.objective = self._log_ML

    def fit(self, X, Y, variances=None, bounds=None):
        ''' Fit the model to the given data.

        Set the hyperparameters by training on the given data.
        Update all dependent values.

        Measurement variances can be given, or
        a global measurement variance will be estimated.

        Parameters:
            X (np.ndarray): n x d
            Y (np.ndarray): n.
            variances (np.ndarray): n. Optional.
        '''
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.Series):
            Y = Y.values
        self.X = X
        self.Y = Y
        self._ell = len(Y)
        self._n_hypers = self.kernel.fit(X)
        self.mean, self.std, self.normed_Y = self._normalize(self.Y)
        self.mean_func.fit(X, self.normed_Y)
        self.normed_Y -= self.mean_func.mean(X).T[0]
        if variances is not None:
            if not len(variances) != len(Y):
                raise ValueError('len(variances must match len(Y))')
            self.variances = variances / self.std**2
        else:
            self.variances = None
            self._n_hypers += 1
        if self.guesses is None:
            guesses = [0.9 for _ in range(self._n_hypers)]
        else:
            guesses = self.guesses
            if len(guesses) != self._n_hypers:
                raise AttributeError(('Length of guesses does not match '
                                      'number of hyperparameters'))
        if bounds is None:
            bounds = [(1e-5, None) for _ in guesses]
        minimize_res = minimize(self.objective,
                                guesses,
                                method='L-BFGS-B',
                                bounds=bounds)
        self.hypers = minimize_res['x']


    def _make_Ks(self, hypers):
        """ Make covariance matrix (K) and noisy covariance matrix (Ky)."""
        if self.variances is not None:
            K = self.kernel.cov(hypers=hypers)
            Ky = K + np.diag(self.variances)
        else:
            K = self.kernel.cov(hypers=hypers[1::])
            Ky = K + np.identity(len(K)) * hypers[0]
        return K, Ky

    def _normalize(self, data):
        """ Normalize the given data.

        Normalizes the elements in data by subtracting the mean and
        dividing by the standard deviation.

        Parameters:
            data (pd.Series)

        Returns:
            mean, standard_deviation, normed
        """
        m = data.mean()
        s = data.std()
        return m, s, (data-m) / s

    def unnormalize(self, normed):
        """ Inverse of _normalize, but works on single values or arrays.

        Parameters:
            normed

        Returns:
            normed*self.std * self.mean
        """
        return normed*self.std + self.mean

    def predict(self, X):
        """ Make predictions for each sequence in new_seqs.

        Predictions are scaled as the original outputs (not normalized)

        Uses Equations 2.23 and 2.24 of RW
        Parameters:
            new_seqs (pd.DataFrame or np.ndarray): sequences to predict.

         Returns:
            means, cov as np.ndarrays. means.shape is (n,), cov.shape is (n,n)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        h = self.hypers[1::]
        k_star = self.kernel.cov(X, self.X, hypers=h)
        k_star_star = self.kernel.cov(X, X, hypers=h)
        E = k_star @ self._alpha
        v = linalg.solve_triangular(self._L, k_star.T, lower=True)
        var = k_star_star - v.T @ v
        E += self.mean_func.mean(X)
        E = self.unnormalize(E)
        E = E[:, 0]
        var *= self.std ** 2
        return E, var

    def _log_ML(self, hypers):
        """ Returns the negative log marginal likelihood for the model.

        Uses RW Equation 5.8.

        Parameters:
            log_hypers (iterable): the hyperparameters

        Returns:
            log_ML (float)
        """
        self._K, self._Ky = self._make_Ks(hypers)
        self._L = np.linalg.cholesky(self._Ky)
        self._alpha = linalg.solve_triangular(self._L, self.normed_Y, lower=True)
        self._alpha = linalg.solve_triangular(self._L.T, self._alpha,
                                              lower=False)
        self._alpha = np.expand_dims(self._alpha, 1)

        first = 0.5 * np.dot(self.normed_Y, self._alpha)
        second = np.sum(np.log(np.diag(self._L)))
        third = len(self._K) / 2. * np.log(2 * np.pi)
        self.ML = (first + second + third).item()
        return self.ML


class GPClassifier(BaseGPModel):

    """ A Gaussian process classification model for proteins. """

    def __init__(self, kernel, **kwargs):
        BaseGPModel.__init__(self, kernel)
        self.guesses = None
        self._set_params(**kwargs)
        self.objective = self._log_ML

    def fit(self, X, Y, bounds=None):
        ''' Fit the model to the given data.

        Set the hyperparameters by training on the given data.
        Update all dependent values.

        Parameters:
            X (np.ndarray): Sequences in training set
            Y (np.ndarray): measurements in training set
        '''
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.Series):
            Y = Y.values
        self.X = X
        self.Y = Y
        self._ell = len(Y)
        self._n_hypers = self.kernel.fit(X)
        if self.guesses is None:
            guesses = [0.9 for _ in range(self._n_hypers)]
        else:
            guesses = self.guesses
            if len(guesses) != self._n_hypers:
                raise AttributeError(('Length of guesses does not match '
                                      'number of hyperparameters'))
        if bounds is None:
            bounds = [(1e-5, None) for _ in guesses]
        minimize_res = minimize(self.objective,
                                guesses,
                                method='L-BFGS-B',
                                bounds=bounds)
        self.hypers = minimize_res['x']

    def predict(self, X):
        """ Make predictions for each input in X.

        Uses Algorithm 3.2 of RW
        Parameters:
            X (np.ndarray): inputs to predict

         Returns:
            pi_star, f_bar, var as np.ndarrays
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        predictions = []
        k_star = self.kernel.cov(X, self.X, hypers=self.hypers)
        k_star_star = self.kernel.cov(X, X, hypers=self.hypers)
        f_bar = np.dot(k_star, self._grad)
        Wk = np.expand_dims(self._W_root, 1) * k_star.T
        v = linalg.solve_triangular(self._L, Wk, lower=True)
        var = k_star_star - np.dot(v.T, v)
        span = 20
        pi_star = np.zeros(len(X))
        for i, preds in enumerate(zip(f_bar, np.diag(var))):
            f, va = preds
            pi_star[i] = integrate.quad(self._p_integral,
                                        -span * va + f,
                                        span * va + f,
                                        args=(f, va))[0]
        return pi_star.flatten(), f_bar.flatten(), var

    def _p_integral(self, z, mean, variance):
        ''' Equation 3.25 from RW with a sigmoid likelihood.

        Equation to integrate when calculating pi_star for classification.

        Parameters:
            z (float): value at which to evaluate the function.
            mean (float): mean of the Gaussian
            variance (float): variance of the Gaussian

        Returns:
            res (float)
        '''
        try:
            first = 1./(1+np.exp(-z))
        except OverflowError:
            first = 0.
        second = 1 / np.sqrt(2 * np.pi * variance)
        third = np.exp(-(z-mean) ** 2 / (2*variance))
        return first*second*third

    def _log_ML(self, hypers):
        """ Returns the negative log marginal likelihood for the model.

        Uses RW Equation 3.32 and Algorithm 3.1.

        Parameters:
            log_hypers (iterable): the log hyperparameters

        Returns:
            log_ML (float)
        """
        ell = len(self.Y)
        self._f_hat = np.zeros(ell)
        self._K = self.kernel.cov(hypers=hypers)
        evals = 1000
        threshold = 1e-15
        for i in range(evals):
            pi = expit(self._f_hat)
            # Line 4
            W = pi * (1 - pi)
            # Line 5
            self._W_root = np.sqrt(W)
            W_sr_K = self._W_root[:, np.newaxis] * self._K
            B = np.eye(W.shape[0]) + W_sr_K * self._W_root
            self._L = np.linalg.cholesky(B)
            # Line 6
            self._grad = (self.Y + 1) / 2 - pi
            b = W * self._f_hat + self._grad
            # Line 7
            self._a = b - self._W_root * linalg.cho_solve((self._L, True),
                                                          W_sr_K.dot(b))
            # Line 8
            f_new = self._K.dot(self._a)
            sq_error = np.sum((self._f_hat - f_new) ** 2)
            self._f_hat = f_new
            if sq_error / abs(np.sum(f_new)) < threshold:
                break
        else:
            raise RuntimeError('Maximum evaluations reached without convergence.')
        _logq = 0.5 * self._a.T @ np.expand_dims(self._f_hat, 1)
        _logq -= np.sum(np.log(1.0 / (1 + np.exp(-self.Y * self._f_hat))))
        _logq += np.sum(np.log(np.diag(self._L)))
        self.ML = _logq
        return self.ML

    def _find_F(self, hypers, guess=None, threshold=1e-15, evals=1000):
        """Calculates f_hat according to Algorithm 3.1 in RW.

        Returns:
            f_hat (np.ndarray)
        """
        ell = len(self.Y)
        if guess is None:
            f_hat = np.zeros(ell)
        elif len(guess) == l:
            f_hat = guess
        else:
            raise ValueError('Initial guess must have same dimensions as Y')
        self._K = self.kernel.cov(hypers=hypers)
        n_below = 0
        for i in range(evals):
            pi = expit(f_hat)
            W = pi * (1 - pi)
            # Line 5
            W_sr = np.sqrt(W)
            W_sr_K = W_sr[:, np.newaxis] * self._K
            B = np.eye(W.shape[0]) + W_sr_K * W_sr
            L = np.linalg.cholesky(B)
            # Line 6
            b = W * f_hat + (self.Y + 1) / 2 - pi
            # Line 7
            a = b - W_sr * linalg.cho_solve((L, True), W_sr_K.dot(b))
            # Line 8
            f_new = self._K.dot(a)
            sq_error = np.sum((f_hat - f_new) ** 2)
            if sq_error / abs(np.sum(f_new)) < threshold:
                self._a = a
                self._L = L
                self._W_root = W_sr
                self._grad = (self.Y + 1) / 2 - pi
                return f_new
            f_hat = f_new
        raise RuntimeError('Maximum evaluations reached without convergence.')


class GPMultiClassifier(BaseGPModel):

    """ A GP multi-class classifier. """

    def __init__(self, kernels, **kwargs):
        self.kernels = kernels
        self.guesses = None
        self._set_params(**kwargs)
        self.objective = self._log_ML

    def score(self, X, Y):
        ''' Score the model on the given points.

        Predicts Y for the sequences in X, then scores the predictions.

        Parameters:
            X (np.ndarray) n x d
            Y (np.ndarray) n x c

        Returns:
            res (dict): 'acc' and 'log_loss'
        '''
        if isinstance(Y, pd.Series):
            Y = Y.values
        pi_star, _, _ = self.predict(X)
        scores = {}
        scores['acc'] = self._accuracy(Y, pi_star)
        scores['log_loss'] = self._log_loss(Y, pi_star)
        return scores

    def _log_loss(self, Y, pi_star):
        """ Calculate the negative log loss. """
        n = len(Y)
        return - np.sum(Y * np.log(pi_star)) / n

    def _accuracy(self, Y, pi_star):
        """ Calculate the fraction correctly predicted. """
        n = len(Y)
        p = np.zeros_like(Y)
        p[range(n), pi_star.argmax(1)] = 1
        return np.sum(p * Y) / n

    def predict(self, X):
        """ Make predictions for each input in X.

        Uses Algorithm 3.4 of RW
        Parameters:
            X (np.ndarray): inputs to predict.

         Returns:
            pi_star (np.ndarray): predictive test probabilities. n x c
            mu (np.ndarray): latent test mean. n x c
            sigma (np.ndarray): latent test covariance. n x c x c
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        P = self._softmax(self._f_hat)
        N, C = self.Y.shape
        P_vector = P.T.reshape((N * C, 1))
        PI = self._stack(P)
        M = np.linalg.cholesky(np.sum(self._E, axis=2))
        hypers = self._split_hypers(self.hypers)
        mu = np.ones((len(X), C))
        k_star = [k.cov(self.X, X, h) for k, h in zip(self.kernels, hypers)]
        sigma = np.zeros((len(X), C, C))
        k_star_star = [np.diag(k.cov(X, X, h))
                       for k, h in zip(self.kernels, hypers)]
        for i in range(C):
            mu[:, i] = (self.Y[:, i] - P[:, i]).reshape((1, N)) @ k_star[i]
            Ec = self._E[:, :, i]
            b = Ec @ k_star[i]
            first = np.linalg.lstsq(M, b)[0]
            c = Ec @ np.linalg.lstsq(M.T, first)[0]
            for j in range(C):
                sigma[:, i, j] = np.sum(c * k_star[j], axis=0)
                if i == j:
                    sigma[:, i, j] += k_star_star[i] - np.sum(b * k_star[i],
                                                              axis=0)
        S = 5000
        pi_star = np.zeros((X.shape[0], C))
        for _ in range(S):
            for i in range(len(X)):
                f = np.random.multivariate_normal(mu[i], sigma[i])
                f = np.exp(f)
                pi_star[i] += f / np.sum(f)
        pi_star /= S
        return pi_star, mu, sigma

    def fit(self, X, Y):
        ''' Fit the model to the given data.

        Set the hyperparameters by training on the given data.
        Update all dependent values.

        Parameters:
            X (np.ndarray): Sequences in training set
            Y (np.ndarray): measurements in training set
        '''
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.Series):
            Y = Y.values
        self.X = X
        self.Y = Y
        self._n_hypers = [k.fit(X) for k in self.kernels]
        if self.guesses is None:
            guesses = [0.9 for _ in range(sum(self._n_hypers))]
        else:
            guesses = self.guesses
            if len(guesses) != sum(self._n_hypers):
                raise AttributeError(('Length of guesses does not match '
                                      'number of hyperparameters'))
        bounds = [(1e-5, 50) for _ in guesses]
        minimize_res = minimize(self.objective,
                                (guesses),
                                bounds=bounds,
                                method='L-BFGS-B')
        self.hypers = minimize_res['x']
        return

    def _log_ML(self, hypers):
        """ Returns the negative log marginal likelihood for the model.

        Uses RW Equation 3.44.

        Parameters:
            hypers (iterable): the hyperparameters

        Returns:
            log_ML (float)
        """
        self._f_hat = self._find_F(hypers)
        n_samples, n_classes = self.Y.shape
        Y_vector = (self.Y.T).reshape((n_samples * n_classes, 1))
        f_hat_vector = (self._f_hat.T).reshape((n_samples * n_classes, 1))
        P = self._softmax(self._f_hat)
        P_vector = P.T.reshape((n_samples * n_classes, 1))
        PI = self._stack(P)
        K_expanded = self._expand(self._K)
        self._E = np.zeros((n_samples, n_samples, n_classes))
        self._L = np.zeros((n_samples, n_samples, n_classes))
        for i in range(n_classes):
            Dc_root = np.sqrt(np.diag(P[:, i]))
            DKD = Dc_root @ self._K[:, :, i] @ Dc_root
            self._L[:, :, i] = np.linalg.cholesky(np.eye(n_samples) + DKD)
            first = np.linalg.lstsq(self._L[:, :, i], Dc_root)[0]
            self._E[:, :, i] = Dc_root @ \
                np.linalg.lstsq(self._L[:, :, i].T, first)[0]
        self._M = np.linalg.cholesky(np.sum(self._E, axis=2))
        D = np.diag(P_vector[:, 0])
        b = (D - PI @ PI.T) @ f_hat_vector + Y_vector - P_vector
        E_expanded = self._expand(self._E)
        c = E_expanded @ K_expanded @ b
        self._a = b - c
        first, _, _, _ = np.linalg.lstsq(self._M, self._R.T @ c)
        self._a += E_expanded @ self._R @ \
            np.linalg.lstsq(self._M.T, first)[0]
        first = 0.5 * self._a.T @ f_hat_vector
        second = -Y_vector.T @ f_hat_vector
        third = np.sum(np.log(np.sum(np.exp(self._f_hat), axis=1)))
        fourth = np.sum([np.sum(np.log(np.diag(self._L[:, :, i])))
                         for i in range(n_classes)])
        self.ML = (first + second + third + fourth).item()
        return self.ML

    def _find_F(self, hypers, guess=None, threshold=1e-3, evals=1000):
        """Calculates f_hat according to Algorithm 3.3 in RW.

        Returns:
            f_hat (np.ndarray): (n_samples x n_classes)
        """
        n_samples, n_classes = self.Y.shape
        Y_vector = (self.Y.T).reshape((n_samples * n_classes, 1))
        if guess is None:
            f_hat = np.zeros_like(self.Y)
        else:
            f_hat = guess
            if guess.shape != self.Y.shape:
                raise ValueError('guess must have same dimensions as Y')
        f_vector = (f_hat.T).reshape((n_samples * n_classes, 1))
        # K[:,:,i] is cov for ith class
        self._K = self._make_K(hypers=hypers)
        # Block diagonal K
        K_expanded = self._expand(self._K)
        self._R = np.concatenate([np.eye(n_samples) for _ in range(n_classes)],
                                 axis=0)
        n_below = 0
        for k in range(evals):
            P = self._softmax(f_hat)
            P_vector = P.T.reshape((n_samples * n_classes, 1))
            PI = self._stack(P)
            E = np.zeros((n_samples, n_samples, n_classes))
            L = np.zeros((n_samples, n_samples, n_classes))
            for i in range(n_classes):
                Dc_root = np.sqrt(np.diag(P[:, i]))
                DKD = Dc_root @ self._K[:, :, i] @ Dc_root
                L[:, :, i] = np.linalg.cholesky(np.eye(n_samples) + DKD)
                first = np.linalg.lstsq(L[:, :, i], Dc_root)[0]
                E[:, :, i] = Dc_root @ \
                    np.linalg.lstsq(L[:, :, i].T, first)[0]
            M = np.linalg.cholesky(np.sum(E, axis=2))
            D = np.diag(P_vector[:, 0])
            b = (D - PI @ PI.T) @ f_vector + Y_vector - P_vector
            E_expanded = self._expand(E)
            c = E_expanded @ K_expanded @ b
            a = b - c
            first, _, _, _ = np.linalg.lstsq(M, self._R.T @ c)
            a += E_expanded @ self._R @ \
                np.linalg.lstsq(M.T, first)[0]
            f_vector_new = K_expanded @ a
            sq_error = np.sum((f_vector - f_vector_new) ** 2)
            f_vector = f_vector_new
            f_hat = f_vector_new.reshape((n_classes, n_samples)).T
            if sq_error / abs(np.sum(f_vector_new)) < threshold:
                n_below += 1
            else:
                n_below = 0
            if n_below > 9:
                break
        return f_hat

    def _expand(self, A):
        """ Expand n x m x c matrix to nm x nc block diagonal matrix. """
        n, m, c = A.shape
        expanded = np.zeros((n*c, m*c))
        for i in range(c):
            expanded[i*n:(i+1)*n, i*m:(i+1)*m] = A[:, :, i]
        return expanded

    def _make_K(self, hypers):
        """ Make the covariance matrix for the training inputs. """
        hypers = self._split_hypers(hypers)
        Ks = np.stack([k.cov(hypers=h) for k, h in zip(self.kernels, hypers)],
                      axis=2)
        return Ks

    def _split_hypers(self, hypers):
        """ Split the hypers for each kernel. """
        inds = np.cumsum(self._n_hypers)
        inds = np.insert(inds, 0, 0)
        return [hypers[inds[i]:inds[i+1]] for i in range(len(inds) - 1)]

    def _softmax(self, f):
        """ Calculate softmaxed probabilities.

        Parameters:
            f (np.ndarray): n_samples x n_classes

        Returns:
            p (np.ndarray): n_samples x n_classes
        """
        P = np.exp(f)
        return P / np.sum(P, axis=1).reshape((P.shape[0], 1))

    def _stack(self, P):
        """ Stack diagonal probability matrices.

        Parameters:
            P(np.ndarray): n_samples x n_classes

        Returns:
            PI(np.ndarray): (n_samples * n_classes) x n_classes
        """
        return np.concatenate([np.diag(p) for p in P.T], axis=0)


class LassoGPRegressor(GPRegressor):

    """ Extends GPRegressor with L1 regression for feature selection. """

    def __init__(self, kernel, **kwargs):
        self._gamma_0 = kwargs.get('gamma', 0)
        self._clf = linear_model.Lasso(alpha=np.exp(self._gamma_0),
                                       warm_start=False,
                                       max_iter=100000)
        GPRegressor.__init__(self, kernel, **kwargs)

    def predict(self, X):
        X, _ = self._regularize(X, mask=self._mask)
        return GPRegressor.predict(self, X)

    def fit(self, X, y, variances=None):
        minimize_res = minimize(self._log_ML_from_gamma,
                                self._gamma_0,
                                args=(X, y, variances),
                                method='Powell',
                                options={'xtol': 1e-8, 'ftol': 1e-8})
        self.gamma = minimize_res['x']

    def _log_ML_from_gamma(self, gamma, X, y, variances=None):
        X, self._mask = self._regularize(X, gamma=gamma, y=y)
        GPRegressor.fit(self, X, y, variances=variances)
        return self.ML

    def _regularize(self, X, **kwargs):
        """ Perform feature selection on X.

        Features can be selected by providing y and gamma, in which
        case L1 linear regression is used to determine which columns
        of X are kept. Or, if a Boolean mask of length equal to the
        number of columns in X is provided, features will be selected
        using the mask.

        Parameters:
            X (pd.DataFrame)

        Optional keyward parameters:
            gamma (float): log amount of regularization
            y (np.ndarray or pd.Series)
            mask (iterable)

        Returns:
            X (pd.DataFrame)
            mask (np.ndarray)
        """
        gamma = kwargs.get('gamma', None)
        y = kwargs.get('y', None)
        mask = kwargs.get('mask', None)
        if gamma is not None:
            if y is None:
                raise ValueError("Missing argument 'y'.")
            self._clf.alpha = np.exp(gamma)
            self._clf.fit(X, y)
            weights = pd.DataFrame()
            weights['weight'] = self._clf.coef_
            mask = ~np.isclose(weights['weight'], 0.0)
        X = X.transpose()[mask].transpose()
        if isinstance(X, pd.DataFrame):
            X.columns = list(range(np.shape(X)[1]))
        return X, mask
