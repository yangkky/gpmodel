"""Kernel functions that calculate the covariance between two inputs."""

import numpy as np
import pandas as pd
from sys import exit


class GPKernel (object):

    """ A Gaussian Process kernel.

       Attributes:
           hypers (list)
           _saved_X (dict)
    """

    def __init__ (self):
        """ Create a GPKernel. """
        self._saved_X = {}
        self.hypers = []

    def set_X(self, X):
        """ Set a default set of inputs X.

        Parameters:
            X (pd.DataFrame)
        """
        self.train(X)

    def train(self, X):
        """ Remember the inputs in X.

        Parameters:
            X (pd.DataFrame): Saves the inputs in X to a dictionary
                using the index as the keys.
        """
        for i in range(len(X.index)):
            if X.index[i] in self._saved_X.keys():
                pass
            else:
                self._saved_X[X.index[i]] = X.iloc[i]

    def delete(self, X):
        """ Forget the inputs in X.

        Parameters:
            X (pd.DataFrame): Deletes the inputs in X
        """
        for i in range(len(X.index)):
            if X.index[i] in self._saved_X.keys():
                del self._saved_X[X.index[i]]

    def _get_X(self, x):
        """ Retrieve an input x.

        Parameters:
            x: x can be the key used to remember x or the actual values.

        Returns:
            x
        """
        try:
            return self._saved_X[x]
        except (KeyError, AttributeError, TypeError):
            return x

class PolynomialKernel(GPKernel):

    """ A Polynomial kernel of the form (s0 + sp * x.T*x)^d

    Attributes:
    hypers (list): names of the hyperparameters required
    _saved_X (dict): dict of saved index:X pairs
    _dots (np.ndarray): saved default x.T*x
    _deg (integer): degree of polynomial
    """

    def __init__(self, d):
        """ Initiate a polynomial kernel.

        Parameters:
            d (integer): degree of the polynomial
        """
        GPKernel.__init__(self)
        if not isinstance(d, int):
            raise TypeError('d must be an integer.')
        if d < 1:
            raise ValueError('d must be greater than or equal to 1.')
        self.hypers = ['sigma_0', 'sigma_p']
        self._deg = d
        self._dot = np.array([[]])

    def calc_kernel(self, x1, x2, hypers):
        """ Calculate the polynomial kernel between x1 and x2.

        Parameters:
            x1 (np.ndarray or string): either an array
                representing the sequence or the key for that sequence
                if it has been saved.
            x2 (np.ndarray or string)
            hypers (iterable):

        Returns:
            k (float)
        """
        x1 = self._get_X(x1)
        x2 = self._get_X(x2)
        sigma_0, sigma_p = hypers
        try:
            dot = np.dot(x1.T, x2)
        except AttributeError:
            dot = x1 * x2
        return np.power(sigma_0 **2 + sigma_p ** 2 * dot, self._deg)

    def make_K(self, Xs=None, hypers=None):
        """ Calculate the Matern kernel matrix for the points in Xs.

        Parameters:
            Xs (np.ndarray or pd.DataFrame): If none given, uses
                saved values.
            hypers (iterable): the hyperparameters. Default is
                sigma_0 = sigma_p = 1.0.

        Returns:
            K (np.ndarray)
        """
        if hypers is None:
            sigma_0, sigma_p = (1.0, 1.0)
        else:
            sigma_0, sigma_p = hypers
        if Xs is None:
            dots = self._dots
        else:
            dots = self._get_all_dots(Xs)
        return np.power(sigma_0 ** 2 + sigma_p ** 2 * dots, self._deg)

    def set_X(self, X):
        """ Remember a default set of inputs X.

        Extends the method from GPKernel by also remembering an array
        of dot products between the inputs in X.

        Parameters:
            X (np.ndarray or pd.DataFrame)
        """
        self.train(X)
        self._dots = self._get_all_dots(X)

    def _get_all_dots(self, X):
        """ Calculates the dot products between x_i in X.

        Each row of X represents one measurement. Each column represents
        a dimension.

        Parameters:
            X: np.ndarray or np.matrix or pd.Dataframe

        Returns:
            D (np.ndarray or float)
        """
        X = np.array(X)
        dims = np.shape(X)
        n = dims[0]
        d = np.empty((n,n))
        for i in range (n):
            for j in range(i+1):
                d[i][j] = np.dot(X[j], X[i])
                d[j][i] = d[i][j]
        return d

class MaternKernel (GPKernel):

    """ A Matern kernel with nu = 5/2 or 3/2.

    Attributes:
        hypers (list): names of the hyperparameters required
        _saved_X (dict): dict of saved index:X pairs
        nu (string): '3/2' or '5/2'
        _d (np.ndarray): saved default geometric distances
    """


    def __init__(self, nu):
        """ Initiate a Matern kernel.

        Parameters:
            nu (string): '3/2' or '5/2'
        """
        GPKernel.__init__(self)
        if nu not in ['3/2', '5/2']:
            raise ValueError("nu must be '3/2' or '5/2'")
        self.hypers = ['ell']
        self.nu = nu
        self._d = np.array([[]])

    def calc_kernel(self, x1, x2, hypers):
        """ Calculate the Matern kernel between x1 and x2.

        Parameters:
            x1 (np.ndarray or string): either an array
                representing the sequence or the key for that sequence
                if it has been saved.
            x2 (np.ndarray or string)
            hypers (iterable): default is ell=1.0.

        Returns:
            k (float)
        """
        d = self._distance(x1, x2)
        return float(self._matern(d, hypers))

    def make_K(self, Xs=None, hypers=[1.0]):
        """ Calculate the Matern kernel matrix for the points in Xs.

        Parameters:
            Xs (np.ndarray or pd.DataFrame): If none given, uses
                saved values.
            hypers (iterable): the hyperparameters. Default is ell=1.0.

        Returns:
            K (np.ndarray)
        """
        if Xs is None:
            return self._matern(self._d, hypers)
        else:
            d = self._get_d(Xs)
            K = self._matern(d, hypers)
            return K

    def set_X(self, X):
        """ Remember a default set of inputs X.

        Extends the method from GPKernel by also remembering an array
        of distances between the inputs in X.

        Parameters:
            X (np.ndarray or pd.DataFrame)
        """
        self.train(X)
        self._d = self._get_d(X)

    def _matern(self, d, hypers=[1.0]):
        """ Calculate the Matern kernel given the distances d.

        Returns a float if a single distance is given. Otherwise,
        returns a nxn array.

        Parameters:
            d (float or iterable):

        Returns:
            M (float or np.ndarray)
        """
        ell = hypers[0]
        if self.nu == '3/2':
            M =  (1.0 + np.sqrt(3.0) * d / ell) * np.exp(-np.sqrt(3) * d / ell)
        elif self.nu == '5/2':
            first = (1.0 + np.sqrt(5.0)*d/ell) + 5.0*np.power(d, 2)/3.0/ell**2
            second = np.exp(-np.sqrt(5.0) * d / ell)
            M =  first * second
        return M

    def _distance(self, x1, x2):
        """ Calculate the geometric distance between two points.

        The geometric distance between two points is the L2 norm of
        their difference.

        Parameters:
            x1 (np.ndarray or string): either a array
                representing the sequence or the key for that sequence
                if it has been saved.
            seq2 (np.ndarray or string)

        Returns:
            d (float)
        """
        x1 = self._get_X(x1)
        x2 = self._get_X(x2)
        return np.linalg.norm(x1 - x2)

    def _get_d (self, xs):
        """ Calculates the geometric distances between x_i in xs.

        Each row of xs represents one measurement. Each column represents
        a dimension.

        Parameters:
            xs: ndarray or np.matrix or pd.Dataframe

        Returns:
            D (np.ndarray or float)
        """
        xs = np.array(xs)
        A = np.sum(xs ** 2, axis=1).reshape((len(xs), 1))
        B = np.sum(xs ** 2, axis=1).reshape((len(xs), 1)).T
        C = 2 * np.dot(xs, xs.T)
        dists = np.sqrt(A + B - C)
        return dists

class SEKernel (GPKernel):

    """ A squared exponential kernel.

    Attributes:
        hypers (list)
        _d_squared (np.ndarray)
        _saved_X (dict)
    """

    def __init__(self):
        """ Initiate a SEKernel. """
        GPKernel.__init__(self)
        self.hypers = ['sigma_f', 'ell']
        self._d_squared = np.array([[]])

    def calc_kernel (self, x1, x2, hypers=[1.0, 1.0]):
        """ Calculate the squared exponential kernel between x1 and x2.

        Parameters:
            x1 (np.ndarray or string): either an array
                representing the sequence or the key for that sequence
                if it has been saved.
            x2 (np.ndarray or string)
            hypers (iterable): default is ell=1.0.

        Returns:
            k (float)
        """
        d = self._distance(x1, x2)
        return float(self._squared_exponential(d, hypers))

    def set_X(self, X):
        """ Set a default set of inputs X.

        Extends the method from GPKernel by also remembering an array
        of squared distances between the inputs in X.

        Parameters:
            X (iterable)
        """
        self.train(X)
        self._d_squared = self._get_d_squared(X)

    def make_K(self, Xs=None, hypers=[1.0, 1.0]):
        """ Calculate the SE kernel matrix for the points in Xs.

        Parameters:
            Xs (np.ndarray or pd.DataFrame): If none given, uses
                saved values.
            hypers (iterable): the hyperparameters. Default is ell=1.0.

        Returns:
            K (np.ndarray)
        """
        if Xs is None:
            return self._squared_exponential(self._d_squared, hypers)
        else:
            d = self._get_d_squared(Xs)
            K = self._squared_exponential(d, hypers)
            return K

    def _squared_exponential (self, d_squared, params):
        """
        Converts a square matrix of squared distances into a SE covariance
        matrix.

        Parameters:
            d_squared (numpy.ndarray): cov_i_j is the 'distance between the x_i
                and x_j

        Returns:
            se (numpy.ndarray)
        """
        sigma_f, ell = params
        se_array = sigma_f**2 * np.exp(-0.5/np.power(ell,2) * d_squared)
        return se_array

    def _distance(self, x1, x2):
        """ Calculate the squared geometric distance between two points.

        The squared geometric distance between two points is the square of
        the L2 norm of their difference.

        Parameters:
            x1 (np.ndarray or string): either a array
                representing the sequence or the key for that sequence
                if it has been saved.
            seq2 (np.ndarray or string)

        Returns:
            d (float)
        """
        x1 = self._get_X(x1)
        x2 = self._get_X(x2)
        return np.linalg.norm(x1 - x2)**2

    def _get_d_squared (self, xs):
        """
        Calculates the square of the geometric distances between x_i in xs.
        Each row of xs represents one measurement. Each column represents
        a dimension. The exception is if xs only has one row with multiple
        columns, then each column is assumed to be a measurement.

        Parameters:
            xs: ndarray or np.matrix or pd.Dataframe

        Returns:
            D (np.ndarray or float)
        """
        xs = np.array(xs)
        A = np.sum(xs ** 2, axis=1).reshape((len(xs), 1))
        B = np.sum(xs ** 2, axis=1).reshape((len(xs), 1)).T
        C = 2 * np.dot(xs, xs.T)
        dists = A + B - C
        return dists

class HammingKernel (GPKernel):

    """A linear Hamming kernel.

    Attributes:
        _seqs (dict)
        hypers (list)
        _base_K (DataFrame)
    """

    def __init__ (self):
        """ Initiate a Hamming kernel."""
        super(HammingKernel,self).__init__()
        self.hypers = ['var_p']

    def calc_kernel (self, seq1, seq2, hypers=[1.0], normalize=True):
        """ Calculates the Hamming kernel between two sequences.

        The Hamming kernel is the number (if not normalized) or the
        the fraction (if normalized) of shared residues between two
        sequences, multiplied by a scale factor (var_p).

        Parameters:
            seq1 (pd.DataFrame or string): either a DataFrame
                representing the sequence or the key for that sequence
                if it has been saved.
            seq2 (pd.DataFrame or string)
            hypers (interable): var_p. Default is 1.0.
            normalize (Boolean): whether to divide by the length of
                sequences. Default is true.

        Returns:
            k (float)
        """
        var_p = hypers[0]
        s1 = self._get_X(seq1)
        s2 = self._get_X(seq2)
        k = sum([1 if str(a) == str(b) else 0 for a,b in zip(s1, s2)])
        if normalize:
            k = float(k) / len(s1)
        return k*var_p

    def make_K (self, seqs=None, hypers=[1.0], normalize=True):
        """ Calculate the Hamming kernel matrix.

        Parameters:
            seqs (np.ndarray or pd.DataFrame): If none given, uses
                saved values.
            hypers (iterable): the hyperparameters. Default is
                sigma_p=1.0.
            normalize (boolean): default is True

        Returns:
            K (np.ndarray)
        """
        if seqs is None:
            var_p = hypers[0]
            return self._base_K*var_p
        n_seqs = len (seqs)
        K = np.zeros((n_seqs, n_seqs))
        for n1,i in zip(range(n_seqs), seqs.index):
            for n2,j in zip(range (n_seqs), seqs.index):
                seq1 = seqs.iloc[n1]
                seq2 = seqs.iloc[n2]
                K[n1,n2] = self.calc_kernel (seq1, seq2,
                                             hypers=hypers,
                                             normalize=normalize)
        return np.array(K)

    def set_X (self, X_seqs):
        """ Set a default set of inputs X_seqs.

        Extends the method from GPKernel by also remembering the
        Hamming matrix when var_p = 1.

        Parameters:
            X_seqs (iterable)
        """
        self.train (X_seqs)
        self._base_K = self.make_K(X_seqs)

    def train(self, X_seqs):
        """ Remember the inputs in X_seqs.

        Parameters:
            X_seqs (pd.DataFrame): Saves the inputs in X_seqs to a dictionary
                using the index as the keys.
        """
        for i in range(len(X_seqs.index)):
            if X_seqs.index[i] in self._saved_X.keys():
                pass
            else:
                self._saved_X[X_seqs.index[i]] = ''.join(s for
                                                         s in X_seqs.iloc[i])

    def delete(self,X_seqs=None):
        """ Forget the inputs in X_seqs.

        Optional parameters:
            X_seqs (pd.DataFrame): sequences to forget. If none
                provided, forget all.
        """
        if X_seqs is None:
            self._saved_X = {}
        for i in range(len(X_seqs.index)):
            if X_seqs.index[i] in self._saved_X.keys():
                del self._saved_X[X_seqs.index[i]]

    def _get_X(self,seq):
        """ Retrieve an input sequence.

        Parameters:
            seq: seq can be the key used to remember x or the actual values.

        Returns:
            seq
        """
        try:
            return self._saved_X[seq]
        except TypeError:
            return ''.join([s for s in seq])

class WeightedHammingKernel (HammingKernel):

    '''
    A Hamming kernel where the covariance between two sequences
    depends on which amino acids are changed to what. The changes
    are weighted using blosum62.
    '''

    def __init__(self):
        """ Intiate a WeightedHammingKernel. """
        from Bio.SubsMat import MatrixInfo
        self._weights = MatrixInfo.blosum62
        self._weights = {k:self.weights[k] for k in self.weights.keys()}
        for k in self._weights.keys():
            self._weights[(k[1], k[0])] = self.weights[k]
        super(WeightedHammingKernel,self).__init__()

    def make_K(self, seqs=None, hypers=[1.0], normalize=True):
        """ Calculate the weighted Hamming kernel matrix.

        Parameters:
            seqs (np.ndarray or pd.DataFrame): If none given, uses
                saved values.
            hypers (iterable): the hyperparameters. Default is
                sigma_p=1.0.
            normalize (boolean): Default is true

        Returns:
            K (np.ndarray)
        """
        return super(WeightedHammingKernel, self).make_K(seqs,
                                                         hypers, normalize)

    def calc_kernel(self, seq1, seq2, hypers=[1.0], normalize=True):
        """ Calculates the weighted Hamming kernel between two sequences.

        Parameters:
            seq1 (pd.DataFrame)
            seq2 (pd.DataFrame)
            hypers (interable): var_p
            normalize (Boolean): whether to divide by the length of
                sequences

        Returns:
            k (float)
        """
        var_p = hypers[0]
        s1 = self._get_X(seq1)
        s2 = self._get_X(seq2)
        k = 0.0
        for a,b in zip(s1, s2):
            try:
                k += self._weights[(a,b)]
            except KeyError:
                k+= 0
        if normalize:
            k = float(k) / len(s1)
        return k*var_p

class StructureKernel (GPKernel):

    """A Structure kernel

    Attributes:
        hypers (list): list of required hyperparameters
        _saved_X (dict): a dict matching the labels for sequences to their contacts
        contacts (list): list of which residues are in contact
    """

    def __init__ (self, contacts):
        """ Initiate a StructureKernel.

        Parameters:
            contacts: A list of tuples, where each tuple defines two
                positions in contact with each other.
        """
        GPKernel.__init__(self)
        self.contacts = contacts
        self.hypers = ['var_p']

    def make_K (self, seqs=None, hypers=[1.0], normalize=True):
        """ Calculate the structure kernel matrix.

        Parameters:
            seqs (np.ndarray or pd.DataFrame): If none given, uses
                saved values.
            hypers (iterable): the hyperparameters. Default is
                sigma_p=1.0.
            normalize (boolean): Default is true

        Returns:
            K (np.ndarray)
        """
        if seqs is None:
            var_p = hypers[0]
            return self._base_K*var_p
        n_seqs = len (seqs)
        K = np.zeros((n_seqs, n_seqs))
        for n1 in range(n_seqs):
            for n2 in range(n1+1):
                if isinstance(seqs, pd.DataFrame):
                    seq1 = seqs.iloc[n1]
                    seq2 = seqs.iloc[n2]
                else:
                    seq1 = seqs[n1]
                    seq2 = seqs[n2]
                K[n1,n2] = self.calc_kernel(seq1, seq2,
                                            hypers=hypers, normalize=normalize)
                if n1 != n2:
                    K[n2, n1] = K[n1, n2]
        return K

    def calc_kernel (self, seq1, seq2, hypers=[1.0], normalize=True):
        """ Calculate the structure kernel between two sequences.

        The structure kernel is the number (if not normalized) or the
        the fraction (if normalized) of shared contacts between two
        sequences, multiplied by a scale factor (var_p).

        Parameters:
            seq1 (pd.DataFrame or string): either a DataFrame
                representing the sequence or the key for that sequence
                if it has been saved.
            seq2 (pd.DataFrame or string)
            hypers (interable): var_p. Default is 1.0.
            normalize (Boolean): whether to divide by the length of
                sequences. Default is true.

        Returns:
            k (float): number of shared contacts * var_p
        """
        var_p = hypers[0]
        contacts1 = self._get_X(seq1)
        contacts2 = self._get_X(seq2)
        k = len(set(contacts1) & set(contacts2))*var_p
        if normalize:
            k = float(k) / len(contacts1)
        return k

    def set_X (self, X_seqs):
        """ Set a default set of inputs X_seqs.

        Extends the method from GPKernel by also remembering the
        contacts matrix when var_p = 1.

        Parameters:
            X_seqs (iterable)
        """
        self.train (X_seqs)
        self._base_K = self.make_K(X_seqs)


    def train(self, X_seqs):
        """ Remember the inputs in X_seqs.

        Parameters:
            X_seqs (pd.DataFrame): Saves the inputs in X_seqs to a dictionary
                using the index as the keys.
        """
        for i in range(len(X_seqs.index)):
            if X_seqs.index[i] in self._saved_X.keys():
                continue
            else:
                self._saved_X[X_seqs.index[i]] = \
                    self._get_X(X_seqs.iloc[i])

    def delete(self, X_seqs=None):
        """ Forget the inputs in X_seqs.

        Optional parameters:
            X_seqs (pd.DataFrame): sequences to forget. If none
                provided, forget all.
        """
        if X_seqs is None:
            self._saved_X = {}
        for i in range (len(X_seqs.index)):
            if X_seqs.index[i] in self._saved_X.keys():
                del self._saved_X[X_seqs.index[i]]

    def _get_X(self, seq):
        """ Retrieve an input sequence.

        Parameters:
            seq: seq can be the key used to remember x or the actual values.

        Returns:
            seq
        """
        if isinstance (seq, basestring):
            try:
                return self._saved_X[seq]
            except:
                raise ValueError ('Key %s not recognized' %seq)
        try:
            return self._saved_X[seq.name]

        except (KeyError, AttributeError):
            seq = ''.join([s for s in seq])
            contacts = []
            for i, con in enumerate(self.contacts):
                term = ((con[0],seq[con[0]]),(con[1],seq[con[1]]))
                contacts.append(term)
            return contacts

class StructureMaternKernel(MaternKernel, StructureKernel):

    """ A Matern structure kernel with nu = 5/2 or 3/2.

    Attributes:
        hypers (list): names of the hyperparameters required
        nu (string): '3/2' or '5/2'
        _d (np.ndarray): saved default geometric distances
        _saved_X (dict): a dict matching the labels for sequences
            to their contacts
        contacts (list): list of which residues are in contact
    """

    def __init__(self, contacts, nu):
        """ Initiate a StructureMaternKernel.

        Parameters:
            contacts: A list of tuples, where each tuple defines two
                positions in contact with each other.
            nu (string): '3/2' or '5/2'
        """
        StructureKernel.__init__(self, contacts)
        MaternKernel.__init__(self, nu)


    def _distance(self, seq1, seq2):
        """ Calculate the contact distance between two sequences.

        The contact distance between two sequences of identical length
        is the square root number of contacts at which they differ.
        This is the 'geometric distance.'

        Parameters:
            seq1 (pd.DataFrame or string): either a DataFrame
                representing the sequence or the key for that sequence
                if it has been saved.
            seq2 (pd.DataFrame or string)

        Returns:
            d (float)
        """
        k = StructureKernel.calc_kernel(self, seq1, seq2, normalize=False)
        n = len(self._get_X(seq1))
        return np.sqrt(n - k)


    def _get_d(self, X_seqs):
        """ Calculate the contact distances between a set of sequences.

        Dij is the contact distance between Xi and Xj. The geometric
        distance is the square root of the number of contacts at which
        two sequences differ.

        Parameters:
            X_seqs (pd.DataFrame)

        Returns:
            D (np.ndarray)
        """
        n = len(X_seqs)
        if n == 2:
            return self._distance(X_seqs.iloc[0], X_seqs.iloc[1])
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                D[i,j] = self._distance(X_seqs.iloc[i], X_seqs.iloc[j])
                D[j,i] = D[i,j]
        return D

class HammingMaternKernel(MaternKernel, HammingKernel):

    """ A Matern Hamming kernel with nu = 5/2 or 3/2.

    Attributes:
        hypers (list): names of the hyperparameters required
        nu (string): '3/2' or '5/2'
        _d (np.ndarray): saved default geometric distances
        _saved_X (dict): a dict matching the labels for sequences
            to the sequences
    """

    def __init__(self, nu):
        """ Initiate a HammingMaternKernel.

        Parameters:
            nu (string): '3/2' or '5/2'
        """
        HammingKernel.__init__(self)
        MaternKernel.__init__(self, nu)

    def _distance(self, seq1, seq2):
        """ Calculate the Hamming distance between two sequences.

        The Hamming distance between two sequences of identical length
        is the square root number of residues at which they differ.
        This is the 'geometric distance.'

        Parameters:
            seq1 (pd.DataFrame or string): either a DataFrame
                representing the sequence or the key for that sequence
                if it has been saved.
            seq2 (pd.DataFrame or string)

        Returns:
            d (float)
        """
        k = HammingKernel.calc_kernel(self, seq1, seq2, normalize=False)
        n = len(self._get_X(seq1))
        return np.sqrt(n - k)

    def _get_d(self, X_seqs):
        """ Calculate the Hamming distances between a set of sequences.

        Dij is the Hamming distance between Xi and Xj. The geometric
        distance is the square root of the number of residues at which
        two sequences differ.

        Parameters:
            X_seqs (pd.DataFrame)

        Returns:
            D (np.ndarray)
        """
        n = len(X_seqs)
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                D[i,j] = self._distance(X_seqs.iloc[i], X_seqs.iloc[j])
                D[j,i] = D[i,j]
        return pd.DataFrame(D, index=X_seqs.index, columns=X_seqs.index)

class StructureSEKernel (SEKernel, StructureKernel):

    """ A squared exponential structure kernel.

    Attributes:
        hypers (list): names of the hyperparameters required
        _d_squared (np.ndarray): saved default squared distances
        _saved_X (dict): a dict matching the labels for sequences
            to their contacts
        contacts (list): list of which residues are in contact
    """

    def __init__(self, contacts):
        """ Initiate a StructureSEKernel.

        Parameters:
            contacts: A list of tuples, where each tuple defines two
                positions in contact with each other.
        """
        StructureKernel.__init__(self, contacts)
        SEKernel.__init__(self)

    def _distance(self, seq1, seq2):
        """ Calculate the contact distance between two sequences.

        The contact distance between two sequences of identical length
        is the number of contacts at which they differ.
        This is the 'geometric distance' aquared.

        Parameters:
            seq1 (pd.DataFrame or string): either a DataFrame
                representing the sequence or the key for that sequence
                if it has been saved.
            seq2 (pd.DataFrame or string)

        Returns:
            d (float)
        """
        k = StructureKernel.calc_kernel(self, seq1, seq2, normalize=False)
        n = len(self._get_X(seq1))
        return n - k

    def _get_d_squared(self, X_seqs):
        """ Calculate the contact distances between a set of sequences.

        Dij is the contact distance between Xi and Xj. The contact
        distance is the number of contacts at which two sequences
        differ.

        Parameters:
            X_seqs (pd.DataFrame)

        Returns:
            D (np.ndarray)
        """
        n = len(X_seqs)
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                D[i,j] = self._distance(X_seqs.iloc[i], X_seqs.iloc[j])
                D[j,i] = D[i,j]
        return pd.DataFrame(D, index=X_seqs.index, columns=X_seqs.index)

class HammingSEKernel (SEKernel, HammingKernel):

    """ A squared exponential Hamming kernel.

    Attributes:
        hypers (list): names of the hyperparameters required
        _d_squared (np.ndarray): saved default squared distances
        _saved_X (dict): a dict matching the labels for sequences
            to their contacts
        contacts (list): list of which residues are in contact
    """

    def __init__(self):
        """ Initiate a HammingKernel. """
        HammingKernel.__init__(self)
        SEKernel.__init__(self)

    def _distance(self, seq1, seq2):
        """ Calculate the Hamming distance between two sequences.

        The Hamming distance between two sequences of identical length
        is the number of residues at which they differ.
        This is the 'geometric distance' aquared.

        Parameters:
            seq1 (pd.DataFrame or string): either a DataFrame
                representing the sequence or the key for that sequence
                if it has been saved.
            seq2 (pd.DataFrame or string)

        Returns:
            d (float)
        """
        k = HammingKernel.calc_kernel(self, seq1, seq2, normalize=False)
        n = len(seq1)
        return n - k

    def _get_d_squared(self, X_seqs):
        """ Calculate the Hamming distances between a set of sequences.

        Dij is the Hamming distance between Xi and Xj. The Hamming
        distance is the number of residues at which two sequences
        differ.

        Parameters:
            X_seqs (pd.DataFrame)

        Returns:
            D (np.ndarray)
        """
        n = len(X_seqs)
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                D[i,j] = self._distance(X_seqs.iloc[i], X_seqs.iloc[j])
                D[j,i] = D[i,j]
        return pd.DataFrame(D, index=X_seqs.index, columns=X_seqs.index)

class SumKernel(GPKernel):

    """
    A kernel that sums over other kernels.

    Attributes:
        _kernels (list): list of member kernels
        hypers (list): the names of the hyperparameters
    """

    def __init__(self, kernels):
        '''
        Initiate a SumKernel containing a list of other kernels.

        Parameters:
            kernels(list): list of member kernels
            hypers (list): list of hyperparameter names
        '''
        self._kernels = kernels
        hypers = []
        for k in self._kernels:
            hypers += k.hypers
        self.hypers = [hypers[i] + \
                       str(hypers[0:i].count(hypers[i])) \
                       for i in range(len(hypers))]
        hypers_inds = [len(k.hypers) for k in self._kernels]
        hypers_inds = np.cumsum(np.array(hypers_inds))
        hypers_inds = np.insert(hypers_inds, 0, 0)
        self.hypers_inds = hypers_inds.astype(int)

    def make_K(self, X=None, hypers=None):
        """ Calculate the summed covariance matrix.

        Calculates K for each member kernel using the hyperparameters
        given and sums them.

        Optional parameters:
            X (pd.DataFrame or np.ndarray): the inputs. The default is
                to use the saved inputs.
            hypers (iterable): the hyperparameters. Default is to use
                the defaults for each kernel.

        Returns:
            K (np.ndarray)
        """
        if hypers is None:
            Ks = [k.make_K(X, hypers) for k in self._kernels]
        else:
            hypers_inds = self.hypers_inds
            Ks = [k.make_K(X, hypers[hypers_inds[i]:hypers_inds[i+1]]) for i,
                  k in enumerate(self._kernels)]

        if len(Ks) == 1:
            return Ks[0]
        else:
            K = Ks.pop()
            if isinstance(K, pd.DataFrame):
                K = K.values
            while len(Ks) > 0:
                new_K = Ks.pop()
                if isinstance(new_K, pd.DataFrame):
                    new_K = new_K.values
                K += new_K
        return K

    def calc_kernel(self, x1, x2, hypers=None):
        """ Calculate the sum kernel between two inputs.

        Parameters:
            x1 (np.ndarray or pd.DataFeame string): either a array
                or DataFrame representing the sequence or the key
                for that sequence if it has been saved.
            x2
            hypers (iterable): the hyperparameters. Default is to use
                the defaults for each kernel.

        Returns:
            k (float)
        """
        if hypers is None:
            ks = [kern.calc_kernel(x1, x2) for kern in self._kernels]
        else:
            hypers_inds = self.hypers_inds
            ks = [kern.calc_kernel(x1,x2,hypers[hypers_inds[i]:hypers_inds[i+1]]) for i,
                  kern in enumerate(self._kernels)]
        return sum(ks)

    def train(self, Xs):
        """ Remember the inputs in X.

        Extends the parent method by training each member kernel.

        Parameters:
            X (pd.DataFrame): Saves the inputs in X to a dictionary
                using the index as the keys.
        """
        for k in self._kernels:
            k.train(Xs)

    def delete(self, X):
        """ Forget the inputs in X.

        Extends the parent method by forgetting the input for each
        member kernel.

        Parameters:
            X (pd.DataFrame): Saves the inputs in X to a dictionary
                using the index as the keys.
        """
        for k in self._kernels:
            k.delete(X)

    def set_X(self, X):
        """ Set a default set of inputs X.

        Extends the parent method by setting the default input for each
        member kernel.

        Parameters:
            X (iterable)
        """
        for k in self._kernels:
            k.set_X(X)

class LinearKernel(GPKernel):

    """ The linear (dot product) kernel for two inputs. """

    def __init__(self):
        """ Initiates a LinearKernel. """
        GPKernel.__init__(self)
        self.hypers = ['var_p']


    def make_K (self, X=None, hypers=[1.0]):
        """ Calculate the linear kernel matrix for the points in Xs.

        Parameters:
            Xs (np.ndarray or pd.DataFrame): If none given, uses
                saved values.
            hypers (iterable): The hyperparameters. Default is var_p=1.0.

        Returns:
            K (np.ndarray)
        """
        vp = hypers[0]
        if X is None:
            return self._base_K * vp
        X_mat = np.matrix(X)
        K = X_mat * X_mat.T * vp
        return np.array(K)

    def calc_kernel (self, x1, x2, hypers=[1.0]):
        """ Calculate the linear kernel between x1 and x2.

        The linear kernel is the dot product of x1 and x2 multiplied
        by a scaling variable var_p.

        Parameters:
            x1 (np.ndarray or string): either an array
                representing the sequence or the key for that sequence
                if it has been saved.
            x2 (np.ndarray or string)
            hypers (iterable): default is var_p=1.0.

        Returns:
            k (float)
        """
        vp = hypers[0]
        x1 = self._get_X(x1)
        x2 = self._get_X(x2)
        return sum(a*b for a, b in zip(x1,x2)) * vp

    def set_X(self, X):
        """ Set a default set of inputs X.

        Extends the parent method by also setting a default K with
        var_p = 1.0.

        Parameters:
            X (iterable)
        """
        self.train(X)
        self._base_K = self.make_K(X)


