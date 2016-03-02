"""Kernel functions that calculate the covariance between two inputs."""

import numpy as np
import pandas as pd
from sys import exit


class GPKernel (object):

    """ A Gaussian Process kernel.

       Attributes:
           hypers (list)
           saved_X (dict)
    """

    def __init__ (self):
        """ Create a GPKernel. """
        self.saved_X = {}

    def set_X(self, X):
        """ Set a default set of inputs X.

        Parameters:
            X (iterable)
        """
        self.train(X)

    def train(self, X):
        """ Remember the inputs in X.

        Parameters:
            X (pd.DataFrame): Saves the inputs in X to a dictionary
                using the index as the keys.
        """
        for i in range(len(X.index)):
            if X.index[i] in self.saved_X.keys():
                pass
            else:
                self.saved_X[X.index[i]] = X.iloc[i]

    def delete(self, X):
        """ Forget the inputs in X.

        Parameters:
            X (pd.DataFrame): Saves the inputs in X to a dictionary
                using the index as the keys.
        """
        for i in range(len(X.index)):
            if X.index[i] in self.saved_X.keys():
                del self.saved_X[X.index[i]]

    def get_X(self, x):
        """ Retrieve an input x.

        Parameters:
            x: x can be the key used to remember x or the actual values.

        Returns:
            x
        """
        try:
            return self.saved_X[x]
        except (KeyError, AttributeError, TypeError):
            return x

class MaternKernel (GPKernel):

    """ A Matern kernel with nu = 5/2 or 3/2.

    Attributes:
        hypers (list): names of the hyperparameters required
        saved_X (dict): dict of saved index:X pairs
        nu (string): '3/2' or '5/2'
        d (np.ndarray): saved default geometric distances
    """


    def __init__(self, nu):
        """ Initiate a Matern kernel.

        Parameters:
            nu (string): '3/2' or '5/2'
        """
        if nu is not in ['3/2', '5/2']:
            raise ValueError("nu must be '3/2' or '5/2'")
        self.hypers = ['ell']
        self.nu = nu
        GPKernel.__init__(self)

    def calc_kernel(self, x1, x2, hypers):
        """ Calculate the Matern kernel between x1 and x2.

        Parameters:
            x1 (iterable)
            x2 (iterable)
            hypers (iterable): default is ell=1.0.

        Returns:
            k (float)
        """
        x1 = self.get_X(x1)
        x2 = self.get_X(x2)
        d = self.calc_d (np.array([x1, x2]))
        return float(self.matern(d, hypers))

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
            return self.matern(self.d, hypers)
        else:
            d = self.calc_d(Xs)
            K = self.matern(d, hypers)
            return K

    def set_X (self, X):
        """ Remember a default set of inputs X.

        Extends the method from GPKernel by also remembering an array
        of distances between the inputs in X.

        Parameters:
            X (np.ndarray or pd.DataFrame)
        """
        self.train(X)
        X = np.array(X)
        self.d = self.calc_d(X)

    def matern (self, d, hypers=[1.0]):
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


    def calc_d (self, xs):
        """ Calculates the geometric distances between x_i in xs.

        Each row of xs represents one measurement. Each column represents
        a dimension. The exception is if xs only has one row with multiple
        columns, then each column is assumed to be a measurement.

        Parameters:
            xs: ndarray or np.matrix or pd.Dataframe

        Returns:
            D (np.ndarray or float)
        """
        dims = np.shape(xs)
        n = dims[0]

        # multiple 1D measurements
        if len(dims) == 1 or dims[1] == 1:
            if n == 2:
                d = np.sqrt((xs[0] - xs[1])**2)
            else:
                d = np.empty((n,n))
                for i in range (n):
                    for j in range(n):
                        d[i][j] = np.linalg.norm(xs[i] - xs[j])

        # column vector of n-dimensional measurements
        else:
            if n == 2:
                d = np.linalg.norm(xs[0] - xs[1])
            else:
                d = np.empty((n,n))
                for i in range (1,n):
                    for j in range(i):
                        d[i][j] = np.linalg.norm(xs[j] - xs[i])
                        d[j][i] = d[i][j]
                # fill in diagonals
                for i in range (n):
                    d[i][i] = 0
        return d

class SEKernel (GPKernel):

    """ A squared exponential kernel.

    Attribute:
        hypers (list)
        d_squared (np.ndarray)
        saved_X (dict)
    """

    def __init__(self):
        """ Initiate a Matern kernel. """
        self.hypers = ['sigma_f', 'ell']
        super(SEKernel, self).__init__()

    def calc_kernel (self, x1, x2, hypers):
        """ Calculate the squared exponential between x1 and x2.

        """
        x1 = self.get_X(x1)
        x2 = self.get_X(x2)
        return float(self.se([x1, x2], hypers))

    def set_X(self, X):
        self.train(X)
        X = np.array(X)
        self.d_squared = self.get_d_squared(X)

    def make_K (self, Xs=None, hypers=[1.0, 1.0]):
        """ Calculate the Matern kernel matrix for the points in Xs.

        Parameters:
            Xs (np.ndarray or pd.DataFrame): If none given, uses
                saved values.
            hypers (iterable): the hyperparameters. Default is ell=1.0.

        Returns:
            K (np.ndarray)
        """
        if Xs is None:
            return self.d_squared_to_se(self.d_squared, hypers)
        else:
            return self.se(Xs, hypers)


    def dist_to_se (self, dist, params):
        """
        Converts a square matrix of 'distance' measures into a SE covariance
        matrix.

        Parameters:
            dist (numpy.ndarray): cov_i_j is the 'distance between the x_i
                and x_j

        Returns:
            se (numpy.ndarray)
        """
        sigma_f, ell = params
        return sigma_f**2 * np.exp(-0.5/np.power(ell,2) * np.power(dist, 2))

    def d_squared_to_se (self, d_squared, params):
        """
        Converts a square matrix of 'distance' measures into a SE covariance
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

    def get_d_squared (self, xs):
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
        dims = np.shape(xs)
        n = dims[0]

        # multiple 1D measurements
        if len(dims) == 1 or dims[1] == 1:
            if n == 2:
                d_squared = (xs[0] - xs[1])**2
            else:
                d_squared = np.empty((n,n))
                for i in range (n):
                    for j in range(n):
                        d_squared[i][j] = (xs[i] - xs[j])**2

        # column vector of n-dimensional measurements
        else:
            if n == 2:
                d_squared = sum([np.power(x1-x2, 2) for x1, x2 in zip(xs[0], xs[1])])
            else:
                d_squared = np.empty((n,n))
                for i in range (1,n):
                    for j in range(i):
                        d_squared[i][j] = np.linalg.norm(xs[i]-xs[j])**2
                        d_squared[j][i] = d_squared[i][j]
                # fill in diagonals
                for i in range (n):
                    d_squared[i][i] = 0
        return d_squared

    def se(self, xs, params):
        """
        Calculates the squared exponential covariance function
        between xs according to RW Eq. 2.31.
        Each row of xs represents one measurement. Each column represents
        a dimension. The exception is if xs only has one row with multiple
        columns, then each column is assumed to be a measurement.

        Parameters:
            xs: ndarray or np.matrix or pd.DataFrame
            params: sigma_f and ell. sigma_f must be a scalar. ell may
                be a scalar.

        Returns:
            res (np.ndarray or float): the squared exponential covariance
                evaluated between xs. res has shape (n,n), where n is the
                number of rows in xs or the number of columns if there is
                only one row unless n = 2, in which case res is a float.
        """

        # if xs is a DataFrame, convert it to an np.ndarray
        is_df = False
        if isinstance(xs, pd.DataFrame):
            is_df = True
            index = xs.index
            xs = xs.as_matrix()

        # calculate the squared radial distances between each pair of
        # measurements

        # Check dimensions
        dims = np.shape(xs)
        n = dims[0]

        # matrices are weird: make them arrays
        if n == 1 and isinstance(xs, np.matrix):
            xs = np.array(xs)[0]
            dims = np.shape(xs)
            n = dims[0]
        elif isinstance(xs, np.matrix):
            xs = np.array(xs)

        # make sure there are multiple measurements
        if n == 1:
            raise RunTimeError ('SE requires at least two items in xs')

        d_squared = self.get_d_squared (xs)

        se_array = self.d_squared_to_se (d_squared, params)

        # convert back to DataFrame if necessary
        if is_df and n > 2:
            return pd.DataFrame(se_array, index=index, columns=index)
        return se_array


class HammingKernel (GPKernel):
    """A Hamming Kernel

    Attributes:
        seqs (dict)
        hypers (list)
        base_K (DataFrame)
    """
    def __init__ (self):
        self.hypers = ['var_p']
        super(HammingKernel,self).__init__()

    def calc_kernel (self, seq1, seq2, hypers=[1.0], normalize=True):
        """ Returns the number of shared amino acids between two sequences"""
        var_p = hypers[0]
        s1 = self.get_X(seq1)
        s2 = self.get_X(seq2)
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
            return self.base_K*var_p
        n_seqs = len (seqs.index)
        K = np.zeros((n_seqs, n_seqs))
        for n1,i in zip(range(n_seqs), seqs.index):
            for n2,j in zip(range (n_seqs), seqs.index):
                seq1 = seqs.iloc[n1]
                seq2 = seqs.iloc[n2]
                K[n1,n2] = self.calc_kernel (seq1, seq2, hypers=hypers, normalize=normalize)
        return np.array(K)

    def set_X (self, X_seqs):
        """
        Store the sequences in X_seqs in the kernel's contacts dict.
        Use X_sequences to set the kernel's base K.
        """
        self.train (X_seqs)
        self.base_K = self.make_K(X_seqs)

    def train(self, X_seqs):
        """
        Stores the sequences in X_seqs in the kernel's saved_seqs dict
        """
        for i in range(len(X_seqs.index)):
            if X_seqs.index[i] in self.saved_X.keys():
                pass
            else:
                self.saved_X[X_seqs.index[i]] = ''.join(s for s in X_seqs.iloc[i])

    def delete(self,X_seqs=None):
        """
        Deletes sequences from the saved_X dict
        """
        if X_seqs is None:
            self.saved_X = {}
        for i in range(len(X_seqs.index)):
            if X_seqs.index[i] in self.saved_X.keys():
                del self.saved_X[X_seqs.index[i]]

    def get_X(self,seq):
        """
        Get the sequence for seq
        """
        try:
            return self.saved_X[seq]
        except TypeError:
            return ''.join([s for s in seq])

class WeightedHammingKernel (HammingKernel):
    '''
    A Hamming kernel where the covariance between two sequences
    depends on which amino acids are changed to what.
    '''
    def __init__(self):
        from Bio.SubsMat import MatrixInfo
        self.weights = MatrixInfo.blosum62
        self.weights = {k:self.weights[k] for k in self.weights.keys()}
        for k in self.weights.keys():
            self.weights[(k[1], k[0])] = self.weights[k]
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
        var_p = hypers[0]
        s1 = self.get_X(seq1)
        s2 = self.get_X(seq2)
        k = 0.0
        for a,b in zip(s1, s2):
            try:
                k += self.weights[(a,b)]
            except KeyError:
                k+= 0
        if normalize:
            k = float(k) / len(s1)
        return k*var_p

class StructureKernel (GPKernel):
    """A Structure kernel

    Attributes:
        contact_terms (iterable): Each element in contact_terms should be of the
          form ((pos1,aa1),(pos2,aa2))
        hypers (list): list of required hyperparameters
        saved_X (dict): a dict matching the labels for sequences to their contacts
        contacts: list of which residues are in contact
    """

    def __init__ (self, contacts):
        self.contacts = contacts
        self.hypers = ['var_p']
        GPKernel.__init__(self)

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
            return self.base_K*var_p
        n_seqs = len (seqs.index)
        K = np.zeros((n_seqs, n_seqs))
        for n1 in range(n_seqs):
            for n2 in range(n1+1):
                seq1 = seqs.iloc[n1]
                seq2 = seqs.iloc[n2]
                K[n1,n2] = self.calc_kernel (seq1, seq2,
                                             hypers=hypers, normalize=normalize)
                if n1 != n2:
                    K[n2, n1] = K[n1, n2]
        return K

    def calc_kernel (self, seq1, seq2, hypers=[1.0], normalize=True):
        """ Determine the number of shared contacts between the two sequences

        Parameters:
            seq1 (iterable): Amino acid sequence
            seq2 (iterable): Amino acid sequence

        Returns:
            k (float): number of shared contacts * var_p
        """
        var_p = hypers[0]
        contacts1 = self.get_X(seq1)
        contacts2 = self.get_X(seq2)
        k = len(set(contacts1) & set(contacts2))*var_p
        if normalize:
            k = float(k) / len(contacts1)
        return k

    def set_X (self, X_seqs):
        """
        Store the sequences in X_seqs in the kernel's contacts dict.
        Use X_sequences to set the kernel's base K.
        """
        self.train (X_seqs)
        self.base_K = self.make_K(X_seqs)


    def train(self, X_seqs):
        """
        Stores the sequences in X_seqs in the kernel's contacts dict
        """
        for i in range(len(X_seqs.index)):
            if X_seqs.index[i] in self.saved_X.keys():
                continue
            else:
                self.saved_X[X_seqs.index[i]] = \
                    self.get_X(X_seqs.iloc[i])

    def delete(self, X_seqs=None):
        """
        Delete these sequences from the kernel's contacts dict
        """
        if X_seqs is None:
            self.saved_X = {}
        for i in range (len(X_seqs.index)):
            if X_seqs.index[i] in self.saved_X.keys():
                del self.saved_X[X_seqs.index[i]]

    def get_X(self, seq):
        """
        Gets the contacts for seq.
        """
        if isinstance (seq, basestring):
            try:
                return self.saved_X[seq]
            except:
                raise ValueError ('Key %s not recognized' %seq)
        try:
            return self.saved_X[seq.name]

        except (KeyError, AttributeError):
            seq = ''.join([s for s in seq])
            contacts = []
            for i,con in enumerate(self.contacts):
                term = ((con[0],seq[con[0]]),(con[1],seq[con[1]]))
                contacts.append(term)
            return contacts

class StructureMaternKernel(MaternKernel, StructureKernel):
    ''' A Matern structure kernel'''

    def __init__(self, contacts, nu):
        StructureKernel.__init__(self, contacts)
        MaternKernel.__init__(self, nu)

    def calc_kernel(self, seq1, seq2, hypers=[1.0]):
        d = self.distance(seq1, seq2)
        return self.matern(d, hypers)

    def distance(self, seq1, seq2):
        """
        Return the contact distance between two sequences of identical length.
        This is the geometric distance.
        """
        k = StructureKernel.calc_kernel(self, seq1, seq2, normalize=False)
        n = len(self.get_X(seq1))
        return np.sqrt(n - k)

    def set_X(self, X_seqs):
        """
        Stores the X_rows in the kernel's saved_Xs dict.
        Stores the distance matrix.
        """
        self.train(X_seqs)
        self.d = self.get_d(X_seqs)

    def get_d(self, X_seqs):
        n = len(X_seqs)
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                D[i,j] = self.distance(X_seqs.iloc[i], X_seqs.iloc[j])
                D[j,i] = D[i,j]
        return pd.DataFrame(D, index=X_seqs.index, columns=X_seqs.index)


class HammingMaternKernel(MaternKernel, HammingKernel):
    ''' A Matern structure kernel'''

    def __init__(self, contacts, nu):
        HammingKernel.__init__(self)
        MaternKernel.__init__(self, nu)

    def calc_kernel(self, seq1, seq2, hypers=[1.0]):
        d = self.distance(seqs, seq2)
        return self.matern(d, hypers)


    def distance(self, seq1, seq2):
        """
        Return the Hamming distance between two sequences of identical length.
        This is the geometric distance.
        """
        k = HammingKernel.calc_kernel(self, seq1, seq2, normalize=False)
        n = len(self.get_X(seq1))
        return np.sqrt(n - k)

    def set_X(self, X_seqs):
        """
        Stores the X_rows in the kernel's saved_Xs dict.
        Stores the distance matrix.
        """
        self.train(X_seqs)
        self.d = self.get_d(X_seqs)

    def get_d(self, X_seqs):
        n = len(X_seqs)
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                D[i,j] = self.distance(X_seqs.iloc[i], X_seqs.iloc[j])
                D[j,i] = D[i,j]
        return pd.DataFrame(D, index=X_seqs.index, columns=X_seqs.index)

class StructureSEKernel (StructureKernel, SEKernel):
    """
    A squared exponential structure kernel
    """

    def __init__(self, contacts):
        StructureKernel.__init__(self, contacts)
        SEKernel.__init__(self)

    def make_K (self, X_seqs=None, hypers=(1.0, 1.0)):
        if X_seqs is None:
            return self.d_squared_to_se(self.d_squared, hypers)
        D = self.make_D(X_seqs)
        K = SEKernel.d_squared_to_se(self, D, hypers)
        return pd.DataFrame(K, index=X_seqs.index, columns=X_seqs.index)

    def calc_kernel (self, seq1, seq2, hypers):
        d = self.distance(seq1, seq2)
        return SEKernel.d_squared_to_se(self, d, hypers)


    def set_X(self, X_seqs):
        """
        Stores the X_rows in the kernel's saved_Xs dict.
        Stores the distance matrix.
        """
        self.train(X_seqs)
        self.d_squared = self.make_D(X_seqs)

    def distance(self, seq1, seq2):
        """
        Return the contact distance between two sequences of identical length.
        This is the geometric distance squared.
        """
        k = StructureKernel.calc_kernel(self, seq1, seq2, normalize=False)
        n = len(self.get_X(seq1))
        return n - k

    def make_D(self, X_seqs):
        n = len(X_seqs)
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                D[i,j] = self.distance(X_seqs.iloc[i], X_seqs.iloc[j])
                D[j,i] = D[i,j]
        return pd.DataFrame(D, index=X_seqs.index, columns=X_seqs.index)


class HammingSEKernel (HammingKernel, SEKernel):
    """
    A squared exponential Hamming kernel
    """

    def __init__(self):
        HammingKernel.__init__(self)
        SEKernel.__init__(self)

    def make_K (self, X_seqs=None, hypers=(1.0, 1.0)):
        if X_seqs is None:
            return self.d_squared_to_se(self.d_squared, hypers)
        D = self.make_D(X_seqs)
        K = SEKernel.d_squared_to_se(self, D, hypers)
        return pd.DataFrame(K, index=X_seqs.index, columns=X_seqs.index)

    def calc_kernel (self, seq1, seq2, hypers):
        d = self.distance(seq1, seq2)
        return SEKernel.d_squared_to_se(self, d, hypers)


    def set_X(self, X_seqs):
        """
        Stores the X_rows in the kernel's saved_Xs dict.
        Stores the distance matrix.
        """
        self.train(X_seqs)
        self.d_squared = self.make_D(X_seqs)

    def distance(self, seq1, seq2):
        """
        Return the hamming distance between two sequences of identical length.
        This is the geometric distance squared.
        """
        k = HammingKernel.calc_kernel(self, seq1, seq2, normalize=False)
        n = len(seq1)
        return n - k

    def make_D(self, X_seqs):
        n = len(X_seqs)
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                D[i,j] = self.distance(X_seqs.iloc[i], X_seqs.iloc[j])
                D[j,i] = D[i,j]
        return pd.DataFrame(D, index=X_seqs.index, columns=X_seqs.index)

class SumKernel(GPKernel):
    '''
    A kernel that sums over other kernels

    Attributes:
        kernels (list): list of member kernels
    '''

    def __init__(self, kernels):
        '''
        Initiate a SumKernel containing a list of other kernels.

        Parameters:
            kernels(list): list of member kernels
        '''
        self.kernels = kernels
        hypers = []
        for k in self.kernels:
            hypers += k.hypers
        self.hypers = [hypers[i] + \
                       str(hypers[0:i].count(hypers[i])) \
                       for i in range(len(hypers))]
        hypers_inds = [len(k.hypers) for k in self.kernels]
        hypers_inds = np.cumsum(np.array(hypers_inds))
        hypers_inds = np.insert(hypers_inds, 0, 0)
        self.hypers_inds = hypers_inds.astype(int)


    def make_K(self, X=None, hypers=None):
        if hypers is None:
            Ks = [k.make_K(X, hypers) for k in self.kernels]
        else:
            hypers_inds = self.hypers_inds
            Ks = [k.make_K(X, hypers[hypers_inds[i]:hypers_inds[i+1]]) for i,
                  k in enumerate(self.kernels)]

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
        if hypers is None:
            ks = [kern.calc_kernel(x1, x2) for kern in self.kernels]
        else:
            hypers_inds = self.hypers_inds
            ks = [kern.calc_kernel(x1,x2,hypers[hypers_inds[i]:hypers_inds[i+1]]) for i,
                  kern in enumerate(self.kernels)]
        return sum(ks)

    def train(self, Xs):
        for k in self.kernels:
            k.train(Xs)

    def delete(self, X):
        for k in self.kernels:
            k.delete(X)

    def set_X(self, X):
        for k in self.kernels:
            k.set_X(X)

class LinearKernel(GPKernel):
    '''
    Calculates the linear (dot product) kernel for two inputs
    '''

    def __init__(self):
        self.hypers = ['var_p']
        GPKernel.__init__(self)

    def make_K (self, X=None, hypers=(1.0,)):
        vp = hypers[0]
        if X is None:
            return self.base_K * vp
        X_mat = np.matrix(X)
        K = X_mat * X_mat.T * vp
        return pd.DataFrame(K, index=X.index, columns=X.index)

    def calc_kernel (self, x1, x2, hypers=(1,)):
        vp = hypers[0]
        x1 = self.get_X(x1)
        x2 = self.get_X(x2)
        return sum(a*b for a, b in zip(x1,x2)) * vp

    def set_X(self, X):
        self.train(X)
        self.base_K = self.make_K(X)


