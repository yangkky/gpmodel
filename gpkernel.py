import numpy as np
import pandas as pd
from sys import exit

class GPKernel (object):
    """
    A Gaussian process kernel for proteins.

       Attribute:
           hypers (list)
    """

    def __init__ (self):
         return

class SEKernel (GPKernel):
    """
    A squared exponential kernel

    Attribute:
        hypers (list)
    """

    def __init__(self):
        self.hypers = ['sigma_f', 'ell']
        super(SEKernel, self).__init__()

    def calc_kernel (self, x1, x2, hypers):
        """ Returns the squared exponential between the points x1 and x2"""
        return self.se([x1, x2], hypers)

    def make_K (self, Xs, hypers):
        """
        Returns the squared exponential covariance matrix for the points
        in Xs.
        """
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
                        d_squared[i][j] = sum([(x1-x2)**2 \
                                         for x1, x2 in zip(xs[i], xs[j])])
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
    """
    def __init__ (self):
        self.saved_seqs = {}
        self.hypers = ['var_p']
        super(HammingKernel,self).__init__()

    def calc_kernel (self, seq1, seq2, hypers=[1.0], normalize=True):
        """ Returns the number of shared amino acids between two sequences"""
        var_p = hypers[0]
        s1 = self.get_sequence(seq1)
        s2 = self.get_sequence(seq2)
        k = sum([1 if str(a) == str(b) else 0 for a,b in zip(s1, s2)])
        if normalize:
            k = float(k) / len(s1)
        return k*var_p

    def make_K (self, seqs=None, hypers=[1.0], normalize=True):
        """ Returns a covariance matrix for two or more sequences of the same length

        Parameters:
            seqs (DataFrame)

        Returns:
            DataFrame
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
        K = np.array(K)
        K_df = pd.DataFrame (K, index = seqs.index, columns = seqs.index)
        return K_df

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
            if X_seqs.index[i] in self.saved_seqs.keys():
                pass
            else:
                self.saved_seqs[X_seqs.index[i]] = ''.join(s for s in X_seqs.iloc[i])

    def delete(self,X_seqs=None):
        """
        Deletes sequences from the saved_seqs dict
        """
        if X_seqs is None:
            self.saved_seqs = {}
        for i in range(len(X_seqs.index)):
            if X_seqs.index[i] in self.saved_seqs.keys():
                del self.saved_seqs[X_seqs.index[i]]

    def get_sequence(self,seq):
        """
        Get the sequence for seq
        """
        try:
            return self.saved_seqs[seq]
        except TypeError:
            return ''.join([s for s in seq])


class StructureKernel (GPKernel):
    """A Structure kernel

    Attributes:
        contact_terms (iterable): Each element in contact_terms should be of the
          form ((pos1,aa1),(pos2,aa2))
        hypers (list): list of required hyperparameters
        saved_contacts (dict): a dict matching the labels for sequences to their contacts
        contacts: list of which residues are in contact
    """

    def __init__ (self, contacts):
        self.contacts = contacts
        self.saved_seqs = {}
        self.contacts = contacts
        self.hypers = ['var_p']
        super (StructureKernel, self).__init__()

    def make_K (self, seqs=None, hypers=[1.0], normalize=True):
        """ Makes the structure-based covariance matrix

            Parameters:
                seqs (DataFrame): amino acid sequences

        Returns:
            Dataframe: structure-based covariance matrix
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
        K_df = pd.DataFrame (K, index=seqs.index, columns=seqs.index)
        return K_df

    def calc_kernel (self, seq1, seq2, hypers=[1.0], normalize=True):
        """ Determine the number of shared contacts between the two sequences

        Parameters:
            seq1 (iterable): Amino acid sequence
            seq2 (iterable): Amino acid sequence

        Returns:
            int: number of shared contacts
        """
        var_p = hypers[0]
        contacts1 = self.get_contacts(seq1)
        contacts2 = self.get_contacts(seq2)
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
            if X_seqs.index[i] in self.saved_seqs.keys():
                continue
            else:
                self.saved_seqs[X_seqs.index[i]] = \
                    self.get_contacts(X_seqs.iloc[i])

    def delete(self, X_seqs=None):
        """
        Delete these sequences from the kernel's contacts dict
        """
        if X_seqs is None:
            self.saved_seqs = {}
        for i in range (len(X_seqs.index)):
            if X_seqs.index[i] in self.saved_seqs.keys():
                del self.saved_seqs[X_seqs.index[i]]

    def get_contacts(self, seq):
        """
        Gets the contacts for seq.
        """
        if isinstance (seq, basestring):
            try:
                return self.saved_seqs[seq]
            except:
                raise ValueError ('Key %s not recognized' %seq)
        try:
            return self.saved_seqs[seq.name]

        except (KeyError, AttributeError):
            seq = ''.join([s for s in seq])
            contacts = []
            for i,con in enumerate(self.contacts):
                term = ((con[0],seq[con[0]]),(con[1],seq[con[1]]))
                contacts.append(term)
            return contacts

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
        n = len(self.get_contacts(seq1))
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







