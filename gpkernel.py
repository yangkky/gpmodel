import numpy as np
import pandas as pd
from numba import jit

def se(xs, params):
    """
    Calculates the squared exponential covariance function
    between xs according to RW Eq. 2.31.
    Each row of xs represents one measurement. Each column represents
    a dimension. The exception is if xs only has one row with multiple
    columns, then each column is assumed to be a measurement.

    Parameters:
        xs: ndarray or np.matrix
        params: sigma_f and ell. sigma_f must be a scalar. ell may
            be a scalar.

    Returns:
        res: the squared exponential covariance evaluated between xs
            res has shape (n,n), where n is the number of rows in xs
            or the number of columns if there is only one row unless
            n = 2, in which case res is a float.
    """
    # check dimensions
    # unpack params
    sigma_f, ell = params


    # calculate the squared radial distances between each pair of
    # measurements
    dims = np.shape(xs)
    n = dims[0]
    # make sure there are multiple measurements
    if n == 1:
        raise RunTimeError ('SE requires at least two items in xs')

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
            d_squared = sum([(x1-x2)**2 for x1, x2 in zip(xs[0], xs[1])])
        else:
            d_squared = np.empty((n,n))
            for i in range (n):
                for j in range(n):
                    d_squared[i][j] = sum([(x1-x2)**2 \
                                     for x1, x2 in zip(xs[i], xs[j])])

    return sigma_f**2 * np.exp(-0.5/np.power(ell,2) * d_squared)



class GPKernel (object):
    """A Gaussian process kernel for proteins.

       Attribute:
           hypers (list)
    """

    def __init__ (self):
         return

class HammingKernel (GPKernel):
    """A Hamming Kernel

    Attributes:
        seqs (dict)
    """
    def __init__ (self):
        self.saved_seqs = {}
        self.hypers = ['var_p']
        super(HammingKernel,self).__init__()

    def calc_kernel (self, seq1, seq2, hypers=[1.0], normalize=False):
        """ Returns the number of shared amino acids between two sequences"""
        var_p = hypers[0]
        s1 = self.get_sequence(seq1)
        s2 = self.get_sequence(seq2)
        k = sum([1 if str(a) == str(b) else 0 for a,b in zip(s1, s2)])*var_p
        if normalize:
            k = float(k) / len(s1)
        return k

    def make_K (self, seqs, hypers=[1.0], normalize=False):
        """ Returns a covariance matrix for two or more sequences of the same length

        Parameters:
            seqs (DataFrame)

        Returns:
            DataFrame
        """
        # note: could probably do this more elegantly without transposing
        n_seqs = len (seqs.index)
        K = np.zeros((n_seqs, n_seqs))
        for n1,i in zip(range(n_seqs), seqs.index):
            for n2,j in zip(range (n_seqs), seqs.index):
                seq1 = seqs.iloc[n1]
                seq2 = seqs.iloc[n2]
                K[n1,n2] = self.calc_kernel (seq1, seq2, hypers=hypers)
        K = np.array(K)
        if normalize:
            K = K/float(K[0][0])

        K_df = pd.DataFrame (K, index = seqs.index, columns = seqs.index)
        return K_df

    def train(self, X_seqs):
        """
        Stores the sequences in X_seqs in the kernel's saved_seqs dict
        """
        for i in range(len(X_seqs.index)):
            if X_seqs.index[i] in self.saved_seqs.keys():
                pass
            else:
                self.saved_seqs[X_seqs.index[i]] = ''.join(s for s in X_seqs.iloc[i])

    def delete(self,X_seqs):
        """
        Deletes sequences from the saved_seqs dict
        """
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

    def __init__ (self, contacts=None, sample_space=None):
        if contacts==None or sample_space==None:
            super (StructureKernel, self).__init__()
        self.contact_terms = self.contacting_terms (sample_space, contacts)
        self.saved_contacts = {}
        self.contacts = contacts
        self.hypers = ['var_p']
        super (StructureKernel, self).__init__()

    def contacting_terms (self, sample_space, contacts):
        """ Lists the possible contacts

        Parameters:
            sample_space (iterable): Each element in sample_space contains the possible
               amino acids at that position
            contacts (iterable): Each element in contacts pairs two positions that
               are considered to be in contact

        Returns:
            list: Each item in the list is a contact in the form ((pos1,aa1),(pos2,aa2))
        """
        contact_terms = []
        for contact in contacts:
            first_pos = contact[0]
            second_pos = contact[1]
            first_possibilities = set(sample_space[first_pos])
            second_possibilities = set(sample_space[second_pos])
            for aa1 in first_possibilities:
                for aa2 in second_possibilities:
                    contact_terms.append(((first_pos,aa1),(second_pos,aa2)))
        return contact_terms


    def make_K (self, seqs, hypers=[1.0], normalize=False):
        """ Makes the structure-based covariance matrix

            Parameters:
                seqs (DataFrame): amino acid sequences

        Returns:
            Dataframe: structure-based covariance matrix
        """
        var_p = hypers[0]
        X = np.matrix(self.make_contacts_X (seqs))
        K = np.einsum('ij,jk->ik', X, X.T)

        if normalize:
            K = K/float(K[0][0])
        K = K*var_p
        return pd.DataFrame(K, index=seqs.index, columns=seqs.index)

    def calc_kernel (self, seq1, seq2, hypers=[1.0], normalize=False):
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


    def make_contacts_X (self, seqs, hypers=[1.0]):
        """ Makes a list with the result of contacts_X_row for each sequence in seqs"""
        X = []
        for i in range(len(seqs.index)):
            X.append(self.contacts_X_row(seqs.iloc[i],hypers))
        return X

    def contacts_X_row (self, seq, hypers=[1.0]):
        """ Determine whether the given sequence contains each of the given contacts

        Parameters:
            seq (iterable): Amino acid sequence
            var_p (float): underlying variance of Gaussian process

        Returns:
            list: 1 for contacts present, else 0
        """
        var_p = hypers[0]
        X_row = []
        for term in self.contact_terms:
            if seq[term[0][0]] == term[0][1] and seq[term[1][0]] == term[1][1]:
                X_row.append (1)
            else:
                X_row.append (0)

        return [var_p*x for x in X_row]

    def train(self, X_seqs):
        """
        Stores the sequences in X_seqs in the kernel's contacts dict
        """
        for i in range(len(X_seqs.index)):
            if X_seqs.index[i] in self.saved_contacts.keys():
                continue
            else:
                self.saved_contacts[X_seqs.index[i]] = self.get_contacts(X_seqs.iloc[i])

    def delete(self, X_seqs):
        """
        Delete these sequences from the kernel's contacts dict
        """
        for i in range (len(X_seqs.index)):
            if X_seqs.index[i] in self.saved_contacts.keys():
                del self.saved_contacts[X_seqs.index[i]]

    def get_contacts(self, seq):
        """
        Gets the contacts for seq.
        """
        try:
            return self.saved_contacts[seq]

        except TypeError:
            seq = ''.join([s for s in seq])
            contacts = []
            for i,con in enumerate(self.contacts):
                term = ((con[0],seq[con[0]]),(con[1],seq[con[1]]))
                contacts.append(term)
#             for term in self.contact_terms:
#                 if seq[term[0][0]] == term[0][1] and seq[term[1][0]] == term[1][1]:
#                     contacts.append(term)
            return contacts






