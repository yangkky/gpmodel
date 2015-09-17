import numpy as np
import pandas as pd

class GPKernel (object):
    """A Gaussian process kernel for proteins. 
    """
        
    def __init__ (self):
         return        
     
class HammingKernel (GPKernel):
    """A Hamming Kernel

    Attributes:
    """
    def __init__ (self):
        super(HammingKernel,self).__init__()
        
    def calc_kernel (self, seq1, seq2, var_p=1):
        """ Returns the number of shared amino acids between two sequences"""
        return sum ([1 if str(a) == str(b) else 0 for a,b in zip(seq1, seq2)])*var_p
        
    def make_K (self, seqs, var_p=1):
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
            for n2,j in zip(range (n_seqs), seqs.transpose().columns):
                seq1 = seqs.loc[i]
                seq2 = seqs.transpose()[j]
                K[n1,n2] = self.calc_kernel (seq1, seq2, var_p=var_p)
        K_df = pd.DataFrame (K, index = seqs.index, columns = seqs.index)
        return K_df    
        
class StructureKernel (GPKernel):
    """A Structure kernel

    Attributes:
        contact_terms (iterable): Each element in contact_terms should be of the
          form ((pos1,aa2),(pos2,aa2))            
    """
    
    def __init__ (self, contacts, sample_space):
        self.contact_terms = self.contacting_terms (sample_space, contacts)
        super (StructureKernel, self).__init__()
        
    def contacting_terms (self, sample_space, contacts):
        """ Lists the possible contacts
    
        Parameters: 
            sample_space (iterable): Each element in sample_space contains the possible 
               amino acids at that position
            contacts (iterable): Each eleent in contacts pairs two positions that 
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

    def make_K (self, seqs, var_p=1):
        """ Makes the structure-based covariance matrix
    
            Parameters: 
                seqs (DataFrame): amino acid sequences
    
        Returns: 
            Dataframe: structure-based covariance matrix
        """
        X = np.matrix(self.make_contacts_X (seqs,var_p))
        return pd.DataFrame(np.einsum('ij,jk->ik', X, X.T), index=seqs.index, columns=seqs.index)
        
    def calc_kernel (self, seq1, seq2, var_p=1):
        """ Determine the number of shared contacts between the two sequences
    
        Parameters: 
            seq1 (iterable): Amino acid sequence
            seq2 (iterable): Amino acid sequence
  
        Returns: 
            int: number of shared contacts
        """
        X1 = self.contacts_X_row (seq1, self.contact_terms)
        X2 = self.contacts_X_row (seq2, self.contact_terms)
        return sum ([1 if a == b else 0 for a,b in zip(X1, X2)])*var_p
        
    def make_contacts_X (self, seqs, var_p):
        """ Makes a list with the result of contacts_X_row for each sequence in seqs"""
        X = []
        for i in seqs.index:
            X.append(self.contacts_X_row(seqs.loc[i],var_p))
        return X
        
    def contacts_X_row (self, seq, var_p=1):
        """ Determine whether the given sequence contains each of the given contacts
    
        Parameters: 
            seq (iterable): Amino acid sequence
            contact_terms (iterable): Each element in contact_terms should be of the
                form ((pos1,aa2),(pos2,aa2))
            var_p (float): underlying variance of Gaussian process  
            
        Returns: 
            list: 1 for contacts present, else 0
        """
        X_row = []
        for term in self.contact_terms:
            if seq[term[0][0]] == term[0][1] and seq[term[1][0]] == term[1][1]:
                X_row.append (1)
            else:
                X_row.append (0)
        return [var_p*x for x in X_row]
        

    
            
