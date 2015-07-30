''' Functions and classes for doing Gaussian process models of proteins'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import math
from sys import exit
######################################################################
# Here are the functions associated with Hamming kernels
######################################################################

def hamming_kernel (seq1, seq2, var_p=1):
    """ Returns the number of shared amino acids between two sequences"""
    return sum ([1 if str(a) == str(b) else 0 for a,b in zip(seq1, seq2)])*var_p
    
def make_hamming (seqs):
    """ Returns a hamming matrix for two or more sequences of the same length

    Parameters: 
        seqs (DataFrame)
    
    Returns: 
        DataFrame
    """
    # note: could probably do this more elegantly without transposing
    n_seqs = len (seqs.index)
    hamming = np.zeros((n_seqs, n_seqs))
    for n1,i in zip(range(n_seqs), seqs.index):
        for n2,j in zip(range (n_seqs), seqs.transpose().columns):
            seq1 = seqs.loc[i]
            seq2 = seqs.transpose()[j]
            hamming[n1,n2] = hamming_kernel (seq1, seq2)
    hamming_df = pd.DataFrame (hamming, index = seqs.index, columns = seqs.index)
    return hamming_df          
        
######################################################################
# Here are functions specific to the structure-based kernel
######################################################################

def generate_library (n_blocks, n_parents):
    """Generates all the chimeras with n_blocks blocks and n_parents parents
    
    Parameters: 
        n_blocks (int): number of blocks
        n_parents (int): number of parents
        
    Returns: 
        list: all possible chimeras
    """
    if n_blocks > 1:
        this = ([i+str(n) for i in generate_library(n_blocks-1,n_parents) for n in range (n_parents)])
        return this
    return [str(i) for i in range (n_parents)]
    
def contacts_X_row (seq, contact_terms, var_p):
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
    for term in contact_terms:
        if seq[term[0][0]] == term[0][1] and seq[term[1][0]] == term[1][1]:
            X_row.append (1)
        else:
            X_row.append (0)
    return [var_p*x for x in X_row]
    
def structure_kernel (seq1, seq2, contact_terms, var_p=1):
    """ Determine the number of shared contacts between the two sequences
    
    Parameters: 
        seq1 (iterable): Amino acid sequence
        seq2 (iterable): Amino acid sequence
        contact_terms (iterable): Each element in contact_terms should be of the
          form ((pos1,aa2),(pos2,aa2))
        var_p (float): underlying variance of Gaussian process  
  
    Returns: 
        int: number of shared contacts
    """
    X1 = contacts_X_row (seq1, contact_terms)
    X2 = contacts_X_row (seq2, contact_terms)
    return sum ([1 if a == b else 0 for a,b in zip(X1, X2)])*var_p

    
def make_contacts_X (seqs, contact_terms,var_p=1):
    """ Makes a list with the result of contacts_X_row for each sequence in seqs"""
    
    X = []
    for i in seqs.index:
        X.append(contacts_X_row(seqs.loc[i],contact_terms,var_p))
    return X
    
def make_structure_matrix (seqs, contact_terms,var_p=1):
    """ Makes the structure-based covariance matrix
    
    Parameters: 
        seqs (DataFrame): amino acid sequences
        contact_terms (iterable): Each element in contact_terms should be of the
          form ((pos1,aa2),(pos2,aa2))    
        var_p (float): underlying variance of Gaussian process  
    
    Returns: 
        Dataframe: structure-based covariance matrix
    """
    X = np.matrix(make_contacts_X (seqs, contact_terms,var_p))
    return pd.DataFrame(np.einsum('ij,jk->ik', X, X.T), index=seqs.index, columns=seqs.index)

def contacting_terms (sample_space, contacts):
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


######################################################################
# Here are tools that are generally useful
######################################################################
def plot_predictions (real_Ys, predicted_Ys,stds=None,file_name=None,title='',label='', line=True):
    if stds is None:
        plt.plot (real_Ys, predicted_Ys, 'g.')
    else:
        plt.errorbar (real_Ys, predicted_Ys, yerr = [stds, stds], fmt = 'g.')
    small = min(set(real_Ys) | set(predicted_Ys))*1.1
    large = max(set(real_Ys) | set(predicted_Ys))*1.1
    if line:
        plt.plot ([small, large], [small, large], 'b--')
    plt.xlabel ('Actual ' + label)
    plt.ylabel ('Predicted ' + label)
    plt.title (title)
    plt.text(small*.9, large*.7, 'R = %.3f' %np.corrcoef(real_Ys, predicted_Ys)[0,1])
    if not file_name is None:
        plt.savefig (file_name)
        
def plot_LOO(Xs, Ys, args=[], kernel='Hamming',save_as=None, lab=''):
    std = []
    predicted_Ys = []
    for i in Xs.index:
        train_inds = list(set(Xs.index) - set(i))
        train_Xs = Xs.loc[train_inds]
        train_Ys = Ys.loc[train_inds]
        verify = Xs.loc[[i]]
        if kernel == 'Hamming':
            predicted = HammingModel(train_Xs,train_Ys,guesses=[.001,.250]).predict(verify)
        [(E,v)] = predicted
        std.append(math.pow(v,0.5))
        predicted_Ys.append (E)
    plot_predictions (Ys.tolist(), predicted_Ys, stds=std, label=lab, file_name=save_as)

######################################################################
# Now we start class definitions
######################################################################  
      
class GPModel(object):
    """A Gaussian process model for proteins. 

    Attributes:
        Ky (np.matrix): noisy covariance matrix [var_p*K+var_n*I]
        L (np.matrix): lower triangular Cholesky decomposition of Ky
        alpha (np.matrix): L.T\(L\Y)
        ML (float): The negative log marginal likelihood
    """
    
    #__metaclass__ = ABCMeta
    
    def __init__ (self, Ky):
        self.Ky = Ky
        self.L = np.linalg.cholesky(self.Ky)
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
    
    def log_ML (self,Ky):
        """ Returns the negative log marginal likelihood.  
        
        Parameters: 
            Ky (np.matrix): [var_p*K+var_n*I]
    
        Uses RW Equation 5.8
        """
        Y_mat = np.matrix(self.Y)
        L = np.linalg.cholesky (Ky)
        alpha = np.linalg.lstsq(L.T,np.linalg.lstsq (L, np.matrix(Y_mat).T)[0])[0]
        ML = (0.5*Y_mat*alpha + 0.5*math.log(np.linalg.det(L)**2) + len(Y_mat)/2*math.log(2*math.pi)).item()
        return ML

class StructureModel(GPModel):
    """A Gaussian process model for proteins with a structure-based kernel

    Attributes:
        X_seqs (DataFrame): The sequences in the training set
        Y (Series): The outputs for the training set
        var_p (float): the hyperparameter to use in in the kernel function
        var_n (float): signal variance        
        K (DataFrame): Covariance matrix
        Ky (np.matrix): [var_p*K+var_n*I]
        L (np.matrix): lower triangular Cholesky decomposition of Ky
        alpha (np.matrix): L.T\(L\Y)
        contacts (iterable): Each element in contacts pairs two positions that 
               are considered to be in contact  
        contact_terms (iterable): Each element in contact_terms should be of the
          form ((pos1,aa2),(pos2,aa2))     
        sample_space (iterable): Each element in sample_space contains the possible 
           amino acids at that position
    """
    def __init__ (self, sequences, outputs, contacts, sample_space, guesses=[100.,1000.]):
        self.X_seqs = sequences
        self.Y = outputs
        self.contacts = contacts
        self.sample_space = sample_space
        self.contact_terms = contacting_terms(self.sample_space, self.contacts)
        self.K = make_structure_matrix (self.X_seqs, self.contact_terms)
        minimize_res = minimize(self.log_ML,(guesses), bounds=[(1e-5,None),(1e-5,None)], method='L-BFGS-B')
        self.var_n,self.var_p = minimize_res['x']
        self.ML = minimize_res['fun']
        super(StructureModel,self).__init__(self.var_p*np.matrix(self.K) + self.var_n*np.identity(len(self.K)))
        
    def log_ML (self, variances):
        """ Returns the negative log marginal likelihood.  
    
        Uses RW Equation 5.8
    
        Parameters: 
            variances (iterable): var_n and var_p

        Returns: 
            L (float): the negative log marginal likelihood
        """
        var_n,var_p = variances
        K_mat = np.matrix (self.K)
        return super(StructureModel,self).log_ML(K_mat*var_p+np.identity(len(K_mat))*var_n)  
        
    def predict (self, new_seqs):
        """ Calculates predicted (mean, variance) for each sequence in new_seqs
    
        Uses Equations 2.23 and 2.24 of RW and a structure-based kernel
        
        Parameters: 
            new_seqs (DataFrame): sequences to predict
            
         Returns: 
            predictions (list): (E,v) as floats
        """
        predictions = []
        for ns in [new_seqs.loc[i] for i in new_seqs.index]:
            k = np.matrix([structure_kernel(ns,seq1,self.var_p) for seq1 in [self.X_seqs.loc[i] for i in self.X_seqs.index]])
            k_star = structure_kernel(ns,ns,self.var_p)
            predictions.append(super(StructureModel,self).predict(k, k_star))
        return predictions   

class HammingModel(GPModel):
    """A Gaussian process model for proteins with a Hamming kernel

    Attributes:
        X_seqs (DataFrame): The sequences in the training set
        Y (Series): The outputs for the training set
        var_p (float): the hyperparameter to use in in the kernel function
        var_n (float): signal variance        
        K (DataFrame): Covariance matrix
        Ky (np.matrix): [var_p*K+var_n*I]
    """
    
    def __init__ (self, sequences, outputs, guesses=[10.,10.]):
        self.X_seqs = sequences
        self.Y = outputs
        self.K = make_hamming (self.X_seqs)
        minimize_res = minimize(self.log_ML,(guesses), bounds=[(1e-5,None),(1e-5,None)], method='TNC')
        self.var_n,self.var_p = minimize_res['x']
        self.ML = minimize_res['fun']
        super(HammingModel,self).__init__(self.var_p*np.matrix(self.K) + self.var_n*np.identity(len(self.K)))
                
    def log_ML (self, variances):
        """ Returns the negative log marginal likelihood.  
    
        Uses RW Equation 5.8
    
        Parameters: 
            variances (iterable): var_n and var_p

        Returns: 
            L (float): the negative log marginal likelihood
        """
        var_n,var_p = variances
        K_mat = np.matrix (self.K)
        return super(HammingModel,self).log_ML(K_mat*var_p+np.identity(len(K_mat))*var_n)
            
    def predict (self, new_seqs):
        """ Calculates predicted (mean, variance) for each sequence in new_seqs
    
        Uses Equations 2.23 and 2.24 of RW and a hamming kernel
        
        Parameters: 
            new_seqs (DataFrame): sequences to predict
            
         Returns: 
            predictions (list): (E,v) as floats
        """
        predictions = []
        for ns in [new_seqs.loc[i] for i in new_seqs.index]:
            k = np.matrix([hamming_kernel(ns,seq1,self.var_p) for seq1 in [self.X_seqs.loc[i] for i in self.X_seqs.index]])
            k_star = hamming_kernel(ns,ns,self.var_p)
            predictions.append(super(HammingModel,self).predict(k, k_star))
        return predictions 
        