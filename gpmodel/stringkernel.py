import itertools
from collections import Counter
import multiprocessing as mp

import numpy as np
import numba
from scipy import sparse

from gpmodel.gpkernel import BaseKernel

@numba.jit(nopython=True)
# @numba.autojit
def wdk(subs, graph):
    K = 0
    # The last one was added for the ones that don't have as many contacts
    for k in numba.prange(len(subs) - 1):
        s = subs[k]
    # for k, s in enumerate(subs[:-1]):
        K += s * np.sum(subs[graph[k]])
    return K

# @numba.jit(nopython=True)
def sdk(subs, adj):
    masked = subs * adj
    masked = masked.sum(axis=-1)
    return np.sum(masked.T * subs)

class MultipleKernel(BaseKernel):

    """ Weighted sum of kernels with no individual hyperparameters. """

    def __init__(self, kernels):
        self.kernels = kernels
        self._n_hypers = 2 * len(kernels)
        return

    def fit(self, X):
        # pool = mp.Pool(processes=8)
        # self._saved = [pool.apply(ke.cov, args=(X, X)) for ke in self.kernels]
        self._saved = [ke.cov(X, X) for ke in self.kernels]
        return self._n_hypers

    def cov(self, X1=None, X2=None, hypers=None):
        if hypers is None:
            hypers = np.ones(self._n_hypers)
        w = np.expand_dims(hypers[:self._n_hypers // 2], 1)
        w = np.expand_dims(w, 2)
        gamma = hypers[self._n_hypers // 2:].reshape(len(self.kernels), 1, 1)
        if X1 is None and X2 is None:
            base = self._saved
        else:
            # pool = mp.Pool(processes=8)
            # base = [pool.apply(ke.cov, args=(X1, X2)) for ke in self.kernels]
            base = [ke.cov(X1, X2)for ke in self.kernels]
        # base = [K ** g for K, g in zip(base, gamma)]
        base = np.array(base)
        base = base ** gamma
        return np.sum(base * w, axis=0)

class WeightedDecompositionKernel(BaseKernel):

    """
    A weighted decomposition kernel.

    Attributes:
        graph (dict):
        S (np.ndarray): Substitution matrix
    """

    def __init__(self, contacts, S, L):
        """ Instantiate a WeightedDecompositionKernel.

        Parameters:
            contacts (list):
            S (np.ndarray): Substitution matrix
            L (int): maximum length
        """
        self.S = S
        self.graph = self.make_graph(contacts, L)
        self._n_hypers = 0
        return

    def make_graph(self, contacts, L):
        """ Return a dict enumerating the neighbors for each position"""
        graph = [[] for i in range(L)]
        for c1, c2 in contacts:
            graph[c1].append(int(c2))
            graph[c2].append(int(c1))
        max_L = max([len(g) for g in graph])
        # Fill with -1s so that every row has the same length
        graph = [g + [-1] * (max_L - len(g)) for g in graph]
        return np.array(graph).astype(int) # numba does not allow float indexers of arrays

    def fit(self, X):
        """ Precompute the kernel for a set of sequences."""
        self._saved = self.cov(X1=X, X2=X)
        return self._n_hypers

    def cov(self, X1=None, X2=None, hypers=None):
        """Calculate the weighted decomposition kernel.

        If no sequences given, then uses precomputed kernel.

        Parameters:
            X1 (np.ndarray): 0-indexed tokens
            X2 (np.ndarray):
            hypers (iterable): the sigma value

        Returns:
            K (np.ndarray): n1 x n2 normalized mismatch string kernel
        """
        if X1 is None and X2 is None:
            return self._saved
        # Get pairwise substitution values
        n1, L = X1.shape
        n2, _ = X2.shape
        K = np.zeros((n1, n2))
        if n1 == n2:
            square = np.allclose(X1, X2)
        else:
            square = False
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                if square:
                    if i > j:
                        K[i, j] = K[j, i]
                subs = np.zeros(L + 1)
                subs[:-1] = self.S[x1, x2]
                K[i, j] = wdk(subs, self.graph)
        k1 = np.zeros((n1, 1))
        for i, x1 in enumerate(X1):
            subs = self.S[x1, x1]
            subs = np.append(subs, 0)
            k1[i, 0] = wdk(subs, self.graph)
        k2 = np.zeros((1, n2))
        for i, x2 in enumerate(X2):
            subs = self.S[x2, x2]
            subs = np.append(subs, 0)
            k2[0, i] = wdk(subs, self.graph)
        return K / np.sqrt(k1) / np.sqrt(k2)

class SmoothDecompositionKernel(BaseKernel):

    """
    A smoothed weighted decomposition kernel.

    Attributes:
        adj (np.ndarray): Adjacency matrix
        S (np.ndarray): Substitution matrix
    """

    def __init__(self, D, S, L=4.5, power=2):
        """ Instantiate a WeightedDecompositionKernel.

        Parameters:
            D (np.ndarray): Position distance matrix
            S (np.ndarray): Substitution matrix
        """
        self.S = S
        self.adj = self.make_graph(D, L, power)
        self._n_hypers = 0
        return

    def make_graph(self, D, L, power):
        """ Make adjacency matrix."""
        np.fill_diagonal(D, 1)
        graph = (D / L) ** (-power)
        np.fill_diagonal(graph, 0)
        return graph

    def fit(self, X):
        """ Precompute the kernel for a set of sequences."""
        self._saved = self.cov(X1=X, X2=X)
        return self._n_hypers

    def cov(self, X1=None, X2=None, hypers=None):
        """Calculate the weighted decomposition kernel.

        If no sequences given, then uses precomputed kernel.

        Parameters:
            X1 (np.ndarray): 0-indexed tokens
            X2 (np.ndarray):
            hypers (iterable): the sigma value

        Returns:
            K (np.ndarray): n1 x n2 normalized mismatch string kernel
        """
        if X1 is None and X2 is None:
            return self._saved
        # Get pairwise substitution values
        n1, L = X1.shape
        n2, _ = X2.shape
        is_square = False
        if n1 == n2:
            if np.allclose(X1, X2):
                is_square = True
        K = np.empty((n1, n2))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                if is_square and i > j:
                        K[i, j] = K[j, i]
                else:
                    subs = self.S[x1, x2]
                    K[i, j] = sdk(subs, self.adj)
        K11 = np.empty((n1, 1))
        for i, x1 in enumerate(X1):
            subs = self.S[x1, x1]
            K11[i, 0] = sdk(subs, self.adj)
        K22 = np.empty((1, n2))
        for i, x2 in enumerate(X2):
            subs = self.S[x2, x2]
            K22[0, i] = sdk(subs, self.adj)
        return K / np.sqrt(K11) / np.sqrt(K22)


class MismatchKernel(BaseKernel):

    """
    A mismatch kmer kernel as described by Leslie et al. here:
    https://academic.oup.com/bioinformatics/article/20/4/467/192308

    Attributes:
        k (int): size of the kmers
        A (list): allowed alphabet. Elements must be strings.
        m (int): allowed mismatches
    """

    def __init__(self, k, A, m):
        """ Instantiate a MismatchKernel.

        Parameters:
            k (int): size of the kmers
            A (list): allowed alphabet. Elements must be strings.
            m (int): allowed mismatches
        """
        self.k = k
        self.A = A
        self.m = m
        self.A_to_num = {a:i for i, a in enumerate(A)}
        self.num_to_A = {i:a for i, a in enumerate(A)}
        self.nums = list(range(len(A)))
        self.nodes = self.make_kmer_tree(k, self.nums)
        self._n_hypers = 1
        self._saved = None
        return

    def make_kmer_tree(self, k, nums):
        """ Return a list representing the kmer tree."""
        nodes = [(np.array([]), [])]
        for it in range(k):
            new_nodes = []
            count = 0
            for i, node in enumerate(nodes):
                n, e = node
                if len(n) < it:
                    continue
                for a in nums:
                    count += 1
                    new_node = (np.append(n, a), [])
                    new_nodes.append(new_node)
                    nodes[i][1].append(len(nodes) + count - 1)
            nodes += new_nodes
        return nodes

    def prune(self, candidates, mutations, prefix):
        """
        Candidates are indices to kmer candidates,
        mutations are corresponding mutation counts
        prefix is kmer as vector
        """
        L = len(prefix)
        if L == 0:
            return candidates, mutations
        mutant = self.observed[candidates][:, L - 1] != prefix[-1]
        mutations[candidates] += mutant
        keep_me = mutations <= self.m
        candidates = candidates[keep_me[candidates]]
        return candidates, mutations

    def fit(self, seqs):
        """ Precompute the kernel for a set of sequences."""
        self._saved = self.cov(seqs1=seqs, seqs2=seqs)
        return self._n_hypers

    def cov(self, seqs1=None, seqs2=None, hypers=(1.0,)):
        """Calculate the mismatch string kernel.

        If no sequences given, then uses precomputed kernel.

        Parameters:
            seqs1 (list): list of strings
            seqs2 (list): list of strings
            hypers (iterable): the sigma value

        Returns:
            K (np.ndarray): n1 x n2 normalized mismatch string kernel
        """
        if seqs1 is None and seqs2 is None:
            return self._saved * hypers[0]
        if not isinstance(seqs1, list):
            seqs1 = list(seqs1)
        if not isinstance(seqs2, list):
            seqs2 = list(seqs2)
        # Break seqs into kmers
        kmers1 = [[seq[i:i + self.k] for i in range(len(seq) - self.k + 1)] for seq in seqs1]
        kmers2 = [[seq[i:i + self.k] for i in range(len(seq) - self.k + 1)] for seq in seqs2]
        # Get all observed kmers
        self.observed = sorted(set(itertools.chain.from_iterable(kmers1 + kmers2)))
        # Get count of each observed kmer for each sequence
        self.X1 = np.zeros((len(seqs1), len(self.observed)))
        self.X2 = np.zeros((len(seqs2), len(self.observed)))
        kmer_counts1 = [Counter(kmer) for kmer in kmers1]
        kmer_counts2 = [Counter(kmer) for kmer in kmers2]
        for j, obs in enumerate(self.observed):
            for i, counts in enumerate(kmer_counts1):
                self.X1[i, j] = counts[obs]
            for i, counts in enumerate(kmer_counts2):
                self.X2[i, j] = counts[obs]
        # Convert observed kmers to an array
        self.observed = np.array([[self.A_to_num[a] for a in obs] for obs in self.observed])
        # Create the covariance matrix
        self.K = np.zeros((len(seqs1), len(seqs2)))
        # Create the variance matrices
        self.K11 = np.zeros((len(seqs1), 1))
        self.K22 = np.zeros((len(seqs2), 1))
        # Initialize the mutation counts
        mutations = np.zeros(len(self.observed))
        # Initialize the candidate indices
        candidates = np.arange(len(mutations))
        # Populate K
        self.dft(candidates.copy(), mutations.copy(), 0)
        # Normalize K
        self.K /= np.sqrt(self.K11)
        self.K /= np.sqrt(self.K22.T)
        self.K *= hypers[0]
        return self.K

    def dft(self, candidates, mutations, ind):
        """ Depth first traversal of kmer tree to calculate K."""
        kmer = self.nodes[ind][0]
        candidates, mutations = self.prune(candidates, mutations, kmer)
        if len(candidates) == 0:
            return
        if len(self.nodes[ind][1]) == 0:
            Y = np.zeros((len(self.observed), 1))
            Y[candidates, 0] = 1
            n_alphas1 = self.X1 @ Y
            n_alphas2 = self.X2 @ Y
            self.K += n_alphas1 @ n_alphas2.T
            self.K11 += n_alphas1 ** 2
            self.K22 += n_alphas2 ** 2
        for e in self.nodes[ind][1]:
            self.dft(candidates.copy(), mutations.copy(), e)
