import sys
import pytest
import itertools
from collections import Counter

import numpy as np

from gpmodel import stringkernel


seqs1 = ['ACGTTTG', 'GTACGGGCT']
seqs2 = ['ACGTTTG', 'CGTACGTA', 'GTACGGGCT']
k = 4
m = 2
A = ['A', 'C', 'G', 'T']

X1 = np.array([[0, 1, 2, 3, 1],
               [0, 2, 1, 3, 2]])
X2 = np.array([[1, 1, 2, 1, 0],
               [0, 2, 1, 3, 2]])
S = np.array([[1.0, 0.0, 0.4, 0.3],
              [0.0, 1.0, 0.2, 0.8],
              [0.4, 0.2, 1.0, 0.5],
              [0.3, 0.8, 0.5, 1.0]])
contacts = [(0, 2), (0, 4), (2, 3), (2, 4), (3, 4)]
L = 5
graph = {
        0: np.array([2, 4]),
        1: np.array([]),
        2: np.array([0, 3, 4]),
        3: np.array([2, 4]),
        4: np.array([0, 2, 3])
    }

S2 = np.array([[1.0, 0.1, 0.4, 0.3],
               [0.1, 0.9, 0.2, 0.9],
               [0.4, 0.2, 1.0, 0.1],
               [0.3, 0.9, 0.1, 0.8]])

def test_mkl():
    k = stringkernel.MultipleDecompositionKernel(contacts, [S, S2], L)
    assert k._n_hypers == 4
    assert len(k.kernels) == 2
    K1 = k.kernels[0].cov(X1, X2)
    K2 = k.kernels[1].cov(X1, X2)
    K = k.cov(X1, X2)
    assert np.allclose(K1 + K2, K)
    h = np.random.random(4)
    K = k.cov(X1, X2, hypers=h)
    w1, w2, g1, g2 = h
    assert np.allclose(K, w1 * K1 ** g1 + w2 * K2 ** g2)
    nh = k.fit(X1)
    assert nh == 4
    assert len(k._saved) == 2
    assert np.allclose(k._saved[0], k.kernels[0].cov(X1, X1))
    assert np.allclose(k._saved[1], k.kernels[1].cov(X1, X1))

def test_wdk():
    k = stringkernel.WeightedDecompositionKernel(contacts, S, L)
    for i, item in graph.items():
        assert np.allclose(item, k.graph[i])
    K = k.cov(X1, X2)
    K_t = np.array([[ 1., 0.36 ],
                    [ 0.264,  5.]])
    assert np.allclose(K, K_t)
    nh = k.fit(X1)
    assert nh == 0.0
    assert np.allclose(k._saved, k.cov(X1, X1))

def count_changes(s1, s2):
    return sum([ss != qq for ss, qq in zip(s1, s2)])

def naive_mismatch_kernel(k, m, A, seqs1, seqs2):
    kmers = [''.join([k for k in km]) for km in itertools.product(A, repeat=k)]
    seq_mers1 = [Counter([seq[i:i + k] for i in range(len(seq) - k + 1)]) for seq in seqs1]
    seq_mers2 = [Counter([seq[i:i + k] for i in range(len(seq) - k + 1)]) for seq in seqs2]
    X1 = np.zeros((len(seqs1), len(kmers)))
    X2 = np.zeros((len(seqs2), len(kmers)))
    for j, km in enumerate(kmers):
        kept = [kmer for kmer in kmers if count_changes(km, kmer) <= m]
        for i, seq_counts in enumerate(seq_mers1):
            for kmer in kept:
                X1[i, j] += seq_counts[kmer]
        for i, seq_counts in enumerate(seq_mers2):
            for kmer in kept:
                X2[i, j] += seq_counts[kmer]
    K11 = np.expand_dims(np.sqrt(np.sum(X1 ** 2, axis=1)), axis=1)
    K22 = np.expand_dims(np.sqrt(np.sum(X2 ** 2, axis=1)), axis=0)

    return X1 @ X2.T / K11 / K22

def test_mismatch_kernel():
    # constructor
    kernel = stringkernel.MismatchKernel(k, A, m)
    assert kernel.k == k
    assert kernel.m == m
    assert kernel.A == A

    # kmer tree
    # pruning
    # saving
    nh = kernel.fit(seqs1)
    assert nh == 1
    K11 = naive_mismatch_kernel(k, m, A, seqs1, seqs1)
    assert np.allclose(K11, kernel._saved)
    # covariance without saving
    K12 = naive_mismatch_kernel(k, m, A, seqs1, seqs2)
    assert np.allclose(K12 * 0.2, kernel.cov(seqs1, seqs2, hypers=(0.2, )))
    # covariance from saved
    assert np.allclose(K11, kernel.cov())
    # normalization
    # hyperparameters

if __name__=="__main__":
    test_mkl()
    test_wdk()
    test_mismatch_kernel()
