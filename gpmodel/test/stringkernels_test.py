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
    test_mismatch_kernel()
