import sys

import pandas as pd
import numpy as np
import pickle
from gpmodel.chimera_tools import *

contacts = [(0, 1), (0, 2), (1, 2)]
sample_space = [('A', 'B', 'C'), ('A', 'A', 'C'), ('B', 'A', 'D')]
contact_terms = [((0, 'A'), (1, 'A')),
                 ((0, 'A'), (1, 'C')),
                 ((0, 'B'), (1, 'A')),
                 ((0, 'B'), (1, 'C')),
                 ((0, 'C'), (1, 'A')),
                 ((0, 'C'), (1, 'C')),
                 ((0, 'A'), (2, 'A')),
                 ((0, 'A'), (2, 'B')),
                 ((0, 'A'), (2, 'D')),
                 ((0, 'B'), (2, 'A')),
                 ((0, 'B'), (2, 'B')),
                 ((0, 'B'), (2, 'D')),
                 ((0, 'C'), (2, 'A')),
                 ((0, 'C'), (2, 'B')),
                 ((0, 'C'), (2, 'D')),
                 ((1, 'A'), (2, 'A')),
                 ((1, 'A'), (2, 'B')),
                 ((1, 'A'), (2, 'D')),
                 ((1, 'C'), (2, 'A')),
                 ((1, 'C'), (2, 'B')),
                 ((1, 'C'), (2, 'D'))]
contact_X = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                       0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                       0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])

sequence_X = np.array([[1, 0, 0, 1, 0, 0, 1, 0],
                       [0, 1, 0, 1, 0, 0, 0, 1],
                       [0, 1, 0, 0, 1, 1, 0, 0]])
sequence_terms = [(0, 'A'), (0, 'B'), (0, 'C'), (1, 'A'),
                  (1, 'C'), (2, 'A'), (2, 'B'), (2, 'D')]

all_X = np.array([[1, 0, 0, 1, 0, 0],
                  [0, 1, 0, 1, 0, 1],
                  [0, 1, 0, 0, 1, 0]])
complete_X = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
all_terms = [[(0, 'A'), (2, 'B'), ((0, 'A'), (1, 'A')),
              ((0, 'A'), (2, 'B')), ((1, 'A'), (2, 'B'))],
             [(0, 'B')],
             [(0, 'C'), ((0, 'A'), (1, 'C')),
              ((0, 'C'), (1, 'A')),
              ((0, 'C'), (1, 'C')), ((0, 'A'), (2, 'A')),
              ((0, 'A'), (2, 'D')), ((0, 'B'), (2, 'B')),
              ((0, 'C'), (2, 'A')), ((0, 'C'), (2, 'B')),
              ((0, 'C'), (2, 'D')), ((1, 'A'), (2, 'A')),
              ((1, 'C'), (2, 'B')), ((1, 'C'), (2, 'D'))],
             [(1, 'A')],
             [(1, 'C'), (2, 'A'), ((0, 'B'), (1, 'C')),
              ((0, 'B'), (2, 'A')), ((1, 'C'), (2, 'A'))],
             [(2, 'D'), ((0, 'B'), (1, 'A')),
              ((0, 'B'), (2, 'D')), ((1, 'A'), (2, 'D'))]]
seqs = ['AAB', 'BAD', 'BCA']
assignments = {0:0, 1:0, 2:1}


def test_contacting_terms():
    made_terms = contacting_terms(sample_space, contacts)
    for a, b in zip(made_terms, contact_terms):
        assert a == b


def test_sequence_terms():
    assert sequence_terms == make_sequence_terms(sample_space)


def test_contact_X():
    X, terms = make_contact_X(seqs, sample_space, contacts)
    assert np.array_equal(X, contact_X)
    assert terms == contact_terms


def test_sequence_X():
    X, terms = make_sequence_X(seqs, sample_space)
    assert np.array_equal(X, sequence_X)
    assert terms == sequence_terms


def test_in_sequence():
    assert in_sequence(seqs[0], contact_terms[0])
    assert ~in_sequence(seqs[0], contact_terms[1])
    assert in_sequence(seqs[0], sequence_terms[0])
    assert ~in_sequence(seqs[0], sequence_terms[1])


def test_present():
    assert present('AAB', (1, 'A'))
    assert ~present('AAB', (1, 'B'))


def test_X():
    X, terms = make_X(seqs, sample_space, contacts)
    assert np.array_equal(X, all_X)
    assert terms == all_terms
    X_2, terms_2 = make_X(seqs, sample_space, contacts, terms)
    assert np.array_equal(X_2, X)
    assert terms == terms_2
    new_seqs = ['CCD', 'CCA']
    new_X = np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    X, _ = make_X(new_seqs, sample_space, contacts, terms)
    assert np.array_equal(X, new_X)
    X, terms = make_X(seqs, sample_space, contacts, collapse=False)
    assert np.array_equal(X, complete_X)
    assert terms == sequence_terms + contact_terms
    X, terms = make_X(seqs, terms=terms)
    assert np.array_equal(X, complete_X)


def test_contacts():
    terms = get_contacts(seqs[0], contacts)
    assert terms == [((0, 'A'), (1, 'A')), ((0, 'A'), (2, 'B')),
                     ((1, 'A'), (2, 'B'))]


def test_terms():
    terms = get_terms(seqs[0])
    assert terms == [(0, 'A'), (1, 'A'), (2, 'B')]


def test_sequence():
    seq = make_sequence('02', assignments, sample_space)
    assert seq == ['A', 'A', 'D']
    seq = substitute_blocks('AAD', [(2, 0)], assignments, sample_space)
    assert seq == 'CCD'


def test_loads():
    with open('gpmodel/test/data/assignment_dict.pkl', 'rb') as f:
        real_dict = pickle.load(f)
    assert load_assignments('gpmodel/test/data/nlibrary.output') == real_dict
    assert make_name_dict('gpmodel/test/data/test_dict.xlsx') == \
        {'first':'012', 'second':'210', 'third':'102'}


def test_zeroing():
    assert zero_index('21123') == '10012'



if __name__=="__main__":
    test_contacting_terms()
    test_sequence_terms()
    test_contact_X()
    test_sequence_X()
    test_present()
    test_in_sequence()
    test_X()
    test_contacts()
    test_terms()
    test_sequence()
    test_zeroing()
    test_loads()
