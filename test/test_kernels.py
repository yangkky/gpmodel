import sys
sys.path.append ('/Users/seinchin/Documents/Caltech/Arnold Lab/Programming tools/GPModel')
import gpkernel
import pandas as pd
import numpy as np

seqs = pd.DataFrame([['R','Y','M','A'],['R','T','H','A'], ['R','T','M','A']],
                    index=['A','B','C'], columns=[0,1,2,3])

seqs = seqs.append(seqs.iloc[0])
seqs.index = ['A','B','C','D']
space = [('R'), ('Y', 'T'), ('M', 'H'), ('A')]
contacts = [(0,1),(2,3)]


def test_hamming_kernel():
    """
    Tests for the HammingKernel
    """
    vp = 0.4
    K = pd.DataFrame([[4.,2.,3.,4.],[2.,4.,3.,2.],[3.,3.,4.,3.],[4.,2.,3.,4.]],
                    index=seqs.index,
                    columns=seqs.index)
    print 'Testing HammingKernel...'
    kern = gpkernel.HammingKernel()

    assert kern.calc_kernel(seqs.iloc[0],seqs.iloc[1]) == 2,\
    'Failed calc_kernel.'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2]) == 3,\
    'Failed calc_kernel'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2],var_p=vp) == 3*vp,\
    'Failed calc_kernel for var_p ~= 1'

    assert kern.make_K(seqs).equals(K),\
    'Failed make_K for var_p = 1'
    assert kern.make_K(seqs,var_p=vp).equals(K*vp),\
    'Failed make_K for var_p ~= 1.'

    # now let's make sure we can train it and use keys to access functions
    kern.train(seqs)
    assert kern.saved_seqs == {'A':'RYMA',
                               'B':'RTHA',
                               'C':'RTMA',
                               'D':'RYMA'},\
    'Failed to train HammingKernel.'
    assert kern.calc_kernel('A','B') == 2,\
    'Failed calc_kernel with two trained sequences.'
    assert kern.calc_kernel('C','D',var_p=vp) == 3*vp,\
    'Failed calc_kernel with vp ~= 1.'
    assert kern.calc_kernel('B',seqs.iloc[0]) == 2,\
    'Failed calc_kernel with one untrained and one trained sequence.'

    print 'HammingKernel passes all tests.'




def test_structure_kernel():
    # test with repeats
    vp = 0.4
    K = pd.DataFrame([[2,0,1,2],[0,2,1,0],[1,1,2,1],[2,0,1,2]],
                     index=seqs.index,
                     columns=seqs.index)
    print 'Testing StructureKernel...'
    kern = gpkernel.StructureKernel(contacts, space)
    assert kern.contact_terms == [((0, 'R'), (1, 'Y')),
                                  ((0, 'R'), (1, 'T')),
                                  ((2, 'H'), (3, 'A')),
                                  ((2, 'M'), (3, 'A'))],\
    'Failed to make correct contact_terms.'
    assert kern.calc_kernel(seqs.iloc[0],seqs.iloc[1]) == 0,\
    'Failed calc_kernel for sequences with no shared contacts.'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2]) == 1,\
    'Failed calc_kernel for sequences with shared contacts.'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2],var_p=vp) == vp,\
    'Failed calc_kernel for var_p ~= 1.'
    assert kern.contacts_X_row(seqs.iloc[0]) == [1,0,0,1],\
    'Failed contacts_X_row for var_p = 1.'
    assert kern.contacts_X_row(seqs.iloc[0],var_p=vp) == [vp,0,0,vp],\
    'Failed contacts_X_row for var_p ~= 1.'
    # test make_contacts_X
    assert kern.make_contacts_X(seqs,1) == [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1],[1,0,0,1]],\
    'Failed make_contacts_X for var_p = 1.'
    assert kern.make_contacts_X(seqs,vp) == [[vp, 0, 0, vp], [0, vp, vp, 0], [0, vp, 0, vp],[vp,0,0,vp]],\
    'Failed make_contacts_X for var_p ~= 1.'
    assert kern.make_K(seqs).equals(K),\
    'Failed make_K for var_p = 1.'
    assert kern.make_K(seqs,var_p=vp).equals(K*vp),\
    'Failed make_K for var_p ~= 1.'


    # now let's make sure we can train it and use keys to access functions
    kern.train(seqs)
    assert kern.saved_contacts == {'A': [((0, 'R'), (1, 'Y')), ((2, 'M'), (3, 'A'))],
                             'C': [((0, 'R'), (1, 'T')), ((2, 'M'), (3, 'A'))],
                             'B': [((0, 'R'), (1, 'T')), ((2, 'H'), (3, 'A'))],
                             'D': [((0, 'R'), (1, 'Y')), ((2, 'M'), (3, 'A'))]},\
    'Failed to train structure kernel.'
    assert kern.calc_kernel('A','B') == 0,\
    'Failed calc_kernel with two trained sequences.'
    assert kern.calc_kernel('A',seqs.iloc[1]) == 0,\
    'Failed calc_kernel with trained, untrained sequences.'
    assert kern.calc_kernel(seqs.iloc[2],'B',var_p=vp) == vp,\
    'Failed calc_kernel with untrained, trained sequences.'


    print 'StructureKernel passes all tests.'







if __name__=="__main__":
    test_hamming_kernel()
    test_structure_kernel()
