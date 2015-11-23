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
    norm = 4.0
    print 'Testing HammingKernel...'
    kern = gpkernel.HammingKernel()

    assert kern.calc_kernel(seqs.iloc[0],seqs.iloc[1]) == 2,\
    'Failed calc_kernel.'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2]) == 3,\
    'Failed calc_kernel'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2],hypers=[vp]) == 3*vp,\
    'Failed calc_kernel for var_p ~= 1'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2],
                            hypers=[vp], normalize=True) == 3*vp/norm,\
    'Failed calc_kernel for var_p ~= 1, normalized'

    assert kern.make_K(seqs).equals(K),\
    'Failed make_K for var_p = 1'
    assert kern.make_K(seqs,hypers=[vp]).equals(K*vp),\
    'Failed make_K for var_p ~= 1.'
    assert kern.make_K(seqs, normalize=True, hypers=[vp]).equals(K/norm*vp),\
    'Failed make_K with normalization for var_p ~= 1.'

    # now let's make sure we can train it and use keys to access functions
    kern.train(seqs)
    assert kern.saved_seqs == {'A':'RYMA',
                               'B':'RTHA',
                               'C':'RTMA',
                               'D':'RYMA'},\
    'Failed to train HammingKernel.'

    kern.delete(seqs.loc[['D']])
    assert kern.saved_seqs == {'A':'RYMA',
                               'B':'RTHA',
                               'C':'RTMA'}
    assert kern.calc_kernel('A','B') == 2,\
    'Failed calc_kernel with two trained sequences.'
    assert kern.calc_kernel('C','B',hypers=[vp]) == 3*vp,\
    'Failed calc_kernel with vp ~= 1.'
    assert kern.calc_kernel('B',seqs.iloc[0]) == 2,\
    'Failed calc_kernel with one untrained and one trained sequence.'
    assert kern.calc_kernel('C','C',hypers=[vp], normalize=True) == 4*vp/norm,\
    'Failed calc_kernel with normalization.'

    print 'HammingKernel passes all tests.'




def test_structure_kernel():
    # test with repeats
    vp = 0.4
    K = pd.DataFrame([[2.0,0.0,1.0,2.0],[0.0,2.0,1.0,0.0],[1.0,1.0,2.0,1.0],[2.0,0.0,1.0,2.0]],
                     index=seqs.index,
                     columns=seqs.index)
    norm = 2.0
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
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2],hypers=[vp]) == vp,\
    'Failed calc_kernel for var_p ~= 1.'
    assert kern.contacts_X_row(seqs.iloc[0]) == [1,0,0,1],\
    'Failed contacts_X_row for var_p = 1.'
    assert kern.contacts_X_row(seqs.iloc[0],hypers=[vp]) == [vp,0,0,vp],\
    'Failed contacts_X_row for var_p ~= 1.'
    # test make_contacts_X
    assert kern.make_contacts_X(seqs,[1.0]) == [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1],[1,0,0,1]],\
    'Failed make_contacts_X for var_p = 1.'
    assert kern.make_contacts_X(seqs,[vp]) == [[vp, 0, 0, vp], [0, vp, vp, 0],
                                               [0, vp, 0, vp],[vp,0,0,vp]],\
    'Failed make_contacts_X for var_p ~= 1.'
    assert kern.make_K(seqs).equals(K),\
    'Failed make_K for var_p = 1.'
    assert kern.make_K(seqs, normalize=True).equals(K/norm),\
    'Failed make_K with normalization for var_p = 1.'
    assert kern.make_K(seqs, hypers=[vp], normalize=True).equals(K/norm*vp),\
    'Failed make_K with normalization for var_p ~= 1.'
    assert kern.make_K(seqs,hypers=[vp]).equals(K*vp),\
    'Failed make_K for var_p ~= 1.'


    # now let's make sure we can train it and use keys to access functions
    kern.train(seqs)
    assert kern.saved_contacts == {'A': [((0, 'R'), (1, 'Y')), ((2, 'M'), (3, 'A'))],
                             'C': [((0, 'R'), (1, 'T')), ((2, 'M'), (3, 'A'))],
                             'B': [((0, 'R'), (1, 'T')), ((2, 'H'), (3, 'A'))],
                             'D': [((0, 'R'), (1, 'Y')), ((2, 'M'), (3, 'A'))]},\
    'Failed to train structure kernel.'

    kern.delete(seqs.loc[['D']])
    assert kern.saved_contacts == {'A': [((0, 'R'), (1, 'Y')), ((2, 'M'), (3, 'A'))],
                             'C': [((0, 'R'), (1, 'T')), ((2, 'M'), (3, 'A'))],
                             'B': [((0, 'R'), (1, 'T')), ((2, 'H'), (3, 'A'))]}

    assert kern.calc_kernel('A','B') == 0,\
    'Failed calc_kernel with two trained sequences.'
    assert kern.calc_kernel('A',seqs.iloc[1]) == 0,\
    'Failed calc_kernel with trained, untrained sequences.'
    assert kern.calc_kernel(seqs.iloc[2],'B', hypers=[vp]) == vp,\
    'Failed calc_kernel with untrained, trained sequences.'
    assert kern.calc_kernel('C','C',hypers=[vp], normalize=True) == 2*vp/norm,\
    'Failed calc_kernel with normalization.'

    print 'StructureKernel passes all tests.'


def test_se():
    # test squared exponential function
    print 'Testing gpmodel.se...'
    # test dimension check

    # test on single values
    first = 4
    second = 2
    params = [0.2, 2]
    sigma_f, ell = params
    actual = sigma_f**2 * np.exp(-0.5*(first-second)**2/ell**2)
    assert gpkernel.se([first, second],params) == actual,\
        'gpkernel.se failed for 1-dimensional, 2 sample, row case.'
    assert gpkernel.se(np.array([[first], [second]]),params) == actual,\
        'gpkernel.se failed for 1-dimensional, 2 sample , column case.'

    # test on numpy.ndarray with 3 measurements, 1 dimension
    xa = np.array([2.0, 3.0, 4.0])
    xb = np.array([1.0, 3.0, 5.0])
    xc = np.array([-2.0, 0.0, 2.0])
    d_squared = np.empty((3,3))
    for i in range(3):
        for j in range(3):
            d_squared[i][j] = (xa[i] - xa[j])**2
    actual = sigma_f**2 * np.exp(-d_squared*0.5/ell**2)
    from_kernel = gpkernel.se(xa, params)
    assert np.array_equal(actual, from_kernel), \
        'gpkernel.se failed for 1-dimensional, 3 sample, row case.'
    x_columns = np.array([[x] for x in xa])
    from_kernel = gpkernel.se(x_columns, params)
    assert np.array_equal(actual, from_kernel), \
        'gpkernel.se failed for 1-dimensional, 3 sample, column case.'


    # test on numpy.ndarray with 2 measurements, 3 dimensions, constant ells
    xs = np.concatenate((xa, xb))
    xs = xs.reshape((2,3))
    d_squared = sum([(x1-x2)**2 for x1, x2 in zip(xs[0], xs[1])])
    actual = sigma_f**2 * np.exp(-d_squared*0.5/ell**2)
    from_kernel = gpkernel.se(xs, params)
    assert actual == from_kernel, \
        'gpkernel.se failed for 3-dimensional, 2 sample case.'

    # test on numpy.ndarray with 3 measurements, 3 dimensions, constant ells
    xs = np.concatenate((xa, xb, xc))
    xs = xs.reshape((3,3))
    d_squared = np.empty((3,3))
    for i in range(3):
        for j in range(3):
            d_squared[i][j] = sum([(x1-x2)**2 for x1, x2 in zip(xs[i], xs[j])])
    actual = sigma_f**2 * np.exp(-d_squared*0.5/ell**2)
    from_kernel = gpkernel.se(xs, params)
    assert np.array_equal(actual,from_kernel), \
        'gpkernel.se failed for 3-dimensional, 3 sample case.'


    # test on numpy.ndarray with multiple rows, multiple columns, multiple ells
    # test on numpy.matrix
    # test with pandas




if __name__=="__main__":
    test_hamming_kernel()
    test_structure_kernel()
    test_se()
