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

X = np.array([[1.0, 2.0], [3.0, -1.0], [2.0, -2.0]])
X_df = pd.DataFrame(X, index=['A','B','C'])

def test_gpkernel():
    """ Test the base class GPKernel. """
    print 'Testing GPKernel...'
    kernel = gpkernel.GPKernel()
    assert kernel._saved_X.keys() == []
    assert kernel.hypers == []
    kernel.set_X(X_df)
    assert np.isclose(kernel._saved_X['A'], X[0]).all()
    kernel.delete(X_df.loc[['A']])
    assert sorted(kernel._saved_X.keys()) == ['B', 'C']
    assert np.isclose(kernel._get_X('B'), X[1]).all()
    assert np.isclose(kernel._get_X(X[2]), X[2]).all()
    print 'GPKernel passes all tests.'

def test_se_kernel():
    """ Tests for the squared exponential kernels. """
    print 'Testing SEKernel...'
    # Test __init__
    kern = gpkernel.SEKernel()
    assert kern.hypers == ['sigma_f', 'ell']

    # test on single values
    first = 4
    second = 2
    params = [0.9, 0.5]
    sigma_f, ell = params
    r_squared = (first - second)**2
    actual = sigma_f**2 * np.exp(-0.5 * r_squared / ell**2)

    assert kern.calc_kernel(first, second, params) == actual
    assert kern._distance(first, second) == np.linalg.norm(first-second)**2

    xa = np.array([2.0, 3.0, 4.0])
    xb = np.array([1.0, 3.0, 5.0])
    xc = np.array([-2.0, 0.0, 2.0])

    Xs = pd.DataFrame ([xa, xb, xc])
    Xs.index = ['A', 'B', 'C']
    actual_ds = np.array([[np.linalg.norm(i-j)**2 for i in [xa, xb, xc]] for
                 j in [xa, xb, xc]])
    actual = sigma_f**2 * np.exp(-0.5 * actual_ds / ell**2)

    # test get_d and _distance
    assert kern._distance(xa, xb) == np.linalg.norm(xa - xb)**2

    # test train, set_X
    kern.set_X(Xs)
    assert sorted(kern._saved_X.keys()) == ['A', 'B', 'C']
    assert np.isclose(kern._saved_X['A'], xa).all()
    assert np.isclose(kern._d_squared, actual_ds).all()

    # test _squared_exponential
    assert np.isclose(kern._squared_exponential(actual_ds, params), actual).all()
    assert np.isclose(kern._squared_exponential(np.linalg.norm(xa-xb)**2, params),
                      actual[0,1])

    # test calculate_kernel and make_K
    assert np.isclose(kern.make_K(Xs, hypers=params), actual).all()
    assert np.isclose(kern.make_K(hypers=params), actual).all()
    assert kern.calc_kernel(xa, xb, params) == actual[0,1]
    assert kern.calc_kernel('A', xb, params) == actual[0,1]


    print 'SEKernel passes all tests.'

def test_matern_kernel():
    """ Tests for the matern kernels. """
    print 'Testing MaternKernel...'

    # Test __init__
    kern1 = gpkernel.MaternKernel(nu='3/2')
    kern2 = gpkernel.MaternKernel(nu='5/2')
    assert kern1.hypers == ['ell']
    assert kern1.nu == '3/2'
    assert kern2.nu == '5/2'

    # test on single values
    first = 4
    second = 2
    params = [0.9]
    ell = params[0]
    r = np.sqrt((first - second)**2)
    actual1 = (1 + np.sqrt(3.0) / ell * r) * np.exp(-np.sqrt(3.0) * r / ell)
    actual2 = (1 + np.sqrt(5.0) / ell * r + 5.0*r**2/3/ell**2) *\
    np.exp(-np.sqrt(5.0) * r / ell)

    assert kern1.calc_kernel(first, second, params) == actual1
    assert kern2.calc_kernel(first, second, params) == actual2
    assert kern2._distance(first, second) == np.linalg.norm(first-second)

    xa = np.array([2.0, 3.0, 4.0])
    xb = np.array([1.0, 3.0, 5.0])
    xc = np.array([-2.0, 0.0, 2.0])

    Xs = pd.DataFrame ([xa, xb, xc])
    Xs.index = ['A', 'B', 'C']
    actual_ds = np.array([[np.linalg.norm(i-j) for i in [xa, xb, xc]] for
                 j in [xa, xb, xc]])
    actual1 = (1 + np.sqrt(3.0) / ell * actual_ds) *\
    np.exp(-np.sqrt(3.0) * actual_ds / ell)
    actual2 = (1 + np.sqrt(5.0) / ell * actual_ds + 5.0*actual_ds**2/3/ell**2) *\
    np.exp(-np.sqrt(5.0) * actual_ds / ell)

    # test get_d and _distance
    assert kern1._distance(xa, xb) == np.linalg.norm(xa - xb)

    # test train, set_X
    kern1.set_X(Xs)
    kern2.set_X(Xs)
    assert sorted(kern1._saved_X.keys()) == ['A', 'B', 'C']
    assert np.isclose(kern1._saved_X['A'], xa).all()
    assert np.isclose(kern1._d, actual_ds).all()

    # test _matern
    assert np.isclose(kern1._matern(actual_ds, params), actual1).all()
    assert np.isclose(kern2._matern(actual_ds, params), actual2).all()
    assert np.isclose(kern1._matern(np.linalg.norm(xa-xb), params),
                      actual1[0,1])
    assert np.isclose(kern2._matern(np.linalg.norm(xa-xb), params),
                      actual2[0,1])

    # test calculate_kernel and make_K
    assert np.isclose(kern1.make_K(Xs, hypers=params), actual1).all()
    assert np.isclose(kern2.make_K(Xs, hypers=params), actual2).all()
    assert np.isclose(kern1.make_K(hypers=params), actual1).all()
    assert kern1.calc_kernel(xa, xb, params) == actual1[0,1]
    assert kern2.calc_kernel('A', xb, params) == actual2[0,1]


    print 'MaternKernel passes all tests.'

def test_hamming_kernel():
    """
    Tests for the HammingKernel
    """
    vp = 0.4
    K = pd.DataFrame([[4.,2.,3.,4.],
                      [2.,4.,3.,2.],
                      [3.,3.,4.,3.],
                      [4.,2.,3.,4.]],
                    index=seqs.index,
                    columns=seqs.index)
    norm = 4.0
    print 'Testing HammingKernel...'
    kern = gpkernel.HammingKernel()
    assert kern.hypers == ['var_p']

    assert kern.calc_kernel(seqs.iloc[0],seqs.iloc[1],normalize=False) == 2,\
    'Failed calc_kernel.'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2],normalize=False) == 3,\
    'Failed calc_kernel'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2],
                            hypers=[vp],normalize=False) == 3*vp,\
    'Failed calc_kernel for var_p ~= 1'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2],
                            hypers=[vp], normalize=True) == 3*vp/norm,\
    'Failed calc_kernel for var_p ~= 1, normalized'

    assert np.isclose(kern.make_K(seqs, normalize=False),K).all(),\
    'Failed make_K for var_p = 1'
    assert np.isclose(kern.make_K(seqs,hypers=[vp],
                                  normalize=False), K*vp).all(),\
    'Failed make_K for var_p ~= 1.'
    assert np.isclose(kern.make_K(seqs, normalize=True,
                                  hypers=[vp]),K/norm*vp).all(),\
    'Failed make_K with normalization for var_p ~= 1.'

    # now let's make sure we can train it and use keys to access functions
    kern.train(seqs)
    assert kern._saved_X == {'A':'RYMA',
                            'B':'RTHA',
                            'C':'RTMA',
                            'D':'RYMA'},\
    'Failed to train HammingKernel.'

    kern.delete(seqs.loc[['D']])
    assert kern._saved_X == {'A':'RYMA',
                            'B':'RTHA',
                            'C':'RTMA'}
    assert kern.calc_kernel('A','B', normalize=False) == 2,\
    'Failed calc_kernel with two trained sequences.'
    assert kern.calc_kernel('C','B',hypers=[vp], normalize=False) == 3*vp,\
    'Failed calc_kernel with vp ~= 1.'
    assert kern.calc_kernel('B',seqs.iloc[0],normalize=False) == 2,\
    'Failed calc_kernel with one untrained and one trained sequence.'
    assert kern.calc_kernel('C','C',hypers=[vp], normalize=True) == 4*vp/norm,\
    'Failed calc_kernel with normalization.'

    kern.set_X(seqs)
    assert np.isclose(kern._base_K, K/norm).all()
    assert np.isclose(kern.make_K(hypers=[vp]),K/norm*vp).all(),\
    'Failed make_K using saved base_K'
    print 'HammingKernel passes all tests.'

def test_structure_kernel():
    # test with repeats
    vp = 0.4
    K = pd.DataFrame([[2.0,0.0,1.0,2.0],
                      [0.0,2.0,1.0,0.0],
                      [1.0,1.0,2.0,1.0],
                      [2.0,0.0,1.0,2.0]],
                     index=seqs.index,
                     columns=seqs.index)
    norm = 2.0
    print 'Testing StructureKernel...'
    kern = gpkernel.StructureKernel(contacts)
    assert kern.hypers == ['var_p']


    # now let's make sure we can train it and use keys to access functions
    kern.train(seqs)
    assert kern._saved_X == {'A': [((0, 'R'), (1, 'Y')), ((2, 'M'), (3, 'A'))],
                      'C': [((0, 'R'), (1, 'T')), ((2, 'M'), (3, 'A'))],
                      'B': [((0, 'R'), (1, 'T')), ((2, 'H'), (3, 'A'))],
                      'D': [((0, 'R'), (1, 'Y')), ((2, 'M'), (3, 'A'))]},\
    'Failed to train structure kernel.'

    kern.delete(seqs.loc[['D']])
    assert kern._saved_X == {'A': [((0, 'R'), (1, 'Y')), ((2, 'M'), (3, 'A'))],
                            'C': [((0, 'R'), (1, 'T')), ((2, 'M'), (3, 'A'))],
                            'B': [((0, 'R'), (1, 'T')), ((2, 'H'), (3, 'A'))]}

    kern.set_X(seqs)
    assert np.isclose(kern._base_K, K/norm).all()


    assert kern.calc_kernel(seqs.iloc[0],seqs.iloc[1]) == 0,\
    'Failed calc_kernel for sequences with no shared contacts.'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2],normalize=False) == 1,\
    'Failed calc_kernel for sequences with shared contacts.'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2],
                            hypers=[vp],normalize=False) == vp,\
    'Failed calc_kernel for var_p ~= 1.'

    # test make_K
    assert np.isclose(kern.make_K(seqs, normalize=False), K).all(),\
    'Failed make_K for var_p = 1.'
    assert np.isclose(kern.make_K(seqs, normalize=True), K/norm).all(),\
    'Failed make_K with normalization for var_p = 1.'
    assert np.isclose(kern.make_K(seqs, hypers=[vp], normalize=True), K/norm*vp).all(),\
    'Failed make_K with normalization for var_p ~= 1.'
    assert np.isclose(kern.make_K(seqs,hypers=[vp], normalize=False), K*vp).all(),\
    'Failed make_K for var_p ~= 1.'
    assert np.isclose(kern.make_K(hypers=[vp]), K/norm*vp).all(),\
    'Failed make_K using saved base_K'


    assert kern.calc_kernel('A','B',normalize=False) == 0,\
    'Failed calc_kernel with two trained sequences.'
    assert kern.calc_kernel('A',seqs.iloc[1],normalize=False) == 0,\
    'Failed calc_kernel with trained, untrained sequences.'
    assert kern.calc_kernel(seqs.iloc[2],'B', hypers=[vp],normalize=False) == vp,\
    'Failed calc_kernel with untrained, trained sequences.'
    assert kern.calc_kernel('C','C',hypers=[vp], normalize=True) == 2*vp/norm,\
    'Failed calc_kernel with normalization.'

    print 'StructureKernel passes all tests.'

def test_se(kern):
    # test squared exponential function
    print 'Testing squared exponential functions...'
    # test dimension check

    # test on single values
    first = 4
    second = 2
    params = [0.2, 2]
    sigma_f, ell = params
    actual = sigma_f**2 * np.exp(-0.5*(first-second)**2/ell**2)
    assert kern.se([first, second],params) == actual,\
        'kern.se failed for 1-dimensional, 2 sample, row case.'
    assert kern.se(np.array([[first], [second]]),params) == actual,\
        'kern.se failed for 1-dimensional, 2 sample , column case.'
    # try it with matrices
    assert kern.se(np.matrix([[first], [second]]),params) == actual,\
        'kern.se failed for 1-dimensional, 2 sample , column matrix case.'
    assert kern.se(np.matrix([first, second]),params) == actual,\
        'kern.se failed for 1-dimensional, 2 sample, row matrix case.'
    # Try it with DataFrames
    df = pd.DataFrame(np.array([[first], [second]]), index=['A', 'B'])
    assert kern.se(df, params) == actual,\
        'kern.se failed for 1-dimensional, 2 sample, DataFrame.'

    xa = np.array([2.0, 3.0, 4.0])
    xb = np.array([1.0, 3.0, 5.0])
    xc = np.array([-2.0, 0.0, 2.0])

    # test on numpy.ndarray with 3 measurements, 1 dimension
    d_squared = np.empty((3,3))
    for i in range(3):
        for j in range(3):
            d_squared[i][j] = (xa[i] - xa[j])**2
    actual = sigma_f**2 * np.exp(-d_squared*0.5/ell**2)
    from_kernel = kern.se(xa, params)
    assert np.array_equal(actual, from_kernel), \
        'kern.se failed for 1-dimensional, 3 sample, row case.'
    x_columns = np.array([[x] for x in xa])
    from_kernel = kern.se(x_columns, params)
    assert np.array_equal(actual, from_kernel), \
        'kern.se failed for 1-dimensional, 3 sample, column case.'
    # with matrices
    from_kernel = kern.se(np.matrix(xa), params)
    assert np.array_equal(actual, from_kernel), \
        'kern.se failed for 1-dimensional, 3 sample, row matrix case.'
    from_kernel = kern.se(np.matrix(x_columns), params)
    assert np.array_equal(actual, from_kernel), \
        'kern.se failed for 1-dimensional, 3 sample, column matrix case.'
    # With DataFrame
    inds = ['A','B','C']
    df = pd.DataFrame(xa, index=inds)
    from_kernel = kern.se(df, params)
    actual = pd.DataFrame(actual, index=inds, columns=inds)
    assert actual.equals(from_kernel), \
        'kern.se failed for 1-dimensional, 3 sample, DataFrame case.'



    # test on numpy.ndarray with 2 measurements, 3 dimensions
    xs = np.concatenate((xa, xb))
    xs = xs.reshape((2,3))
    d_squared = sum([(x1-x2)**2 for x1, x2 in zip(xs[0], xs[1])])
    actual = sigma_f**2 * np.exp(-d_squared*0.5/ell**2)
    from_kernel = kern.se(xs, params)
    assert actual == from_kernel, \
        'kern.se failed for 3-dimensional, 2 sample case.'
    # with a matrix
    from_kernel = kern.se(np.matrix(xs), params)
    assert actual == from_kernel, \
        'kern.se failed for 3-dimensional, 2 sample, matrix case.'
    # with a DataFrame
    from_kernel = kern.se(pd.DataFrame(xs), params)
    assert actual == from_kernel, \
        'kern.se failed for 3-dimensional, 2 sample, DataFrame case.'

    # test on numpy.ndarray with 3 measurements, 3 dimensions, constant ells
    xs = np.concatenate((xa, xb, xc))
    xs = xs.reshape((3,3))
    d_squared = np.empty((3,3))
    for i in range(3):
        for j in range(3):
            d_squared[i][j] = sum([(x1-x2)**2 for x1, x2 in zip(xs[i], xs[j])])
    actual = sigma_f**2 * np.exp(-d_squared*0.5/ell**2)
    from_kernel = kern.se(xs, params)
    assert np.allclose(actual,from_kernel), \
        'kern.se failed for 3-dimensional, 3 sample case.'
    # with a matrix
    from_kernel = kern.se(np.matrix(xs), params)
    assert np.allclose(actual,from_kernel), \
        'kern.se failed for 3-dimensional, 3 sample matrix case.'
    # with a DataFrame
    df = pd.DataFrame(xs, index=inds)
    actual = pd.DataFrame(actual, index=inds, columns=inds)
    from_kernel = kern.se(df, params)
    assert np.allclose(np.array(from_kernel), np.array(actual)), \
        'kern.se failed for 3-dimensional, 3 sample DataFrame case.'
    assert all([a==f for a, f in zip(actual.index, from_kernel.index)]), \
        'kern.se failed for 3-dimensional, 3 sample DataFrame case.'
    assert all([a==f for a, f in zip(actual.columns, from_kernel.columns)]), \
        'kern.se failed for 3-dimensional, 3 sample DataFrame case.'

    # test dist_to_se
    dists = np.array ([[1, 3, 2],
                       [3, 1, 0],
                       [2, 0, 1]])
    from_kernel = kern.dist_to_se(dists, params)
    actual = sigma_f**2 * np.exp(dists**2 * -0.5/ell**2)
    assert np.array_equal(actual, from_kernel), \
        'kern.dist_to_se failed.'
    # with a DataFrame
    dists = pd.DataFrame(dists, index=inds, columns=inds)
    actual = pd.DataFrame(actual, index=inds, columns=inds)
    from_kernel = kern.dist_to_se(dists, params)
    assert from_kernel.equals(actual), \
        'kern.dist_to_se failed with a DataFrame.'

    print 'Squared exponential functions pass all tests.'

def test_protein_matern_kernels():
    """ Test StructureMaternKernel and HammingMaternKernel. """
    print 'Testing StructureMaternKernel...'

    ssek1 = gpkernel.StructureMaternKernel(contacts, '3/2')
    ssek2 = gpkernel.StructureMaternKernel(contacts, '5/2')
    assert ssek1.hypers == ['ell']
    K = np.array([[2.0,0.0,1.0,2.0],
                  [0.0,2.0,1.0,0.0],
                  [1.0,1.0,2.0,1.0],
                  [2.0,0.0,1.0,2.0]])
    r = np.sqrt(2 - K)
    params = [0.5]
    ell = params[0]
    actual1 = (1 + np.sqrt(3.0) / ell * r) * np.exp(-np.sqrt(3.0) * r / ell)
    actual2 = (1 + np.sqrt(5.0) / ell * r + 5.0*r**2/3/ell**2) *\
    np.exp(-np.sqrt(5.0) * r / ell)

    # test distance and make_D functions
    assert ssek1._distance(seqs.loc['A'], seqs.loc['B']) == np.sqrt(2),\
        'StructureMaternKernel fails to calculate distance.'
    assert np.isclose(ssek2._get_d(seqs), r).all(), \
        'StructureMaternKernel fails to D.'

    # Test make_K and calc_kernel
    assert np.isclose(ssek1.make_K(seqs, params),
                      actual1).all(),\
        'StructureMaternKernel fails make_K.'
    assert ssek1.calc_kernel(seqs.loc['A'], seqs.loc['B'], params) == \
    actual1[0,1], 'StructureMaternKernel fails calc_kernel.'
    assert np.isclose(ssek2.make_K(seqs, params),
                      actual2).all(),\
        'StructureMaternKernel fails make_K.'
    assert ssek2.calc_kernel(seqs.loc['A'], seqs.loc['B'], params) == \
    actual2[0,1], 'StructureMaternKernel fails calc_kernel.'

    # Set a d_squared to speed things up
    ssek1.set_X(seqs)
    assert ssek1.calc_kernel('A', 'B', params) == actual1[0,1]
    assert np.isclose(ssek1._d, r).all(), 'ssek._d is wrong.'
    # Retest make_K
    assert np.isclose(ssek1.make_K(hypers=params), actual1).all(),\
        'StructureMaternKernel fails make_K.'

    print 'StructureMaternKernel passes all tests.'

    print 'Testing HammingMaternKernel...'

    ssek1 = gpkernel.HammingMaternKernel('3/2')
    ssek2 = gpkernel.HammingMaternKernel('5/2')
    assert ssek1.hypers == ['ell']
    K = np.array([[4.,2.,3.,4.],
                  [2.,4.,3.,2.],
                  [3.,3.,4.,3.],
                  [4.,2.,3.,4.]])
    r = np.sqrt(4 - K)
    params = [0.5]
    ell = params[0]
    actual1 = (1 + np.sqrt(3.0) / ell * r) * np.exp(-np.sqrt(3.0) * r / ell)
    actual2 = (1 + np.sqrt(5.0) / ell * r + 5.0*r**2/3/ell**2) *\
    np.exp(-np.sqrt(5.0) * r / ell)

    # test distance and make_D functions
    assert ssek1._distance(seqs.loc['A'], seqs.loc['B']) == np.sqrt(2),\
        'HammingMaternKernel fails to calculate distance.'
    assert np.isclose(ssek2._get_d(seqs), r).all(), \
        'HammingMaternKernel fails to D.'

    # Test make_K and calc_kernel
    assert np.isclose(ssek1.make_K(seqs, params),
                      actual1).all(),\
        'HammingMaternKernel fails make_K.'
    assert ssek1.calc_kernel(seqs.loc['A'], seqs.loc['B'], params) == \
    actual1[0,1], 'HammingMaternKernel fails calc_kernel.'
    assert np.isclose(ssek2.make_K(seqs, params),
                      actual2).all(),\
        'HammingMaternKernel fails make_K.'
    assert ssek2.calc_kernel(seqs.loc['A'], seqs.loc['B'], params) == \
    actual2[0,1], 'HammingMaternKernel fails calc_kernel.'

    # Set a d_squared to speed things up
    ssek1.set_X(seqs)
    assert ssek1.calc_kernel('A', 'B', params) == actual1[0,1]
    assert np.isclose(ssek1._d, r).all(), 'ssek._d is wrong.'
    # Retest make_K
    assert np.isclose(ssek1.make_K(hypers=params), actual1).all(),\
        'HammingMaternKernel fails make_K.'

    print 'HammingMaternKernel passes all tests.'

def test_protein_se_kernels():
    """ Test StructureSEKernel and HammingSEKernel. """

    print 'Testing StructureSEKernel...'

    ssek = gpkernel.StructureSEKernel(contacts)
    assert ssek.hypers == ['sigma_f', 'ell']
    K = np.array([[2.0,0.0,1.0,2.0],
                      [0.0,2.0,1.0,0.0],
                      [1.0,1.0,2.0,1.0],
                      [2.0,0.0,1.0,2.0]])
    d2 = 2 - K
    params = [0.5, 0.6]
    sigma_f, ell = params
    actual = sigma_f**2 * np.exp(-0.5 * d2 / ell**2)

    # test distance and make_D functions
    assert ssek._distance(seqs.loc['A'], seqs.loc['B']) == 2,\
        'StructureSEKernel fails to calculate distance.'
    assert np.isclose(ssek._get_d_squared(seqs), d2).all(), \
        'StructureSEKernel fails to D.'

    # Test make_K and calc_kernel
    assert np.isclose(ssek.make_K(seqs, params),
                      actual).all(),\
        'StructureSEKernel fails make_K.'
    assert ssek.calc_kernel(seqs.loc['A'], seqs.loc['B'], params) == \
    actual[0,1], 'StructureSEKernel fails calc_kernel.'

    # Set a d_squared to speed things up
    ssek.set_X(seqs)
    assert ssek.calc_kernel('A', 'B', params) == actual[0,1]
    assert np.isclose(ssek._d_squared, d2).all(), 'ssek.d_squared is wrong.'
    # Retest make_K
    assert np.isclose(ssek.make_K(hypers=params), actual).all(),\
        'StructureSEKernel fails make_K.'

    print 'StructureSEKernel passes all tests.'


    print 'Testing HammingSEKernel...'

    ssek = gpkernel.HammingSEKernel()

    assert ssek.hypers == ['sigma_f', 'ell']


    K = np.array([[4.,2.,3.,4.],
                  [2.,4.,3.,2.],
                  [3.,3.,4.,3.],
                  [4.,2.,3.,4.]])
    d2 = 4 - K
    actual = sigma_f**2 * np.exp(-0.5 * d2 / ell**2)

    # test distance and make_D functions
    assert ssek._distance(seqs.loc['A'], seqs.loc['B']) == 2,\
        'HammingSEKernel fails to calculate distance.'
    assert np.isclose(ssek._get_d_squared(seqs), d2).all(), \
        'HammingSEKernel fails to D.'

    # Test make_K and calc_kernel
    assert np.isclose(ssek.make_K(seqs, params),
                      actual).all(),\
        'HammingSEKernel fails make_K.'
    assert ssek.calc_kernel(seqs.iloc[0], seqs.iloc[1], params) == \
    actual[0,1], 'HammingSEKernel fails calc_kernel.'

    # Set a d_squared to speed things up
    ssek.set_X(seqs)
    assert np.isclose(ssek._d_squared, d2).all(), 'ssek.d_squared is wrong.'
    # Retest make_K
    assert np.isclose(ssek.make_K(hypers=params), actual).all(),\
        'HammingSEKernel fails make_K.'

    print 'HammingSEKernel passes all tests.'

def test_linear_kernel():
    print 'Testing LinearKernel...'
    kern = gpkernel.LinearKernel()
    X = pd.DataFrame([[4.,2.,3.,4.],
                      [2.,0.,1.,2.],
                      [1.,1.,4.,3.],
                      [-3.,2.,3.,0.]],
                     index=seqs.index)
    K = pd.DataFrame([[45.,19.,30.,1.],
                      [19.,9.,12.,-3.],
                      [30.,12.,27.,11.],
                      [1.,-3.,11.,22.]],
                     index=seqs.index,
                    columns=seqs.index)

    assert kern.hypers == ['var_p']


    # now let's make sure we can train it and use keys to access functions
    kern.train(X)
    saved_X = {i:np.array(X.loc[i]) for i in X.index}

    kern.set_X(X)
    assert kern._base_K.equals(K)


    assert kern.calc_kernel(X.iloc[0],X.iloc[1]) == 19,\
    'Failed calc_kernel for vp=1.'
    assert kern.calc_kernel(X.iloc[0],X.iloc[1],
                            hypers=[2]) == 38,\
    'Failed calc_kernel for var_p ~= 1.'

    # test make_K
    assert kern.make_K(X).equals(K),\
    'Failed make_K for var_p = 1.'
    assert kern.make_K(X, hypers=[2]).equals(K*2),\
    'Failed make_K for var_p ~= 1.'
    assert kern.make_K(hypers=[2]).equals(K*2),\
    'Failed make_K using saved base_K'


    assert kern.calc_kernel('A','B') == 19,\
    'Failed calc_kernel with two trained sequences.'
    assert kern.calc_kernel('A',X.iloc[1]) == 19,\
    'Failed calc_kernel with trained, untrained sequences.'
    assert kern.calc_kernel(X.iloc[2],'B') == 12,\
    'Failed calc_kernel with untrained, trained sequences.'


    print 'LinearKernel passes all tests.'


def test_sum_kernel():
    """ Test SumKernel. """
    print 'Testing SumKernel...'
    kern52 = gpkernel.MaternKernel('5/2')
    kernSE = gpkernel.SEKernel()
    kern32 = gpkernel.MaternKernel('3/2')
    kernels = [kern52, kernSE, kern32]
    h1 = [0.1]
    h2 = [1.0, 0.9]
    h3 = [0.8]
    params = h1 + h2 + h3
    actual_K = kern52.make_K(X_df, h1) + kernSE.make_K(X_df, h2) + \
        kern32.make_K(X_df, h3)
    kernel = gpkernel.SumKernel(kernels)
    assert kernel._kernels == [kern52, kernSE, kern32]
    assert kernel.hypers == ['ell0', 'sigma_f0', 'ell1', 'ell2']
    assert np.isclose(kernel.make_K(X_df, params), actual_K).all()
    assert kernel.calc_kernel(X[0], X[1], params) == actual_K[0,1]
    kernel.set_X(X_df)
    for k1, k2 in zip(kernel._kernels, kernels):
        assert np.isclose(k1._saved_X['A'], X[0]).all()
    kernel.delete(X_df.loc[['A']])
    for k in kernel._kernels:
        assert sorted(k._saved_X.keys()) == ['B', 'C']
    print "SumKernel passes all tests."

if __name__=="__main__":
    test_gpkernel()
    test_matern_kernel()
    test_linear_kernel()
    test_hamming_kernel()
    test_structure_kernel()
    test_se_kernel()
    test_protein_se_kernels()
    test_protein_matern_kernels()
    test_sum_kernel()
