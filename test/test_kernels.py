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


def test_matern_kernel():
    '''
    Tests for the matern kernels.
    '''
    print 'Testing Matern kernels...'

    # test on single values
    kern1 = gpkernel.MaternKernel(nu='3/2')
    kern2 = gpkernel.MaternKernel(nu='5/2')
    first = 4
    second = 2
    params = [0.9]
    ell = params[0]
    r = np.sqrt((first - second)**2)
    actual1 = (1 + np.sqrt(3.0) / ell * r) * np.exp(-np.sqrt(3.0) * r / ell)
    actual2 = (1 + np.sqrt(5.0) / ell * r + 5.0*r**2/3/ell**2) * np.exp(-np.sqrt(5.0) * r / ell)

    assert kern1.calc_kernel(first, second, params) == actual1
    assert kern2.calc_kernel(first, second, params) == actual2

    # Test calc_kernel on vectors
    xa = np.array([2.0, 3.0, 4.0])
    xb = np.array([1.0, 3.0, 5.0])
    xc = np.array([-2.0, 0.0, 2.0])

    Xs = pd.DataFrame ([xa, xb, xc])
    Xs.index = ['A', 'B', 'C']




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

    assert kern.make_K(seqs, normalize=False).equals(K),\
    'Failed make_K for var_p = 1'
    assert kern.make_K(seqs,hypers=[vp], normalize=False).equals(K*vp),\
    'Failed make_K for var_p ~= 1.'
    assert kern.make_K(seqs, normalize=True, hypers=[vp]).equals(K/norm*vp),\
    'Failed make_K with normalization for var_p ~= 1.'

    # now let's make sure we can train it and use keys to access functions
    kern.train(seqs)
    assert kern.saved_X == {'A':'RYMA',
                            'B':'RTHA',
                            'C':'RTMA',
                            'D':'RYMA'},\
    'Failed to train HammingKernel.'

    kern.delete(seqs.loc[['D']])
    assert kern.saved_X == {'A':'RYMA',
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
    assert kern.base_K.equals(K/norm)
    assert kern.make_K(hypers=[vp]).equals(K/norm*vp),\
    'Failed make_K using saved base_K'

    print 'HammingKernel passes all tests.'

#     kern = gpkernel.WeightedHammingKernel()
#     print kern.make_K(seqs, normalize=False)





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
    assert kern.saved_X == {'A': [((0, 'R'), (1, 'Y')), ((2, 'M'), (3, 'A'))],
                      'C': [((0, 'R'), (1, 'T')), ((2, 'M'), (3, 'A'))],
                      'B': [((0, 'R'), (1, 'T')), ((2, 'H'), (3, 'A'))],
                      'D': [((0, 'R'), (1, 'Y')), ((2, 'M'), (3, 'A'))]},\
    'Failed to train structure kernel.'

    kern.delete(seqs.loc[['D']])
    assert kern.saved_X == {'A': [((0, 'R'), (1, 'Y')), ((2, 'M'), (3, 'A'))],
                            'C': [((0, 'R'), (1, 'T')), ((2, 'M'), (3, 'A'))],
                            'B': [((0, 'R'), (1, 'T')), ((2, 'H'), (3, 'A'))]}

    kern.set_X(seqs)
    assert kern.base_K.equals(K/norm)


    assert kern.calc_kernel(seqs.iloc[0],seqs.iloc[1]) == 0,\
    'Failed calc_kernel for sequences with no shared contacts.'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2],normalize=False) == 1,\
    'Failed calc_kernel for sequences with shared contacts.'
    assert kern.calc_kernel(seqs.iloc[1],seqs.iloc[2],
                            hypers=[vp],normalize=False) == vp,\
    'Failed calc_kernel for var_p ~= 1.'

    # test make_K
    assert kern.make_K(seqs, normalize=False).equals(K),\
    'Failed make_K for var_p = 1.'
    assert kern.make_K(seqs, normalize=True).equals(K/norm),\
    'Failed make_K with normalization for var_p = 1.'
    assert kern.make_K(seqs, hypers=[vp], normalize=True).equals(K/norm*vp),\
    'Failed make_K with normalization for var_p ~= 1.'
    assert kern.make_K(seqs,hypers=[vp], normalize=False).equals(K*vp),\
    'Failed make_K for var_p ~= 1.'
    assert kern.make_K(hypers=[vp]).equals(K/norm*vp),\
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

def test_se_kernel():
    print 'Testing SEKernel...'
    sek = gpkernel.SEKernel()
    assert sek.hypers == ['sigma_f', 'ell']

    test_se(sek)

    params = (0.2, 0.5)

    xa = np.array([2.0, 3.0, 4.0])
    xb = np.array([1.0, 3.0, 5.0])
    xc = np.array([-2.0, 0.0, 2.0])
    xs = np.concatenate((xa, xb, xc))
    xs = xs.reshape((3,3))
    inds = ['A','B','C']
    xs = pd.DataFrame(xs, index=inds)

    dists = np.array ([[1, 3, 2],
                       [3, 1, 0],
                       [2, 0, 1]])
    dists = pd.DataFrame(dists, index=inds, columns=inds)

    # test calc_kernel
    assert sek.calc_kernel(xa, xb, params) == sek.se([xa, xb], params)

    # test make_K
    assert sek.make_K(xs, params).equals(sek.se(xs, params))

    print 'SEKernel passes all tests.'

    print 'Testing StructureSEKernel...'

    ssek = gpkernel.StructureSEKernel(contacts)

    assert ssek.hypers == ['sigma_f', 'ell']


    K = pd.DataFrame([[2.0,0.0,1.0,2.0],
                      [0.0,2.0,1.0,0.0],
                      [1.0,1.0,2.0,1.0],
                      [2.0,0.0,1.0,2.0]],
                     index=seqs.index,
                     columns=seqs.index)
    d2 = 2 - K

    # test distance and make_D functions
    assert ssek.distance(seqs.loc['A'], seqs.loc['B']) == 2,\
        'StructureSEKernel fails to calculate distance.'
    assert ssek.make_D(seqs).equals(d2), \
        'StructureSEKernel fails to D.'

    # Test make_K and calc_kernel
    assert ssek.make_K(hypers=params, X_seqs=seqs)\
        .equals((sek.d_squared_to_se(d2, params))),\
        'StructureSEKernel fails make_K.'
    assert ssek.calc_kernel(seqs.iloc[0], seqs.iloc[1], params) == \
        sek.d_squared_to_se(2, params), 'StructureSEKernel fails calc_kernel.'

    # Set a d_squared to speed things up
    ssek.set_X(seqs)
    assert ssek.d_squared.equals(d2), 'ssek.d_squared is wrong.'
    # Retest make_K
    assert ssek.make_K(hypers=params).equals((sek.d_squared_to_se(d2, params))),\
        'StructureSEKernel fails make_K.'

    print 'StructureSEKernel passes all tests.'


    print 'Testing HammingSEKernel...'

    ssek = gpkernel.HammingSEKernel()

    assert ssek.hypers == ['sigma_f', 'ell']


    K = pd.DataFrame([[4.,2.,3.,4.],
                      [2.,4.,3.,2.],
                      [3.,3.,4.,3.],
                      [4.,2.,3.,4.]],
                    index=seqs.index,
                    columns=seqs.index)
    d2 = 4 - K

    # test distance and make_D functions
    assert ssek.distance(seqs.loc['A'], seqs.loc['B']) == 2,\
        'HammingSEKernel fails to calculate distance.'
    assert ssek.make_D(seqs).equals(d2), \
        'HammingSEKernel fails to D.'

    # Test make_K and calc_kernel
    assert ssek.make_K(hypers=params, X_seqs=seqs)\
        .equals((sek.d_squared_to_se(d2, params))),\
        'HammingSEKernel fails make_K.'
    assert ssek.calc_kernel(seqs.iloc[0], seqs.iloc[1], params) == \
        sek.d_squared_to_se(2, params), 'StructureSEKernel fails calc_kernel.'

    # Set a d_squared to speed things up
    ssek.set_X(seqs)
    assert ssek.d_squared.equals(d2), 'ssek.d_squared is wrong.'
    # Retest make_K
    assert ssek.make_K(hypers=params).equals((sek.d_squared_to_se(d2, params))),\
        'HammingSEKernel fails make_K.'

    print 'HammingSEKernel passes all tests.'

def test_linear_kernel():
    print 'Testing linear kernel...'
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
    assert np.array([(saved_X[k] == kern.saved_X[k]).all() \
                     for k in saved_X.keys()]).all(),\
    'Failed to train linear kernel.'

    kern.delete(X.loc[['D']])
    saved_X = {i:X.loc[i] for i in ['A','B','C']}
    assert np.array([(saved_X[k] == kern.saved_X[k]).all() \
                     for k in saved_X.keys()]).all()
    kern.set_X(X)
    assert kern.base_K.equals(K)


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

if __name__=="__main__":
    test_matern_kernel()
#     test_linear_kernel()
#     test_hamming_kernel()
#     test_structure_kernel()
#     test_se_kernel()

