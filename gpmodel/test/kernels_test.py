import sys
import pytest

import pandas as pd
import numpy as np

from gpmodel import gpkernel


seqs = pd.DataFrame([['R','Y','M','A'],['R','T','H','A'], ['R','T','M','A']],
                    index=['A','B','C'], columns=[0,1,2,3])

seqs = seqs.append(seqs.iloc[0])
seqs.index = ['A','B','C','D']
space = [('R'), ('Y', 'T'), ('M', 'H'), ('A')]
contacts = [(0,1),(2,3)]

X = np.array([[1.0, 2.0], [3.0, -1.0], [2.0, -2.0]])
X_df = pd.DataFrame(X, index=['A','B','C'])

def test_gpkernel():
    kernel = gpkernel.GPKernel()
    assert not kernel._saved_X
    assert kernel.hypers == []
    kernel.set_X(X_df)
    assert np.isclose(kernel._saved_X['A'], X[0]).all()
    kernel.delete(X_df.loc[['A']])
    assert sorted(kernel._saved_X.keys()) == ['B', 'C']
    assert np.isclose(kernel._get_X('B'), X[1]).all()
    assert np.isclose(kernel._get_X(X[2]), X[2]).all()


def test_se_kernel():
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


def test_polynomial_kernel():
    kern1 = gpkernel.PolynomialKernel(1)
    kern3 = gpkernel.PolynomialKernel(3)
    assert kern1.hypers == ['sigma_0', 'sigma_p']
    assert kern1._deg == 1
    assert kern3._deg == 3
    pytest.raises(ValueError, 'gpkernel.PolynomialKernel(0)')
    pytest.raises(TypeError, 'gpkernel.PolynomialKernel(1.2)')

    # test on single values
    first = 4
    second = 2
    params = [0.9, 0.1]
    assert np.isclose(kern1.calc_kernel(first, second, params), 0.89)
    assert np.isclose(kern3.calc_kernel(first, second, params), 0.89 ** 3)
    xa = np.array([2.0, 3.0, 4.0])
    xb = np.array([1.0, 3.0, 5.0])
    xc = np.array([-2.0, 0.0, 2.0])
    Xs = pd.DataFrame ([xa, xb, xc])
    Xs.index = ['A', 'B', 'C']
    assert kern1.calc_kernel(xa, xb, params) == 1.12
    assert kern3.calc_kernel(xa, xb, params) == 1.12 ** 3

    dot = np.array([[29, 31, 4],
                    [31, 35, 8],
                    [4, 8, 8]])

    assert np.isclose(kern1._get_all_dots(Xs), dot).all()
    K3 = (params[0] ** 2 + params[1] ** 2 * dot) ** 3
    assert np.isclose(kern3.make_K(Xs, params), K3).all()

    kern1.set_X(Xs)
    kern3.set_X(Xs)
    assert sorted(kern1._saved_X.keys()) == ['A', 'B', 'C']
    assert np.isclose(kern1._saved_X['A'], xa).all()
    assert np.isclose(kern1._dots, dot).all()
    assert np.isclose(kern3.make_K(hypers=params), K3).all()


def test_matern_kernel():
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
    assert kern1.calc_kernel(xa, xb, params) == actual1[0, 1]
    assert kern2.calc_kernel('A', xb, params) == actual2[0, 1]


def test_hamming_kernel():
    vp = 0.4
    K = pd.DataFrame([[4.,2.,3.,4.],
                      [2.,4.,3.,2.],
                      [3.,3.,4.,3.],
                      [4.,2.,3.,4.]],
                    index=seqs.index,
                    columns=seqs.index)
    norm = 4.0
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
    assert kern._saved_X == {'A': ['0R', '1Y', '2M', '3A'],
                             'C': ['0R', '1T', '2M', '3A'],
                             'B': ['0R', '1T', '2H', '3A'],
                             'D': ['0R', '1Y', '2M', '3A']},\
    'Failed to train HammingKernel.'

    kern.delete(seqs.loc[['D']])
    assert kern._saved_X == {'A': ['0R', '1Y', '2M', '3A'],
                             'C': ['0R', '1T', '2M', '3A'],
                             'B': ['0R', '1T', '2H', '3A']}

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
    kern = gpkernel.StructureKernel(contacts)
    assert kern.hypers == ['var_p']


    # now let's make sure we can train it and use keys to access functions
    kern.train(seqs)
    assert kern._saved_X == {'A': ['0R1Y', '2M3A'],
                             'C': ['0R1T', '2M3A'],
                             'B': ['0R1T', '2H3A'],
                             'D': ['0R1Y', '2M3A']},\
    'Failed to train structure kernel.'

    kern.delete(seqs.loc[['D']])
    assert kern._saved_X == {'A': ['0R1Y', '2M3A'],
                             'C': ['0R1T', '2M3A'],
                             'B': ['0R1T', '2H3A']}

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


def test_protein_matern_kernels():
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


def test_protein_se_kernels():
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


def test_linear_kernel():
    kern = gpkernel.LinearKernel()
    X = pd.DataFrame([[4.,2.,3.,4.],
                      [2.,0.,1.,2.],
                      [1.,1.,4.,3.],
                      [-3.,2.,3.,0.]],
                     index=seqs.index)
    K = np.array([[45.,19.,30.,1.],
                  [19.,9.,12.,-3.],
                  [30.,12.,27.,11.],
                  [1.,-3.,11.,22.]])
    assert kern.hypers == ['var_p']


    # now let's make sure we can train it and use keys to access functions
    kern.train(X)
    saved_X = {i:np.array(X.loc[i]) for i in X.index}

    kern.set_X(X)
    assert np.array_equal(kern._base_K,K)


    assert kern.calc_kernel(X.iloc[0],X.iloc[1]) == 19,\
    'Failed calc_kernel for vp=1.'
    assert kern.calc_kernel(X.iloc[0],X.iloc[1],
                            hypers=[2]) == 38,\
    'Failed calc_kernel for var_p ~= 1.'

    # test make_K
    assert np.array_equal(kern.make_K(X),K),\
    'Failed make_K for var_p = 1.'
    assert np.array_equal(kern.make_K(X, hypers=[2]),K*2),\
    'Failed make_K for var_p ~= 1.'
    assert np.array_equal(kern.make_K(hypers=[2]),K*2),\
    'Failed make_K using saved base_K'


    assert kern.calc_kernel('A','B') == 19,\
    'Failed calc_kernel with two trained sequences.'
    assert kern.calc_kernel('A',X.iloc[1]) == 19,\
    'Failed calc_kernel with trained, untrained sequences.'
    assert kern.calc_kernel(X.iloc[2],'B') == 12,\
    'Failed calc_kernel with untrained, trained sequences.'


def test_sum_kernel():
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


if __name__=="__main__":
    test_gpkernel()
    test_matern_kernel()
    test_linear_kernel()
    test_polynomial_kernel()
    test_hamming_kernel()
    test_structure_kernel()
    test_se_kernel()
    test_protein_se_kernels()
    test_protein_matern_kernels()
    test_sum_kernel()
