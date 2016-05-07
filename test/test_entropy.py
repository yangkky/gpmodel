import sys
sys.path.append ('/Users/seinchin/Documents/Caltech/Arnold Lab/Programming tools/GPModel')
import gpkernel
import gpentropy
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

kernel = gpkernel.StructureKernel(contacts)
vp = 0.4
vn = 0.2
K = pd.DataFrame([[2.0,0.0,1.0,2.0],
                  [0.0,2.0,1.0,0.0],
                  [1.0,1.0,2.0,1.0],
                  [2.0,0.0,1.0,2.0]],
                 index=seqs.index,
                 columns=seqs.index)

ent = gpentropy.GPEntropy(kernel=kernel, hypers=[vp], var_n=vn, observations=seqs)

assert np.isclose(ent._Ky, K.values / 2*vp + np.eye(4) * vn).all()

# add observations and check Ky again
new_obs = seqs
ent.observe(new_obs)
K_no_noise = kernel.make_K(ent.observed, hypers=[vp])
real_K = K_no_noise + np.eye(8) * vn
assert np.isclose(ent._Ky, real_K).all()
assert np.isclose(ent._L, np.linalg.cholesky(real_K)).all()
# k_star
real_k_star = K_no_noise[0]
assert np.isclose(ent.k_star(seqs.loc['A']), real_k_star).all()
assert np.isclose(ent.k_star(seqs.loc[['A', 'B']]), K_no_noise[0:2]).all()
assert np.isclose(ent.k_star(seqs.loc[['A', 'B', 'D']]),
                             K_no_noise[[0,1,3]]).all()
# posterior covariance
new_seqs = pd.DataFrame([['B','Y','M','A'],['N','T','H','A'], ['G','T','M','A']],
                    index=[1, 2, 3], columns=[0,1,2,3])
k_off = np.matrix(ent.k_star(new_seqs))
cov = kernel.make_K(new_seqs, hypers=[vp])
real_post = cov - k_off * np.linalg.inv(ent._Ky) * k_off.T
assert np.isclose(ent.posterior_covariance(new_seqs), real_post).all()
# entropy
H =  0.5 * (np.log(np.linalg.det(real_post))
            + len(new_seqs) * np.log(2*np.pi*np.exp(1)))
assert np.isclose(ent.entropy(new_seqs), H)
# expected entropy
probabilities = np.array([[0.1, 0.9, 0.4]]).T
assert np.isclose(ent.expected_entropy(new_seqs, probabilities), 1.11849477318)
print '_____'
print ent.maximize_expected_entropy(new_seqs, probabilities, 2)

new_seqs = pd.DataFrame([['R','Y','H','A'],
                         ['N','T','H','A'],
                         ['G','T','M','A'],
                         ['N', 'T', 'M', 'A']],
                        index=['1', '2', '3', '4'], columns=[0,1,2,3])
probabilities = np.array([[0.1, 0.9, 0.4, 0.3]]).T

print ent.maximize_entropy(new_seqs, 2)


