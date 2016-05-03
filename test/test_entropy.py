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
K = pd.DataFrame([[2.0,0.0,1.0,2.0],
                  [0.0,2.0,1.0,0.0],
                  [1.0,1.0,2.0,1.0],
                  [2.0,0.0,1.0,2.0]],
                 index=seqs.index,
                 columns=seqs.index)

ent = gpentropy.GPEntropy(kernel, [vp], var_n=0.2, observations=seqs)

print ent._Ky
