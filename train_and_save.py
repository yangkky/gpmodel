''' Reads in an ncr training set, builds a GPModel, and pickles the model'''

import argparse, gpmodel, gpkernel, os, sys, pickle
import pandas as pd
import numpy as np

def main ():
    dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path',required=True)
    parser.add_argument('-a','--a_and_c',required=False)
    parser.add_argument('-t', '--training',required=True)
    parser.add_argument('-as', '--assignments',required=True)
    parser.add_argument('-k', '--kernel',required=True)

    args = parser.parse_args() 
    path = args.path
    a_and_c = os.path.join(dir, path+args.a_and_c)  
    sample_space, contacts = pickle.load(open(a_and_c))
    training_chimeras = os.path.join(dir,path+args.training)
    assignments_file = os.path.join(dir,path+args.assignments)
    
    print 'Creating Kernel...'
    if args.kernel == 'hamming':
        kern = gpkernel.HammingKernel()
    elif args.kernel == 'structure':
        kern = gpkernel.StructureKernel(contacts, sample_space)
    else: 
        sys.exit ('Kernel type must be hamming or structure')
    
    
    
    print 'Creating training set...'
    # this reads in the chimeras for the training set
    # we need to shift all the parent numbers down by one to be consistent with 
    # how they are recorded elsewhere
    X_inds = pd.read_table (training_chimeras)    # overkill, but it works
    X_inds = [str(X_inds.loc[x][0]) for x in X_inds.index]
    for i,x in enumerate(X_inds):
        this = ''
        for j in x:
            this += (str(int(j)-1))
        X_inds[i] = this
    X_inds = list(set(X_inds))                    # remove duplicates
    
    #inds = generate_library (n_blocks, n_parents)
    
    ## load the assignments file
    assignments_line = [l for l in open(assignments_file).read().split('\n') if len(l)>0 and l[0]!='#']
    assignment = [ord(l.split('\t')[2]) - ord('A') for l in assignments_line if l.split('\t')[2] !='-']
    nodes_outputfile = [int(l.split('\t')[1])-1 for l in assignments_line if l.split('\t')[2] !='-'] # -1 because counting 0,1,2...
    
    # put this info in a dictionary because it's easier
    assignments_dict = dict(zip(nodes_outputfile, assignment)) # maps each position to a block    
    
    # for each member of the training set, we want to generate a sequence
    # that can be used to generate the hamming matrix
    X_seqs = []
    for x in X_inds:
        seq = []
        for pos,aa in enumerate(sample_space):
            # Figure out which parent to use at that position
            if pos in assignments_dict:
                parent = int(x[assignments_dict[pos]])
            else:
                parent = 0
            seq.append (sample_space[pos][parent])
            
        X_seqs.append(seq)
    X_seqs = pd.DataFrame(X_seqs, index = X_inds)
    
    Ys = np.random.normal(size=len(X_seqs.index)) # need real Ys at some point :-)
    Ys = pd.Series(Ys, index=X_inds)
    
    print 'Training model...'
    model = gpmodel.GPModel(X_seqs,Ys,kern,guesses=[100,10])
    print 'Pickling model...'
    f = open (os.path.join(path, args.kernel +'_kernel.pkl'), 'wb')
    pickle.dump(model, f)

    

          
if __name__ == "__main__":
    main()
