from __future__ import print_function

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
 
import numpy as np
import matplotlib.pyplot as plt
import function_blocks as fb 

def main():
    # Softmax layer 
    softmax = lambda s: s / np.sum(s, axis=1, dtype=np.float64, keepdims=True)

    # SC network parameters 
    len_lst = [pow(2,5),pow(2,8),pow(2,11),pow(2,13),pow(2,14),pow(2,15),pow(2,16),pow(2,17)]

    l1_params = {
        'matmul_inp': 1,     # Matric multiplication input scaling 
        'matmul_int': None,  # Matrix multiplication intermediate scaling
        'matmul_out': 1024,  # Matrix multiplication output scaling 
        'matmul_nod': None,  # Number of nodes in dot product decomposition
        'matmul_usc': True,  # Upscale the result of matrix multiplication
        'matmul_gint':32,    # Gain factor for intermediate dot products
        'matmul_gout':None,  # Gain factor for the output of dot product   
        'matadd_inp': 32,    # Matrix addition input scaling
        'matadd_out': 32,    # Matrix addition output scaling 
        'matadd_usc': True   # Upscale the result of matrix addition
    }

    lo_params = {
        'matmul_inp': 32,    # Matrix multiplication input scaling
        'matmul_int': None,  # Matrix multiplication intermediate scaling 
        'matmul_out': 2048,  # Matrix multiplication output scaling
        'matmul_nod': None,  # Number of nodes in dot product decomposition
        'matmul_usc': True,  # Upscale the result of matrix multiplication
        'matmul_gint':8,     # Gain factor for intermediate dot products
        'matmul_gout':None,  # Gain factor for the output of dot product 
        'matadd_inp': 256,   # Matrix addition input scaling
        'matadd_out': 256,   # Matrix addition output scaling 
        'matadd_usc': True   # Upscale the result of matrix addition
    }

    # Load data 
    features = np.genfromtxt('../../../data/features.csv', delimiter=',')
    labels = np.genfromtxt('../../../data/labels.csv',   delimiter=',')

    # Load trained coefficients 
    weights = {
        'h1':   np.genfromtxt('coeffs/W1.csv', delimiter=','),
        'out':  np.genfromtxt('coeffs/Wout.csv', delimiter=','),
    }

    biases = {
        'b1':   np.genfromtxt('coeffs/b1.csv', delimiter=','),
        'out':  np.genfromtxt('coeffs/bout.csv', delimiter=','),
    }

    # Slice a subset of the data
    test_size = 10
    X = features[:test_size,:]
    Y = labels[:test_size,:]

    print('Data & Model Restored')

    sc_accuracy_lst = []
    for no_samples in len_lst:
        print('SC inference using bit-stream length of', no_samples)

        # Convert data to SC bit-streams 
        S = fb.mat_sng(X,no_samples)

        print('Bit-streams Generated')

        # SC network graph 
        l1_matmul = fb.sc_matmul(S,
                                 weights['h1'],
                                 Sin=l1_params['matmul_inp'],
                                 Sout=l1_params['matmul_out'],
                                 no_nodes=l1_params['matmul_nod'],
                                 Sint=l1_params['matmul_int'],
                                 upscale=l1_params['matmul_usc'],
                                 g_int=l1_params['matmul_gint'],
                                 g_out=l1_params['matmul_gout'])
        print('L1 matmul done')

        l1_matadd = fb.sc_matvec_add(l1_matmul,
                                     biases['b1'],
                                     l1_params['matadd_inp'],
                                     l1_params['matadd_usc'])

        print('L1 matadd done')

        l2_matmul = fb.sc_matmul(l1_matadd,
                                 weights['out'],
                                 Sin=lo_params['matmul_inp'],
                                 Sout=lo_params['matmul_out'],
                                 no_nodes=lo_params['matmul_nod'],
                                 Sint=lo_params['matmul_int'],
                                 upscale=lo_params['matmul_usc'],
                                 g_int=lo_params['matmul_gint'],
                                 g_out=lo_params['matmul_gout'])

        print('L2 matmul done')

        l2_matadd = fb.sc_matvec_add(l2_matmul,
                                     biases['out'],
                                     lo_params['matadd_inp'],
                                     lo_params['matadd_usc'])

        print('L2 matadd done')

        # Convert back to floating point & calculate accuracy 
        logits = fb.mat_sc_value(l2_matadd) 
        logits = logits*lo_params['matadd_out']

        probs = np.exp(logits,dtype=np.float64)

        prediction = softmax(probs)

        correct_pred = np.equal(np.argmax(prediction,axis=1),np.argmax(Y,axis=1))

        accuracy = np.mean(correct_pred)
        sc_accuracy_lst.append(accuracy)
        print("Testing Accuracy: ", accuracy)

    # Plot the results
    float_net_accuracy = 0.870455
    float_net_accuracy_lst = np.ones(len(len_lst),dtype=np.float64)*float_net_accuracy
    plt.semilogx(np.array(len_lst),np.array(accuracy_lst),basex=2)
    plt.semilogx(np.array(len_lst),float_net_accuracy_lst,color='r',basex=2)
    plt.title('Classification Accuracy versus Bit-Stream length')
    plt.ylabel('Classification Accuracy')
    plt.xlabel('Bit-Stream Length')
    plt.grid(True)

if __name__ == "__main__":    
    main()
    plt.show()