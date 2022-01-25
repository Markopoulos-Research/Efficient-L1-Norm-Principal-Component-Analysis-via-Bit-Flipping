import sys
import time
import numpy as np
from numpy.linalg import svd

def l1pca_bf(X, K, num_init, print_flag):
    # Parameters
    toler =10e-8

    # Get the dimensions of the matrix.
    D = X.shape[0]	# Row dimension.
    N = X.shape[1]	# Column dimension.

    # Initialize the matrix with the SVD.
    _, S_x, V_x = svd(X , full_matrices = False)	# Hint: The singular values are in vector form.
    if D < N:
        V_x = V_x.T
    
    Y = np.diag(S_x)@V_x.T # X_t is Y = S*V'
    # Initialize the required matrices and vectors.
    Bprop = np.ones((N,K),dtype=float)
    nucnormmax = 0
    iterations = np.zeros((1,num_init),dtype=float)
    # For each initialization do.
    for ll in range(0, num_init):

        start_time = time.time()	# Start measuring execution time.

        z = X.T @ np.random.randn(D,1)	# Random initialized vector.
        if ll<1:    # In the first initialization, initialize the B matrix to sign of the product of the first
                    # right singular vector of the input matrix with an all-ones matrix.
            z = V_x[:,0]
            z = z.reshape(N,1)
        v = z@np.ones((1,K), dtype=float)
        v = np.random.randn(N,K)
        B = np.sign(v)	# Get a binary vector containing the signs of the elements of v.

        iter_ = 0
        L = list(range(N * K))

        while True:
            iter_ = iter_ + 1
            # Calculate all the possible binary vectors and all possible bit flips.

            a = np.zeros((N,K)) # nuclear norm of when the (m,l)th bit of B is flipped
            
            nucnorm = np.linalg.norm(Y@B, 'nuc')
            
            for x in L:
                l = x//N
                m = x-N*l
                elK = np.zeros(K)
                elK[l] = 1
                a[m,l] = np.linalg.norm(Y@B - 2*B[m,l]*(Y[:,m,None]@ [elK]), 'nuc')
            nucnorm_flip = np.max(a) # Choose the largest nuclear norm of YB

            n,k = np.unravel_index(np.nanargmax(a, axis=None), a.shape) # Pick the best bit flip

            if nucnorm_flip > nucnorm: # If the best bit flip increases the nuclear norm of YB, then flip the bit
                B[n,k] = -B[n,k]
                L.remove(k*N+n) # No longer flip that (n,k) bit
            elif nucnorm_flip <= nucnorm + toler and len(L)<N*K: # Else, but there has been bit-flips, reset bit-flipping process
                L = list(range(N*K))
            else:
                break

        # Calculate the final subspace.
        U, _, V = svd(X@B, full_matrices=False)
        Utemp = U[:,0:K]
        Vtemp = V[:,0:K]
        Q = Utemp@Vtemp.T
        
        nucnorm = sum(sum(abs(Q.T@X)))
        
        # Find the maximum nuclear norm across all initializations.
        if nucnorm > nucnormmax:
            nucnormmax = nucnorm
            Bprop = B
            Qprop = Q
            vmax = nucnorm
        iterations[0,ll] = iter_

    end_time = time.time()	# End of execution timestamp.
    timelapse = (end_time - start_time)	# Calculate the time elapsed.

    convergence_iter = np.mean(iterations, dtype=float) # Calculate the mean iterations per initialization.
    
    if print_flag:
        print("------------------------------------------")
        print("Avg. iterations/initialization: ", (convergence_iter))
        print("Time elapsed (sec):", (timelapse))
        print("Metric value:", vmax)
        print("------------------------------------------")

    return Qprop, Bprop, vmax