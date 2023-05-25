import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import time
from numpy.linalg import matrix_power

# mod 2 inverse function
# adapted from https://towardsdatascience.com/find-the-inverse-of-a-matrix-using-python-3aeb05b48308
# write rows in reduced row echelon (rref) form
def invert_matrix(M):
    # store dimension
    n = M.shape[0]    
    # A must be square with non-zero determinant
    # assert np.linalg.det(M) != 0
    # identity matrix with same shape as A
    I = np.identity(n=n, dtype = int)
    # form the augmented matrix by concatenating A and I
    M = np.concatenate((M, I), axis=1)
    # move all zeros to buttom of matrix
    M = np.concatenate((M[np.any(M != 0, axis=1)], M[np.all(M == 0, axis=1)]), axis=0)
    # iterate over matrix rows
    for i in range(0, n):
        # initialize row-swap iterator
        j = 1
        # select pivot value
        pivot = M[i][i]
        # find next non-zero leading coefficient
        while pivot == 0 and i + j < n:
            # perform row swap operation
            M[[i, i + j]] = M[[i + j, i]]
            # incrememnt row-swap iterator
            j += 1
            # get new pivot
            pivot = M[i][i]
        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            # return inverse matrix
            return M[:, n:]
        # iterate over all rows except pivot to get augmented matrix into reduced row echelon form
        for j in [k for k in range(0, n) if k != i]:
            # subtract current row from remaining rows
            M[j] = (M[j] - M[i] * M[j][i]) % 2
    # return inverse matrix
    return M[:, n:]


## function for calculating logical operators:
def logical_from_Hparity(H_XZ, n, k, r):

    # step1: transform into the standard form:
    # Gaussian elimination on certain blocks:
    H_Ainv = invert_matrix(H_XZ[0:r,0:r])
    H_Einv = invert_matrix(H_XZ[r:n-k, n+r:2*n-k])
    H_XZ1 = np.zeros((n-k, 2*n), dtype = int)
    H_XZ1[0:r, :] = H_Ainv @ H_XZ[0:r, :]
    H_XZ1[r:n-k, :] = H_Einv @ H_XZ[r:n-k, :]
    H_XZ1 = H_XZ1 % 2

    # step2: solve the logical operator by different matrix section in the standard form:
    # notice: k sets of X/Z logical operators in general
    logicals_X = np.zeros((k, 2*n),dtype = int)
    logicals_Z = np.zeros((k, 2*n),dtype = int)

    # logical_X: (U1 U2 U3 | V1 V2 V3)
    # U1 = 0 (k*r), U2 = E^T (k*(n-k-r)), U3 = I (k*k), 
    # V1 = E^T*C_1^T + C_2^T (k*r), V2 = 0 (k*(n-k-r)), V3 = 0 (k*k):
    E = H_XZ1[r:n-k, 2*n-k:2*n]
    C_1 = H_XZ1[0:r, n+r:2*n-k]
    C_2 = H_XZ1[0:r, 2*n-k:2*n]
    A_2 = H_XZ1[0:r, n-k:n]
    logicals_X[0:k,0:r] = np.zeros((k,r),dtype=int)
    logicals_X[0:k,r:n-k] = np.transpose(E)
    logicals_X[0:k,n-k:n] = np.eye(k,dtype=int)
    logicals_X[0:k,n:n+r] = np.transpose(E)@np.transpose(C_1)+np.transpose(C_2)
    logicals_X[0:k,n+r:2*n-k] = np.zeros((k,n-k-r),dtype=int)
    logicals_X[0:k,2*n-k:2*n] = np.zeros((k,k),dtype=int)

    # logical_Z: (U1' U2' U3' | V1' V2' V3')
    # U1' = 0 (k*r), U2' = 0 (k*(n-k-r)), U3' = 0 (k*k), 
    # V1' = A_2^T (k*r), V2' = 0 (k*(n-k-r)), V3' = I (k*k):
    logicals_Z[0:k,0:r] = np.zeros((k,r),dtype=int)
    logicals_Z[0:k,r:n-k] = np.zeros((k,n-k-r),dtype=int)
    logicals_Z[0:k,n-k:n] = np.zeros((k,k),dtype=int)
    logicals_Z[0:k,n:n+r] = np.transpose(A_2)
    logicals_Z[0:k,n+r:2*n-k] = np.zeros((k,n-k-r),dtype=int)
    logicals_Z[0:k,2*n-k:2*n] = np.eye(k,dtype=int)
    logicals_X = logicals_X % 2
    logicals_Z = logicals_Z % 2

    return (logicals_X, logicals_Z)


## adapted from: stackoverflow...
def gf2_rank(rows):
    """
    Find rank of a matrix over GF2.
    The rows of the matrix are given as nonnegative integers, thought
    of as bit-strings.
    This function modifies the input list. Use gf2_rank(rows.copy())
    instead of gf2_rank(rows) to avoid modifying rows.
    """
    rows_new = []
    rank = 0
    while rows:
        #print(rows)
        pivot_row = rows.pop()
        #print(pivot_row)
        if pivot_row:
            rows_new.append(pivot_row)
            rank += 1
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
    return (rows_new, rank)


## transform back to the binary matrix:
def Hb_to_H(n, r, H_list):
    H_tot = []
    for i in range(r):
        H_row = []
        H_num = H_list[i]
        for k in range(n-1,-1,-1):
            H_row.append(H_num // (2**k))
            H_num = H_num % (2**k)
        H_tot.append(H_row)
    H_new = np.array(H_tot)
    return H_new

## get the reduced form of H_XZ --> with all independent rows (in H_X and H_Z)
## for CSS code only
def H_standard(H_XZ, n, r):
    H_X = H_XZ[:, 0:n]
    H_X = H_X[~np.all(H_X == 0, axis=1)]
    H_Z = H_XZ[:, n:2*n]
    H_Z = H_Z[~np.all(H_Z == 0, axis=1)]
    n_row = np.size(H_X,axis=0)
    H_Xb = []
    H_Zb = []
    for i_r in range(n_row):
        H_Xb.append(np.sum(H_X[i_r,:]* 2**np.arange(n-1,-1,-1)))
        H_Zb.append(np.sum(H_Z[i_r,:]* 2**np.arange(n-1,-1,-1)))

    H_Xbc = H_Xb.copy()
    H_Zbc = H_Zb.copy()
    H_Xb1, RoX = gf2_rank(H_Xbc)
    H_Zb1, RoZ = gf2_rank(H_Zbc)

    H_X_new = Hb_to_H(n,r,H_Xb1)
    H_Z_new = Hb_to_H(n,r,H_Zb1)
    H_XZ_new = np.zeros((2*r,2*n), dtype = int)
    H_XZ_new[0:r, 0:n] = H_X_new
    H_XZ_new[r:2*r, n:2*n] = H_Z_new
    return H_XZ_new