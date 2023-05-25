import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import time
from numpy.linalg import matrix_power

### generate GB (Generalized Bicycle) code from circulant size l and 
### lists of x power coefficient a_l, b_l in a(x), b(x).
def GB_gen(l,a_l,b_l):
    P_l = np.zeros((l,l),dtype=int)
    P_l[0,l-1] = 1
    P_l[1:l,0:l-1] = np.eye(l-1, dtype=int)
    A_part = np.eye(l, dtype=int)
    B_part = np.eye(l, dtype=int)
    for k_l in a_l:
        A_part += matrix_power(P_l,k_l)
    for k_l in b_l:
        B_part += matrix_power(P_l,k_l)
    A_part = A_part % 2
    B_part = B_part % 2
    H_X = np.concatenate((A_part, B_part), axis=1)
    H_Z = np.concatenate((np.transpose(B_part), np.transpose(A_part)), axis=1)
    H_XZ = np.zeros((2*l,4*l), dtype = int)
    H_XZ[0:l, 0:2*l] = H_X
    H_XZ[l:2*l, 2*l:4*l] = H_Z
    return H_XZ

### generate HP (hypergraph product) codes:


### generate LP (lifted product) codes:

