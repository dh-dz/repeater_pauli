import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import time
from EffP import *
from QLPDCgen import *
from PLmatrix_CSS import *


from joblib import Parallel, delayed
import multiprocessing
# what are your inputs, and what operation do you want to
# perform on each input. For example...
num_cores = 12#multiprocessing.cpu_count()                                     


bdy = True ## boundary condition, true (obc), false(pbc)
repeat = 100
Nrep = 10000 # number of iterations
Nl_list = np.arange(1,2,2)
p_list = [0.1]

# in layer stabilizer group
Sx_mat = np.array([[1,1,1,1,0,0,0],\
              [1,1,0,0,1,1,0],\
              [1,0,1,0,1,0,1]])
Nq_l = np.size(Sx_mat,1) # number of data qubits per layer
Ns_l = np.size(Sx_mat,0) # number of stabilizers per layer

for i_L, Nl in enumerate(Nl_list):
    print("L= %d" % (Nl))

    N = Nl*(Nq_l+Ns_l)+Ns_l # number of data qubits
    Ns = Nl*Ns_l # number of stabilizers
    s_nodes = ["s%d" % s for s in np.arange(Ns)]

    B_orig = foliated_graph(Sx_mat,s_nodes, Nl, bdy, "even")
    logical = np.zeros((1,N))
    for i_l in range(Nl):
        logical[0,i_l*(Nq_l+Ns_l):i_l*(Nq_l+Ns_l)+Nq_l] = np.ones(Nq_l)

    def runner(i_rep):
        tic = time.time()

        succ_prob_7_ml = np.zeros(len(p_list))
        for i_p, p_r in enumerate(p_list):
            p_stab = 1-(1-p_r)**0.5
            for i_r in range(Nrep):
                loss_inds = np.random.permutation(np.argwhere(np.random.rand(N)<p_stab)[:,0])
                succ_prob_7_ml[i_p] += succ_prob_css_calc(B_orig, logical, s_nodes, loss_inds)

        succ_prob_7_ml /= Nrep

        toc = time.time()
        print("finished L = %d, r=%d in %.1f secs" % (Nl,i_rep,toc-tic))

        if bdy:
            fname = "data_fig2/7q/" + "even_Nl_%d_i_%d.npz" % (Nl,i_rep)
            # fname = "data_7q/" + "even_Nl_%d_i_%d.npz" % (Nl,i_rep)
        else:
            assert 0

        np.savez(fname, succ_prob=succ_prob_7_ml, p_list=p_list, Nrep=Nrep)

        return 0

    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))

