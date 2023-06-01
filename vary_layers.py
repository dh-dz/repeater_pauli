#one-layer steane code
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import time
from EffP import *
from QLPDCgen import *
from PLmatrix_CSS import *
from ldpc import bposd_decoder

from joblib import Parallel, delayed
import multiprocessing
# what are your inputs, and what operation do you want to
# perform on each input. For example...
num_cores = 20 #multiprocessing.cpu_count()

# list the constant parameters: 
# 2. error rate of pauli errors:
p_pauli = 0.001
# 3. max number of iteratios for BP, set to be the number of physical qubits in the code
#N_iter = steane_code.N
bdy = True ## boundary condition, true (obc), false(pbc)
repeat = 20
Nrep = 500 # number of iterations
Nl_list = [1,2,3,4,5,6,7,8,9,10]
p_list = [0.05]

def succ_prob_css_calc_new(B_orig, logicals_in, s_nodes, loss_inds):
    ######################################################
    ## inputs:
    ## B_orig [type: networkx]: stabilizer graph, two kinds of nodes: qubit 1...N and stabilizer s1...s_{Ns}
    ## logicals_in [type: list of numpy arrays]: logical operators in every row, columns act on qubits
    ## s_nodes [type: list]: list of stabilizer nodes s1...s_{Ns}
    ## loss_inds [type: numpy array]: index of erased qubits
    #####################
    ## output:
    ## succ_fail [type: binary value]: 0 (failure), 1(success)
    ######################################################
    N = np.size(logicals_in,1)
    B = B_orig.copy()
    logicals = list(np.copy(logicals_in))
    s_nodes_set = set(np.copy(s_nodes))

    Ns_remain = len(s_nodes_set) # number of stabilizer generators
    #print(Ns_remain)
    q_remain = list(set(B.nodes())-s_nodes_set) # number of qubits (anciall+data)
    node_list = list(s_nodes_set) + q_remain  # indices of all nodes in graph
    adj_mat_new = nx.to_numpy_array(B, nodelist = node_list) # adjaceny matrix of stabilizer graph
    Sx_mat = adj_mat_new[:Ns_remain,Ns_remain:] # stabilizer group matrix

    for i_q, q in enumerate(loss_inds):
        ## correct logical operators
        logicals = correct_logical(q,logicals, Sx_mat)
        ## update stabilizer group
        ## first: update graph
        if q in B:
            B, s_nodes_set = modify_graph(q,B,s_nodes_set)
        ## second: update stabilizer group matrix
            Ns_remain = len(s_nodes_set)
            if Ns_remain> 0:
                q_remain = list(set(B.nodes())-s_nodes_set)
                node_list = list(s_nodes_set) + q_remain
                adj_mat_new = nx.to_numpy_matrix(B, nodelist = node_list)
                Sx_red = adj_mat_new[:Ns_remain,Ns_remain:]
                Sx_mat = np.zeros((Ns_remain,N))
                Sx_mat[:,q_remain] = Sx_red
            else:
                #Sx_mat = []
                Sx_mat = np.zeros((Ns_remain,N))
                # break
    num_qs = 0
    if len(logicals)>=1:
        #print(len(logicals))
        for i_l in range(len(logicals)):
            if np.sum(logicals[i_l][loss_inds])==0:
                num_qs += 1 
            # print(logicals)
    return num_qs, Sx_mat, logicals

# in layer stabilizer group
Sx_mat = np.array([[1,1,1,1,0,0,0],\
              [1,1,0,0,1,1,0],\
              [1,0,1,0,1,0,1]])
Nq_l = np.size(Sx_mat,1) # number of data qubits per layer
Ns_l = np.size(Sx_mat,0) # number of stabilizers per layer

for i_L, Nl in enumerate(Nl_list):
    #print("L= %d" % (Nl))

    N = Nl*(Nq_l+Ns_l)+Ns_l # number of data qubits
    #print("N:",N)
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
            N_ls = 0 #N_loss_success
            for i_r in range(Nrep):
                loss_inds = np.random.permutation(np.argwhere(np.random.rand(N)<p_stab)[:,0])
                succ_prob_7_num, Sx_mat, logicals = succ_prob_css_calc_new(B_orig, logical, s_nodes, loss_inds)
                if succ_prob_7_num != 0:
                    N_ls += 1
                    bpd=bposd_decoder(
                        Sx_mat,#the parity check matrix
                        error_rate = 2/3*p_pauli,
                        xyz_error_bias= [1, 0, 0],
                        channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
                        max_iter = len(Sx_mat), #the maximum number of iterations for BP)
                        bp_method="ms",
                        ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
                        osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
                        osd_order=4 #the osd search depth
                        )
                    
                    error=np.zeros(N).astype(int)
                    loss_inds = np.random.permutation(np.argwhere(np.random.rand(N)<p_pauli)[:,0])
                    error[loss_inds] = 1
                    syndrome=Sx_mat @error %2
                    bpd.decode(syndrome)
                    #Decoding is successful if the residual error commutes with the logical operators
                    residual_error=(bpd.osdw_decoding+error) %2
                    a = (logicals@residual_error%2).any()
                    if not a: 
                        succ_prob_7_ml[i_p] += 1
            succ_prob_7_ml[i_p] /= N_ls
        print(succ_prob_7_ml)
        toc = time.time()
        print("finished L = %d, r=%d in %.1f secs" % (Nl,i_rep,toc-tic))

        if bdy:
            fname = "pauli/7q_layers/" + "even_Nl_%d_i_%d.npz" % (Nl,i_rep)
        else:
            assert 0

        np.savez(fname, succ_prob=succ_prob_7_ml, p_list=p_list, Nrep=Nrep)
        return 0
    #results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))
    results = runner(0)