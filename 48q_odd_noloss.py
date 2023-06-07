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

'''
def succ_prob_css_q_resolved_new(B_orig, logicals_in, s_nodes, loss_inds):
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
    logicals = np.copy(logicals_in)
    s_nodes_set = set(np.copy(s_nodes))

    Ns_remain = len(s_nodes_set) # number of stabilizer generators
    q_remain = list(set(B.nodes())-s_nodes_set) # number of qubits (anciall+data)
    node_list = list(s_nodes_set) + q_remain  # indices of all nodes in graph
    adj_mat_new = nx.to_numpy_array(B, nodelist = node_list) # adjaceny matrix of stabilizer graph
    Sx_mat = adj_mat_new[:Ns_remain,Ns_remain:] # stabilizer group matrix

    N_logic = np.size(logicals_in,0)
    logic_list = np.ones(N_logic)
    for i_q, q in enumerate(loss_inds):
        ## correct logical operators
        logic_remained = np.argwhere(logic_list==1)[:,0]
        if len(logic_remained)>0:
            # print(logic_remained)
            # print(np.shape(logicals))
            # print(logicals[logic_remained])
            logic_removed,logic_modified, logic_op = correct_logical_q_resolved(q,logicals[logic_remained,:], Sx_mat)
            logic_list[logic_remained[logic_removed]] = 0
            if len(logic_modified)>0:
                logicals[logic_remained[logic_modified],:] = np.array(logic_op)
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
                    Sx_mat = np.zeros((Ns_remain,N))
    
    return logic_list, Sx_mat, logicals
'''


# 2. error rate of pauli errors:
p_pauli = 0.001
# 3. max number of iteratios for BP, set to be the number of physical qubits in the code
#N_iter = steane_code.N
# 4. N_q: just the number of physical qubits in the code 
#N_q = steane_code.N

bdy = True ## boundary condition, true (obc), false(pbc)
repeat = 20
Nrep = 1000 # number of iterations
Nl_list = [2,4,6,8,10,12,14,16,18,20]
p_list = [0]
p_r_list = [0]

######## define quantum code here ########
l=24
n = 48
k = 6
r = 21
H_XZ = GB_gen(l,[2,8,15],[2,12,17])
##########
Sx_mat = H_XZ[:, 0:n]
Sx_mat = Sx_mat[~np.all(Sx_mat == 0, axis=1)]
Sz_mat = H_XZ[:, n:]
Sz_mat = Sz_mat[~np.all(Sz_mat == 0, axis=1)]
print("Sx, Sz shapes:", Sx_mat.shape,Sz_mat.shape)
print("[Sx,Sz] = ", np.linalg.norm(Sx_mat@Sz_mat.T %2))

from ldpc.mod2 import rank,row_basis,inverse
# print(row_basis(Sx_mat).shape)
Sx_mat = row_basis(Sx_mat)
Sz_mat = row_basis(Sz_mat)
print("Sx, Sz shapes:", Sx_mat.shape,Sz_mat.shape)
print("[Sx,Sz] = ", np.linalg.norm(Sx_mat@Sz_mat.T %2))


from ldpc.codes import hamming_code
from bposd.css import css_code

qcode=css_code(hx=Sx_mat,hz=Sz_mat)

lx=qcode.lx #x logical operators
lz=qcode.lz #z logical operators
print("X weight: ", np.sum(lx,axis=1))
print("Z weight: ", np.sum(lz,axis=1))

# lz[0,:] = (lz[1,:]+lz[0,:])%2

# print(qcode.compute_code_distance())
temp=inverse(lx@lz.T %2)
lx=temp@lx %2
    
print("lx, lz shapes:", lx.shape,lz.shape)

print("[lx,lz] = ", (lz@lx.T)% 2)
print("[Sx,Sz] = ", np.linalg.norm((Sx_mat@Sz_mat.T) % 2))
print("[Sz,lx] = ", np.linalg.norm((Sz_mat@lx.T) % 2))
print("[Sx,lz] = ", np.linalg.norm((Sx_mat@lz.T) % 2))

print("X weight: ", np.sum(lx,axis=1))
print("Z weight: ", np.sum(lz,axis=1))

logical_tZ = lz
logical_tX = lx
N_logic = np.size(logical_tZ,0)
Nq_l = np.size(Sx_mat,1) # number of data qubits 
Ns_l = np.size(Sx_mat,0) # number of stabilizers 
##########################################
for p_r in p_r_list:
    p_stab = 1-(1-p_r)**0.5
    for i_L, Nl in enumerate(Nl_list):
        #N_ls = 0 #N_loss_success
        print("L= %d" % (Nl))

        N = Nl*(Nq_l+Ns_l) # number of data qubits
        Ns = Nl*Ns_l # number of stabilizers
        s_nodes = ["s%d" % s for s in np.arange(Ns)]

        B_orig_X = foliated_graph(Sx_mat,s_nodes, Nl, bdy)

        logical_in_X = np.zeros((N_logic,N))
        data_qs = np.zeros((1,N))
        for i_l in range(Nl):
            logical_in_X[:,i_l*(Nq_l+Ns_l):i_l*(Nq_l+Ns_l)+Nq_l] = logical_tX
            data_qs[:,i_l*(Nq_l+Ns_l):i_l*(Nq_l+Ns_l)+Nq_l] = np.ones(Nq_l)
        ancilla_qs = 1- data_qs
        
        def runner(i_rep):
            tic = time.time()

            succ_prob_X = np.zeros((len(p_list),np.size(logical_tX,0)))
            succ_prob_word_X = np.zeros(len(p_list))
            for i_p, p in enumerate(p_list):
                N_ls = np.zeros(np.size(logical_tX,0))
                for i_r in range(Nrep):
                    '''
                    # loss_inds = np.random.permutation(np.argwhere(np.random.rand(N)<p)[:,0])
                    loss_inds_data = np.random.permutation(np.where(np.random.rand(N)<p*data_qs)[1])
                    loss_inds_ancilla = np.random.permutation(np.where(np.random.rand(N)<p_stab*ancilla_qs)[1])
                    loss_inds = np.concatenate((loss_inds_data,loss_inds_ancilla))
                    succ_prob_X_val, Sx_mat, logicals = succ_prob_css_q_resolved_new(B_orig_X, logical_in_X, s_nodes, loss_inds)
                    '''
                    error=np.zeros(N).astype(int)
                    loss_inds = np.random.permutation(np.argwhere(np.random.rand(N)<p_pauli)[:,0])
                    error[loss_inds] = 1
                    
                    for i_log in range(len(succ_prob_X_val)):
                        if succ_prob_X_val[i_log] != 0:
                            N_ls[i_log] += 1
                            bpd=bposd_decoder(
                                Sx_mat,#the parity check matrix
                                error_rate = 2/3*p_pauli,
                                xyz_error_bias= [1, 0, 0],
                                channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
                                #max_iter = len(Sx_mat), #the maximum number of iterations for BP)
                                max_iter = Nl,
                                bp_method="ms",
                                ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
                                osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
                                osd_order=4 #the osd search depth
                                )


                            syndrome=Sx_mat @error %2
                            bpd.decode(syndrome)
                            #Decoding is successful if the residual error commutes with the logical operators
                            residual_error=(bpd.osdw_decoding+error) %2
                            a = (logicals[i_log] @ residual_error % 2).any()
                            if not a: 
                                succ_prob_X[i_p,i_log] += succ_prob_X_val[i_log]
                                #succ_prob_7_ml[i_p] += 1                    
                        #succ_prob_X[i_p,:] += succ_prob_X_val
                        #succ_prob_word_X[i_p] += (np.sum(succ_prob_X_val)==N_logic)                    
                for i_log in range(len(succ_prob_X_val)):
                    succ_prob_X[i_p,i_log] /= N_ls[i_log]
            #succ_prob_X /= (Nrep)
            #succ_prob_word_X /= (Nrep)
            toc = time.time()
            
            print(succ_prob_X)
            print("finished p_r= %.2f, L = %d, r=%d in %.1f secs" % (p_r,Nl,i_rep,toc-tic))

            if bdy:
                fname = "pauli/48q_noloss/" + "odd_p_%.2f_Nl_%d_i_%d.npz" % (p_r,Nl,i_rep)
            else:
                assert 0

            #np.savez(fname, succ_prob_word_X=succ_prob_word_X, succ_prob_X=succ_prob_X, p_list=p_list, Nrep=Nrep)
            np.savez(fname, succ_prob_X=succ_prob_X, p_list=p_list, N_ls=N_ls)

            return 0
        results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))