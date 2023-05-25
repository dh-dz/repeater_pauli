import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import time
from numpy.linalg import matrix_power
# from PLmatrix_CSS import *

def succ_prob_css_calc(B_orig, logicals_in, s_nodes, loss_inds):
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
                Sx_mat = []
                # break
    num_qs = 0
    if len(logicals)>=1:
        for i_l in range(len(logicals)):
            if np.sum(logicals[i_l][loss_inds])==0:
                num_qs += 1 
            # print(logicals)
    return num_qs
    
def modify_graph(q,B,s_nodes_set):
    sq = [n for n in B.neighbors(q)]
    if len(sq)==1:
        B.remove_nodes_from(sq)
        s_nodes_set -= set(sq)
    elif len(sq)>1:
        double_edgs = []
        for i in range(len(sq)-1):
            n0 = set(B.neighbors(sq[i]))
            n1 = set(B.neighbors(sq[i+1]))
            rep_qs = n0.intersection(n1)
            q0 = list(set(n0)-rep_qs)
            q1 = list(set(n1)-rep_qs)
            double_edgs += [(sq[i], e) for e in q0]
            double_edgs += [(sq[i], e) for e in q1]
        G = nx.Graph()
        G.add_nodes_from(sq[:-1])
        G.add_edges_from(double_edgs)
        sq_remain = list(s_nodes_set-set(sq))
        for s in sq_remain:
            G.add_edges_from([(s, e) for e in B.neighbors(s)])
        B = G
        s_nodes_set -= {sq[-1]}
    return B, s_nodes_set


def correct_logical(q,logicals_in, Sx_mat):
    logicals = list(np.copy(logicals_in))
    Ns_remain = np.size(Sx_mat,0)
    if len(logicals) == 1:
        if logicals[0][q]>0:
            if Ns_remain> 0:
                st_ind = np.argwhere(Sx_mat[:,q]>0)[:,0]
                if len(st_ind)>0:
                    logicals[0] = (logicals[0]+Sx_mat[st_ind[0],:]) % 2
                else:
                    logicals.pop()
            else:
                logicals.pop()
    else:
        for i_log in np.arange(len(logicals)-1,-1,-1):
            if logicals[i_log][q]>0:
                if Ns_remain> 0:
                    st_ind = np.argwhere(Sx_mat[:,q]>0)[:,0]
                    if len(st_ind)>0:
                        logicals[i_log] = (logicals[i_log]+Sx_mat[st_ind[0],:]) % 2
                    else:
                        logicals.pop(i_log)            
                else:
                    logicals.pop(i_log) 
    return logicals

def succ_prob_css_q_resolved(B_orig, logicals_in, s_nodes, loss_inds):
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
                    Sx_mat = []
                    # break
    
    return logic_list

def correct_logical_q_resolved(q,logicals_in, Sx_mat):
    logicals = list(np.copy(logicals_in))
    N_logicals = len(logicals)
    logic_removed = []
    Ns_remain = np.size(Sx_mat,0)
    if len(logicals) == 1:
        if logicals[0][q]>0:
            if Ns_remain> 0:
                st_ind = np.argwhere(Sx_mat[:,q]>0)[:,0]
                if len(st_ind)>0:
                    logicals[0] = (logicals[0]+Sx_mat[st_ind[0],:]) % 2
                else:
                    logicals.pop()
                    logic_removed.append(0)
            else:
                logicals.pop()
                logic_removed.append(0)

    else:
        # print(np.argwhere(np.array(logicals)[:,q]>0))
        # print( len(logicals) )
        # for i_l in range(len(logicals)):
        #     print(logicals[i_l][q])
        # print("------")
        log_inds = np.argwhere(np.array(logicals)[:,q]>0)[:,0]
        # for i_log in np.arange(len(logicals)-1,-1,-1):
        for i_log in log_inds[::-1]:
            # if logicals[i_log][q]>0:
            if Ns_remain> 0:
                st_ind = np.argwhere(Sx_mat[:,q]>0)[:,0]
                if len(st_ind)>0:
                    logicals[i_log] = (logicals[i_log]+Sx_mat[st_ind[0],:]) % 2
                else:
                    logicals.pop(i_log)
                    logic_removed.append(i_log)
            else:
                logicals.pop(i_log) 
                logic_removed.append(i_log)
                    
    logic_modified = list(set(range(N_logicals))-set(logic_removed))
    return logic_removed,logic_modified, logicals


def foliated_graph(S_mat,s_nodes, Nl,bdy=True,even_odd="odd"):
    # bdy = True: OBC and False: PBC
    # bdy = False only works with even_odd = "odd"
    Nq_l = np.size(S_mat,1) # number of data qubits per layer
    Ns_l = np.size(S_mat,0) # number of stabilizers per layer

    if even_odd == "odd":
        N = Nl*(Nq_l+Ns_l) # number of data qubits
    else:
        N = Nl*(Nq_l+Ns_l)+Ns_l # number of data qubits

    B_orig = nx.Graph()
    B_orig.add_nodes_from(np.arange(N))
    B_orig.add_nodes_from(s_nodes)
    for row in range(Ns_l):
        qs = np.argwhere(S_mat[row,:]>0)[:,0]
        for i_l in range(Nl):
            B_orig.add_edges_from([("s%d" % ((i_l*Ns_l)+row), i_l*(Nq_l+Ns_l)+q) for q in qs])

    if bdy:
        if even_odd == "odd":
            for i_l in range(Nl):
                B_orig.add_edges_from([("s%d" % ((i_l*Ns_l)+sq), i_l*(Nq_l+Ns_l)+Nq_l+sq) for sq in range(Ns_l)])
                if i_l> 0:
                    B_orig.add_edges_from([("s%d" % ((i_l*Ns_l)+sq), (i_l-1)*(Nq_l+Ns_l)+Nq_l+sq) for sq in range(Ns_l)])
        else:
            for i_l in range(Nl):
                B_orig.add_edges_from([("s%d" % ((i_l*Ns_l)+sq), i_l*(Nq_l+Ns_l)+Nq_l+sq) for sq in range(Ns_l)])
                if i_l> 0:
                    B_orig.add_edges_from([("s%d" % ((i_l*Ns_l)+sq), (i_l-1)*(Nq_l+Ns_l)+Nq_l+sq) for sq in range(Ns_l)])
                else:
                    B_orig.add_edges_from([("s%d" % ((i_l*Ns_l)+sq), Nl*(Nq_l+Ns_l)+sq) for sq in range(Ns_l)])
                    
    return B_orig