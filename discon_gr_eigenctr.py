#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 20:55:42 2022

@author: julliard

Motives:
    Available Python packages to compute eigenvector centrality are not able
    to handle the computation of disconnected graphs.

Current version
    Python: 3.10.6
    Networkx: 2.8.6
    Graph-tool: 2.45
    i-Graph: 0.10.1
    Numpy: 1.23.3


Those functions are implemented the way Matlab handle such graphs as:

The 'eigenvector' centrality type uses the eigenvector corresponding to the
    largest eigenvalue of the graph adjacency matrix.
    The scores are normalized such that the sum of all centrality scores is 1.

If there are several disconnected components, then the algorithm computes
    the eigenvector centrality individually for each component, then scales
    the scores according to the percentage of graph nodes in that component.

Ref: https://www.mathworks.com/help/matlab/ref/graph.centrality.html

"""

import numpy as np
# from numba import jit # All following functions are numba ready


# @jit(nopython=True)
def neighb_nodes(node, adj_mat, neighbors):

    for i1 in range(node, np.shape(neighbors)[0], 1): # Row
        if adj_mat[node, i1] != 0:
            neighbors[i1] = 1

    for i1 in range(0, node, 1):                      # Col
        if adj_mat[i1, node] != 0:
            neighbors[i1] = 1

    return neighbors


# @jit(nopython=True)
def node_grp(adj_mat):
    '''
    Find and label connected component of the graph
    Algorithm:
        - Declare all nodes as unvisited
        - Pick a starting node and declare it as visited
        - Find all its neighbours
        - Remove already visited node among neighbours
        - Pick one node among neighbours as starting point and declare it as visited
        - Stop when all neighbours have been visited

    '''
    XX = np.shape(adj_mat)[0]                   # Initialization
    neighbors = np.zeros(XX, dtype=np.int8)

    neighbors[0] = 1                            # Start with the first node on the list
    group_count = 101                           # Start from 101 as label for the group
    vertex_grp = np.zeros(XX, dtype=np.int16)   # Group Zero for un visited node

    while np.count_nonzero(vertex_grp) != XX:

        if np.sum(neighbors) == 0:
            group_count +=1                                    # Increment the group number counter
            start_pt = np.where(vertex_grp==0)[0][0]           # Pick a new starting node
        else:
            start_pt = np.where(neighbors==1)[0][0]            # New starting node

        vertex_grp[start_pt] = group_count                     # declare the node as visited and assign group
        neighbors = neighb_nodes(start_pt, adj_mat, neighbors) # find the neighbor
        neighbors[vertex_grp != 0] = 0                         # Remove the previously visited node

    return vertex_grp


# @jit(nopython=True)
def singl_eig_ctr(grp_adj_mat):
    eig_val, eig_vect = np.linalg.eigh(grp_adj_mat)  # Direct Linear Algebra computation
    xx = np.where(eig_val == np.max(eig_val))[0][0]  # eigenvector corresponding to the maximum eigenvalue
    eig_ctr = eig_vect[:, xx]
    eig_ctr = eig_ctr/np.sum(eig_ctr)                # Normalize

    return eig_ctr


# @jit(nopython=True)
def grp_eig_ctr(vertex_grp, adj_mat):

    adj_mat = adj_mat.astype(np.double)  # Cast to float for numba to digest

    if np.max(vertex_grp)-101+1 == 1:
        # Only one group
        eigen_ctr = singl_eig_ctr(adj_mat)

    else:
        # Compute the centrality per group and put back the values to the original order
        eigen_ctr_0 = np.zeros(np.shape(adj_mat)[0], dtype=np.double)
        for i1 in range(101, np.max(vertex_grp)+1, 1):
            group_num = i1
            group_list = np.where(vertex_grp==group_num)[0]

            grp_adj_mat = adj_mat[group_list, :]                                       # Line
            grp_adj_mat = grp_adj_mat[:, group_list]                                   # Column
            eigen_ctr = singl_eig_ctr(grp_adj_mat)                                     # Compute the centrality
            eigen_ctr = eigen_ctr * (np.shape(grp_adj_mat)[0] / np.shape(adj_mat)[0])  # Normalize the group's eigenvector

            for i2 in range(0, np.shape(group_list)[0], 1):                            # Put back to the original order
                eigen_ctr_0[group_list[i2]] = eigen_ctr[i2]

        eigen_ctr = eigen_ctr_0

    return eigen_ctr


def discon_gr_eigenctr(adj_mat):
    '''
    input: (nxn) numpy array of adjacency matrix (direct graph no loop)
    return: a numpy array of (n,) containing eigencentralies of each node
    '''
    vertex_grp = node_grp(adj_mat)
    eigen_ctr = grp_eig_ctr(vertex_grp, adj_mat)

    return eigen_ctr





if __name__ == '__main__':

    # Example 1
    M4 = np.array([[0, 1, 1, 0],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0]])

    # Example 2
    M5 = np.array([[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0]])

    # Example 3: One-component-graph
    M6 = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    M6 = M6 + np.transpose(M6)

    # Example 4: 2-components-graph
    M7 = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    M7 = M7 + np.transpose(M7)

    # Example 5: 3-components-graph
    M8 = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.double)
    M8 = M8 + np.transpose(M8)


    # Call the function
    eigen_cent = discon_gr_eigenctr(M8)