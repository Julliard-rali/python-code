Compute the eigenvector centrality of a disconnected graph in python.
 
The module is implemented the way Matlab compute eigenvector centrality for disconnected graphs.

The 'eigenvector' centrality type uses the eigenvector corresponding to the
    largest eigenvalue of the graph adjacency matrix.
    The scores are normalized such that the sum of all centrality scores is 1.

If there are several disconnected components, then the algorithm computes
    the eigenvector centrality individually for each component, then scales
    the scores according to the percentage of graph nodes in that component.

Ref: https://www.mathworks.com/help/matlab/ref/graph.centrality.html

environment:
  Python: 3.10.6
  Networkx: 2.8.6
  Graph-tool: 2.45
  i-Graph: 0.10.1
  Numpy: 1.23.3
  
  
How to use:
eigen_cent = discon_gr_eigenctr(adjacency_matrix)

adjacency_matrix: a (nxn) numpy array of adjacency matrix (undirected graph, no loops)
eigen_cent: a (n,) numpy array of the corresponding eigenvalues
      # Example 1
    M4 = np.array([[0, 1, 1, 0],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0]])
                    
    eigen_cent = discon_gr_eigenctr(M4)

    # Example 2
    M5 = np.array([[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0]])
    eigen_cent = discon_gr_eigenctr(M5)

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
    eigen_cent = discon_gr_eigenctr(M7)

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
    eigen_cent = discon_gr_eigenctr(M8)
