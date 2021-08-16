import numpy as np
import scipy.sparse as sp

def load_dataset(file_name):
    """Load a graph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph

g = load_dataset('cora_ml.npz')
A, X, z = g['A'], g['X'], g['z']
node_id = np.arange(2995).reshape(2995, 1)
A = A.A
X = X.A
X = np.hstack((node_id, X, z.reshape(2995, 1)))
# with open("cora.cites", 'w') as f:
#     for index1, row in enumerate(A):
#         for index2, column in enumerate(row):
#             if column == 1:
#                 f.write(f'{index1} {index2}\n')
with open("cora.content", 'w') as f:
    size = len(X[0])
    for row in X:
        for index, column in enumerate(row):
            f.write(str(column))
            f.write(' ' if index != size-1 else '\n')