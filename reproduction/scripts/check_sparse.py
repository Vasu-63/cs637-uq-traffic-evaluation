# Optional check for the critique: does _build_sparse_matrix keep each value with its own (row, col)?
# Run from inside a variant folder.
import os, sys, numpy as np, torch
sys.path.insert(0, os.getcwd())
from lib import utils
from model.pytorch.dcrnn_cell import DCGRUCell
_, _, adj = utils.load_graph_data('data/sensor_graph/adj_mx.pkl')
for name, L in [('forward random walk', utils.calculate_random_walk_matrix(adj).T),
                ('backward random walk', utils.calculate_random_walk_matrix(adj.T).T)]:
    S = DCGRUCell._build_sparse_matrix(L).to_dense().cpu().numpy()
    D = L.toarray()
    print(f'{name}: max |torch matrix - intended matrix| = {np.abs(S - D).max():.4f} '
          f'(max entry {np.abs(D).max():.4f}); same sparsity pattern: {np.array_equal(S != 0, D != 0)}')
