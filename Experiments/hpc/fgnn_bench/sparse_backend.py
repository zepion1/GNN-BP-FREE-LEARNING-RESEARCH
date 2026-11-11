"""Optional SparseTensor message passing for forwardgnn's models.

forwardgnn hands every convolution an `edge_index` tensor, so PyG first gathers one feature row per
edge (an E x C tensor) and then aggregates it. On Reddit2 (23.2M directed edges) that is ~12 GB per
128-channel tensor and ~56 GB for SAGE's first layer, which aggregates the 602 raw input features,
so most Reddit2 configurations cannot run on a 40 GB A100 that way.

`install()` converts each `edge_index` into a cached `torch_sparse.SparseTensor` (adj_t, rows are the
target nodes, i.e. PyG's `ToSparseTensor` layout) right before forwardgnn's GNN layers and PyG's
GCN / GraphSAGE / GAT models run. GCNConv and SAGEConv then aggregate with a sparse-dense matrix
product instead of materialising per-edge features; GATConv still builds per-edge attention messages,
because it has no fused kernel. The model computes the same function either way: GCN normalisation
with self-loops, mean aggregation and attention are unchanged; only the kernel and summation order
differ (floating-point round-off).

Use it for every dataset and method of a benchmark, or for none, so memory and time stay comparable.
"""
import weakref

import torch
from torch_sparse import SparseTensor


class AdjacencyCache:
    """edge_index tensor -> its SparseTensor, keyed by the live tensor object.

    Entries are dropped when the edge_index tensor is garbage collected, so a new tensor that happens
    to reuse a freed tensor's memory (e.g. the next run's augmented graph) can never get a stale
    adjacency."""

    def __init__(self):
        self._entries = {}

    def get(self, edge_index, num_nodes):
        key = id(edge_index)
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is edge_index and entry[1] == num_nodes:
            return entry[2]
        adj_t = SparseTensor(row=edge_index[1], col=edge_index[0], sparse_sizes=(num_nodes, num_nodes))
        ref = weakref.ref(edge_index, lambda _, key=key: self._entries.pop(key, None))
        self._entries[key] = (ref, num_nodes, adj_t)
        return adj_t

    def __len__(self):
        return len(self._entries)


CACHE = AdjacencyCache()


def _with_sparse_adjacency(forward):
    def forward_sparse(self, x, edge_index, *args, **kwargs):
        if isinstance(edge_index, torch.Tensor):
            edge_index = CACHE.get(edge_index, x.size(0))
        return forward(self, x, edge_index, *args, **kwargs)

    forward_sparse.fgnn_sparse = True
    return forward_sparse


def install():
    from gnn.gnn_conv import GNNConv  # layers of the forward-learning models
    from torch_geometric.nn.models.basic_gnn import BasicGNN  # GCN / GraphSAGE / GAT used by backprop

    for cls in (GNNConv, BasicGNN):
        if not getattr(cls.forward, 'fgnn_sparse', False):
            cls.forward = _with_sparse_adjacency(cls.forward)