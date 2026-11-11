"""SF-noVN: single forward with the virtual nodes removed (ablation).

Model name: GNN_SingleForwardNoVN-<GCN|SAGE|GAT>, trained by train_forward.py through run_bench.

What virtual nodes do in forwardgnn's single forward (forward_learning/nodeclass/gnn_sf.py):
  1. training target  each layer's loss is a softmax over dot products between training-node embeddings and
                      the C virtual-node embeddings (cross-entropy on the true labels);
  2. classifier       a node is predicted as its most similar virtual node, softmax summed over the layers;
  3. label flow       GraphAugmenter adds C nodes and links every training node to its class's virtual node
                      (both directions with --aug-edge-direction bidirection), so labels enter message passing
                      and each class prototype is computed by message passing from that class's members.

SF-noVN removes the virtual nodes from the graph completely (no virtual edges: role 3 is gone) and gives every
layer C learnable vectors, used exactly where the virtual-node embeddings were in roles 1-2. They are
initialised like a bias-free nn.Linear(hidden, C) and trained by the layer's own optimiser (Adam, same lr and
weight decay). In other words: greedy layer-wise training with a local linear classifier per layer.

Everything else is forwardgnn's own code, unmodified: the layer-by-layer training loop, the loss (dot product,
temperature, cross-entropy), early stopping and its checkpoints, and the summed-softmax prediction.

How it plugs in: single forward's layers return an (N + C) x d table, real nodes first, virtual nodes last, and
gnn_sf slices the last C rows out as prototypes. Here each layer runs the GNN on the N real nodes and the
original edges only, then appends its C prototype vectors as those last rows. The augmenter keeps C placeholder
rows in x (so every mask and index in gnn_sf lines up) but strips all virtual edges.

The SF-uni ablation needs no code: it is single forward with --aug-edge-direction unidirection (virtual nodes
still aggregate their class members, but send nothing back). run_bench labels those runs 'SF-uni'.
"""
import math

import torch
import torch.nn as nn

MODEL_KEY = 'singleforwardnovn'  # matched case-insensitively against the model name
_classes = None


def classes():
    """Create (once) and return (NoVNGraphAugmenter, NoVNLayer, NoVNModel), built on forwardgnn's gnn_sf."""
    global _classes
    if _classes is not None:
        return _classes
    from forward_learning.nodeclass import gnn_sf

    base_augmenter = gnn_sf.GraphAugmenter  # captured before install() replaces the module attribute

    class NoVNGraphAugmenter(base_augmenter):
        """forwardgnn's augmenter with every virtual edge removed; the C placeholder rows stay in x."""

        def augment(self, *args, **kwargs):
            graph = super().augment(*args, **kwargs)
            graph.edge_index = self.edge_index  # the original graph's edges only
            graph.edge_type = torch.zeros(self.edge_index.size(1), dtype=torch.long,
                                          device=self.edge_index.device)
            return graph

    class NoVNLayer(gnn_sf.GNNSingleForwardLayer):
        """A single-forward layer whose class prototypes are learnable vectors, not virtual-node embeddings."""

        def __init__(self, gnn_layer, optimizer_name, optimizer_kwargs, args, num_classes, out_channels):
            super().__init__(gnn_layer, optimizer_name, optimizer_kwargs, args)
            self.num_classes = int(num_classes)
            self.prototypes = nn.Parameter(torch.empty(self.num_classes, out_channels))
            nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))  # nn.Linear's default weight init
            # same optimiser as the parent layer, now also updating the prototypes
            self.optimizer = getattr(torch.optim, optimizer_name)(
                list(gnn_layer.parameters()) + [self.prototypes], **optimizer_kwargs)

        def forward(self, x, edge_index, edge_type=None):
            real = super().forward(x[:-self.num_classes], edge_index, edge_type)  # drop placeholder rows
            return torch.cat([real, self.prototypes], dim=0)  # prototypes take the virtual nodes' place

    class NoVNModel(gnn_sf.GNNSingleForward):
        def __init__(self, layer_sizes, optimizer_name, optimizer_kwargs, num_classes, gnn, args):
            super().__init__(layer_sizes=layer_sizes, optimizer_name=optimizer_name,
                             optimizer_kwargs=optimizer_kwargs, num_classes=num_classes, gnn=gnn, args=args)
            if self.append_label is not None:
                raise ValueError('SF-noVN supports --append-label none only')
            self.layers = nn.ModuleList([
                NoVNLayer(gnn_layer=gnn_sf.GNNConv(gnn=gnn, in_channels=layer_sizes[i],
                                                   out_channels=layer_sizes[i + 1]),
                          optimizer_name=optimizer_name, optimizer_kwargs=optimizer_kwargs, args=args,
                          num_classes=num_classes, out_channels=layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ])

    _classes = (NoVNGraphAugmenter, NoVNLayer, NoVNModel)
    return _classes


def install(train_module):
    """Route GNN_SingleForwardNoVN-* model names in train_forward.py to NoVNModel.

    Without this, train_forward's factory would match 'SingleForward' and silently build plain single forward.
    Every other model name goes to the original factory untouched.
    """
    original = train_module.build_node_classification_model

    def build_node_classification_model(model_type, n_layers, hidden_size, loss_fn_name, lr, data, args):
        if MODEL_KEY not in model_type.lower():
            return original(model_type=model_type, n_layers=n_layers, hidden_size=hidden_size,
                            loss_fn_name=loss_fn_name, lr=lr, data=data, args=args)
        assert n_layers >= 1, n_layers
        from forward_learning.nodeclass import gnn_sf
        augmenter, _, model_cls = classes()  # must run before the swap below
        if gnn_sf.GraphAugmenter is not augmenter:  # this process trains SF-noVN only
            gnn_sf.GraphAugmenter = augmenter
            print('[sf-noVN] virtual nodes removed: no virtual edges; one learnable prototype per class per layer',
                  flush=True)
        return model_cls(layer_sizes=[data.num_features] + [hidden_size] * n_layers,
                         optimizer_name='Adam', optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
                         num_classes=data.num_classes, gnn=model_type.split('-')[-1], args=args)

    train_module.build_node_classification_model = build_node_classification_model
