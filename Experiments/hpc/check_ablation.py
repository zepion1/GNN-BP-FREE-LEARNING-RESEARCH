#!/usr/bin/env python3
"""Self-test for the virtual-node ablation, run against the real forwardgnn code on PubMed. Takes seconds.

    source $FGNN_ROOT/activate.sh && $FGNN_PY $FGNN_ROOT/check_ablation.py

Checks, on the actual graph the experiments use:
  SF       bidirection adds 2 edges per training node (to and from its class's virtual node)
  SF-uni   unidirection adds 1 edge per training node, into the virtual node only
  SF-noVN  no virtual edges at all; each layer outputs real-node embeddings followed by its own prototypes,
           the prototypes are trained by the layer's optimiser and saved in its early-stopping checkpoint,
           and the model factory routes the new name to the new class (and only the new name).
Runs on CPU if no GPU is visible.
"""
import argparse
import os
import sys

src = os.environ['FGNN_SRC']
os.chdir(src)
sys.path.insert(0, src)

import torch  # noqa: E402

failures = 0


def check(condition, message):
    global failures
    print(('  ok    ' if condition else '  FAIL  ') + message)
    if not condition:
        failures += 1


from datasets.dataloader import load_dataset  # noqa: E402
from forward_learning.nodeclass import gnn_sf  # noqa: E402
from fgnn_bench import sf_novn  # noqa: E402

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = load_dataset('CitationFull-PubMed')[0].to(dev)
N, E, C = data.num_nodes, data.edge_index.size(1), int(data.y.max()) + 1
data.num_classes = C
gen = torch.Generator().manual_seed(0)
train_mask = torch.zeros(N, dtype=torch.bool)
train_mask[torch.randperm(N, generator=gen)[: N // 5]] = True
train_mask = train_mask.to(dev)
T = int(train_mask.sum())
print(f'PubMed on {dev}: {N:,} nodes, {E:,} directed edges, {C} classes, {T:,} training nodes (test mask)')

NoVNAugmenter, NoVNLayer, NoVNModel = sf_novn.classes()
Base = gnn_sf.GraphAugmenter
check(Base is not NoVNAugmenter, 'forwardgnn augmenter is untouched before an SF-noVN model is built')


def augment(cls, direction):
    return cls(data=data, append_label=None, aug_edge_direction=direction, device=dev).augment(
        y=data.y, aug_node_mask=train_mask)


print('\n= graphs')
bi, uni, nov = augment(Base, 'bidirection'), augment(Base, 'unidirection'), augment(NoVNAugmenter, 'bidirection')
check(bi.edge_index.size(1) == E + 2 * T, f'SF: {E:,} real + {2 * T:,} virtual edges (both directions)')
check(uni.edge_index.size(1) == E + T, f'SF-uni: {E:,} real + {T:,} virtual edges')
check(int((uni.edge_index[0] >= N).sum()) == 0 and int((uni.edge_index[1] >= N).sum()) == T,
      'SF-uni: virtual nodes only receive messages, never send them')
check(nov.edge_index.size(1) == E and torch.equal(nov.edge_index, data.edge_index),
      'SF-noVN: exactly the original edges')
check(int((nov.edge_index >= N).sum()) == 0, 'SF-noVN: no edge touches a virtual node')
check(nov.x.size(0) == N + C and int(nov.real_node_mask.sum()) == N,
      'SF-noVN: C placeholder rows keep the (N + C) layout single forward expects')

print('\n= layer')
args = argparse.Namespace(temperature=1.0, device=dev, patience=100, val_every=2, append_label=None,
                          aug_edge_direction='bidirection')
layer = NoVNLayer(gnn_layer=gnn_sf.GNNConv(gnn='GCN', in_channels=data.num_features, out_channels=128),
                  optimizer_name='Adam', optimizer_kwargs={'lr': 0.001, 'weight_decay': 5e-4},
                  args=args, num_classes=C, out_channels=128).to(dev)
out = layer(nov.x, nov.edge_index, nov.edge_type)
check(tuple(out.shape) == (N + C, 128), f'output is (N + C) x 128 = {tuple(out.shape)}')
check(torch.equal(out[N:], layer.prototypes), 'last C rows are the layer\'s prototypes')
check('prototypes' in layer.state_dict(), 'prototypes are saved in the early-stopping checkpoint')
in_opt = {id(p) for group in layer.optimizer.param_groups for p in group['params']}
check(id(layer.prototypes) in in_opt, 'prototypes are updated by the layer\'s optimiser')
check(all(id(p) in in_opt for p in layer.gnn_layer.parameters()), 'GNN weights are still in the optimiser')

augmenter = NoVNAugmenter(data=data, append_label=None, aug_edge_direction='bidirection', device=dev)
augmenter.augment(y=data.y, aug_node_mask=train_mask)
before_p = layer.prototypes.detach().clone()
before_w = [p.detach().clone() for p in layer.gnn_layer.parameters()]
loss = layer.forward_train(nov.x, augmenter, train_mask, nov.edge_index, nov.edge_type)
check(loss == loss and loss > 0, f'one training step runs (loss {loss:.4f})')
check(not torch.equal(before_p, layer.prototypes.detach()), 'the training step moved the prototypes')
check(any(not torch.equal(b, p.detach()) for b, p in zip(before_w, layer.gnn_layer.parameters())),
      'the training step moved the GNN weights')

print('\n= model factory')
import train_forward  # noqa: E402

sf_novn.install(train_forward)
build = train_forward.build_node_classification_model
plain = build(model_type='GNN_SingleForward-GCN', n_layers=2, hidden_size=128, loss_fn_name=None, lr=0.001,
              data=data, args=args)
check(type(plain) is gnn_sf.GNNSingleForward and gnn_sf.GraphAugmenter is Base,
      'GNN_SingleForward-GCN still builds plain single forward and leaves its augmenter alone')
model = build(model_type='GNN_SingleForwardNoVN-SAGE', n_layers=3, hidden_size=128, loss_fn_name=None, lr=0.001,
              data=data, args=args)
check(isinstance(model, NoVNModel) and isinstance(model, gnn_sf.GNNSingleForward),
      'GNN_SingleForwardNoVN-SAGE builds SF-noVN (a single-forward subclass, so run_bench\'s hooks apply)')
check(len(model.layers) == 3 and all(isinstance(m, NoVNLayer) for m in model.layers), '3 SF-noVN layers')
check(type(model).forward_train is gnn_sf.GNNSingleForward.forward_train
      and type(model).eval_model is gnn_sf.GNNSingleForward.eval_model,
      'training loop and evaluation are forwardgnn\'s own, unmodified')
check(gnn_sf.GraphAugmenter is NoVNAugmenter, 'building SF-noVN switches this process to the no-virtual-edge graph')

print('\nALL CHECKS PASSED' if failures == 0 else f'\n{failures} CHECK(S) FAILED')
sys.exit(1 if failures else 0)
