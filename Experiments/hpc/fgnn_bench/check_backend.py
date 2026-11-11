"""Check that SparseTensor message passing computes what edge_index message passing computes.

    cd forwardgnn/src && python -m fgnn_bench.check_backend [--dataset WebKB-Washington] [--gpu 0]

Builds the models both training paths use (PyG GCN / GraphSAGE / GAT for backprop, forwardgnn's GNNConv
layers on the virtual-node graph for the forward methods), runs them on `edge_index`, installs the
sparse backend, runs the same parameters again and compares outputs and gradients. Exit code 1 on a
mismatch beyond floating-point round-off.
"""
import argparse
import gc
import sys

import torch
from torch_geometric.nn import GAT, GCN, GraphSAGE

from fgnn_bench import datasets_ext, sparse_backend


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='WebKB-Washington')
    parser.add_argument('--gpu', type=int, default=0, help='-1 for CPU')
    args = parser.parse_args(argv)
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')

    from gnn.gnn_conv import GNNConv
    from forward_learning.nodeclass.gnn_sf import GraphAugmenter

    datasets_ext.install()
    data = datasets_ext.load(args.dataset)[0].to(device)
    num_features, num_classes = data.num_features, int(data.y.max()) + 1
    train_mask = torch.rand(data.num_nodes, generator=torch.Generator().manual_seed(0)) < 0.64
    aug = GraphAugmenter(data=data, append_label=None, aug_edge_direction='bidirection',
                         device=device).augment(y=data.y, aug_node_mask=train_mask.to(device))

    torch.manual_seed(0)
    models = {
        'BP GCN': GCN(num_features, 128, num_layers=3, out_channels=num_classes, dropout=0.0, act='relu'),
        'BP SAGE': GraphSAGE(num_features, 128, num_layers=3, out_channels=num_classes, dropout=0.0, act='relu'),
        'BP GAT': GAT(num_features, 128, heads=4, num_layers=3, out_channels=num_classes, dropout=0.0, act='relu'),
        'SF GCN layer': GNNConv('GCN', num_features, 128),
        'SF SAGE layer': GNNConv('SAGE', num_features, 128),
        'SF GAT layer': GNNConv('GAT', num_features, 128),
    }
    for model in models.values():
        model.to(device)

    def run_all():
        results = {}
        for name, model in models.items():
            model.zero_grad()
            graph = data if name.startswith('BP') else aug
            out = model(graph.x, graph.edge_index)
            (out.pow(2).sum() + out.sum()).backward()
            results[name] = (out.detach().clone(), [p.grad.detach().clone() for p in model.parameters()])
        return results

    reference = run_all()
    sparse_backend.install()
    sparse = run_all()

    ok = True
    print(f'{args.dataset} on {device}: edge_index vs SparseTensor (same parameters)')
    for name in models:
        (y0, g0), (y1, g1) = reference[name], sparse[name]
        out_diff = (y0 - y1).abs().max().item()
        grad_diff = max((a - b).abs().max().item() / (a.abs().max().item() + 1e-12) for a, b in zip(g0, g1))
        good = torch.allclose(y0, y1, rtol=1e-4, atol=1e-5) and grad_diff < 1e-3
        ok &= good
        print(f'  {name:14s} max |output diff| {out_diff:.1e}   max relative grad diff {grad_diff:.1e}   '
              f'{"OK" if good else "MISMATCH"}')

    entries = len(sparse_backend.CACHE)
    del aug, data, reference, sparse
    gc.collect()
    print(f'  adjacency cache: {entries} entries while the test graphs were alive, {len(sparse_backend.CACHE)} '
          f'after releasing them (entries live exactly as long as their edge_index tensor)')
    print('backend check passed' if ok else 'backend check FAILED: use MP_BACKEND = "edge_index"')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()