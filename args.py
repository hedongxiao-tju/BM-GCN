import re
import argparse
import torch


def parameter_parser():
    ap = argparse.ArgumentParser(description="BM-GCN.")

    ap.add_argument('--dataset', type=str, default='texas', help='Dataset.')
    ap.add_argument('--hidden_dim', type=int, default=64, help='Dimension of hidden embeddings.')
    ap.add_argument('--num_mlp_layers', type=int, default=2, help='Number pf layers for pre-train mlp')
    ap.add_argument('--num_gcn_layers', type=int, default=3, help='Number pf layers for gcn')
    ap.add_argument('--dropout_mlp', type=float, default=0.5, help='Dropout rate. Default is 0.5.')
    ap.add_argument('--dropout_gcn', type=float, default=0.7, help='Dropout rate. Default is 0.5.')
    ap.add_argument('--lr', type=float, default=0.001, help='Learning rate. Default is 0.005.')
    ap.add_argument('--loss_balance', type=str, default='1.0, 1.0', help='Balancing param of loss.')
    ap.add_argument('--weight_decay', type=float, default=0.0005, help='L2 regularization weight')
    ap.add_argument('--epoch_mlp', type=int, default=400, help='Number of epochs for pre-train mlp')
    ap.add_argument('--epoch_gcn', type=int, default=1600, help='Max number of epochs for gcn. Default is 1600.')
    ap.add_argument('--patience', type=int, default=100, help='Patience for early stopping.')
    ap.add_argument('--enhance', type=float, default=2.0, help='Enlarge diagonal in matrix Q')
    ap.add_argument('--self_loop', type=float, default=0.0, help='Enlarge diagonal in adj.')
    ap.add_argument('--seed', type=int, default=222, help='Seed.')
    ap.add_argument('--no_cuda', action='store_false', default=True,
                    help='Using CUDA or not. Default is True (Using CUDA).')

    args, _ = ap.parse_known_args()
    args.device = torch.device('cuda:0' if args.no_cuda and torch.cuda.is_available() else 'cpu')
    args.loss_weight = [float(x.strip()) for x in re.split(',', args.loss_balance)]

    return args
