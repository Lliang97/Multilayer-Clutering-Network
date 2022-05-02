import argparse


def parse_args():
    """
    Parses the arguments.
    """
    parser = argparse.ArgumentParser(description="Multilayer Graph Contrastive Clustering Network")
    parser.add_argument('--dataset', nargs='?', default='acm', help='Input dataset')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate. Default is 1e-3')
    parser.add_argument('--seed', type=int, default=1, help='Seed for fixing the results')
    parser.add_argument('--n-epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--hidden-dims1', type=list, nargs='+', default=[786, 256], help='Number of dimensions1')
    parser.add_argument('--hidden-dims2', type=list, nargs='+', default=[786, 256], help='Number of dimensions2')
    parser.add_argument('--embedding', type=int, default=256, help='The dimension of hidden layer')
    parser.add_argument('--lambda_1', default=0.5, type=float, help='Edge reconstruction loss function')
    parser.add_argument('--lambda_2', default=10, type=float, help='Contrastive loss function')
    parser.add_argument('--lambda_3', default=0.5, type=float, help='Contrastive loss function')
    parser.add_argument('--beta_1', default=1, type=float, help='Combination coefficient')
    parser.add_argument('--beta_2', default=1, type=float, help='Combination coefficient')
    parser.add_argument('--beta_3', default=1, type=float, help='Combination coefficient')
    parser.add_argument('--cluster', default=3, type=float, help='The number of clusters')
    parser.add_argument('--n_sample', type=int, default=3025, help='The number of the samples')
    parser.add_argument('--alpha', type=float, default=1.0, help='Self supervised clustering parameter')
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout')
    parser.add_argument('--init', default=72, type=int, help='Fix initial centroids')
    parser.add_argument('--gradient_clipping', default=5.0, type=float, help='Gradient clipping')

    return parser.parse_args()