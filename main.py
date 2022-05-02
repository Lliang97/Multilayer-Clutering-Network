import os
import tensorflow as tf
import numpy as np
from utils import config
from utils import process
from model.train import Trainer
import warnings

def main(args):

    # load data
    G_list, X, Y, idx_train, idx_val, idx_test = process.load_data(args.dataset)
    y_true = np.array([np.argmax(l) for l in Y])


    # prepare the data
    if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'wiki':
        A, S, R = process.prepare_graph_data(G_list)
        A2 = A
        S2 = S
        R2 = R
        X2 = X.dot(np.transpose(X))
    else:
        G = G_list[0]
        G2 = G_list[1]
        A, S, R = process.prepare_graph_data(G)
        A2, S2, R2 = process.prepare_graph_data(G2)
        X2 = X

    # n_sample
    args.n_sample = X.shape[0]

    # add input feature into hidden layer
    feature_dim1 = X.shape[1]
    args.hidden_dims1 = [feature_dim1] + args.hidden_dims1
    feature_dim2 = X2.shape[1]
    args.hidden_dims2 = [feature_dim2] + args.hidden_dims2

    # initial the cluster centers
    train = Trainer(args)
    _ = train.assign_cluster(A, X, S, R, A2, X2, S2, R2)

    # training the model
    train(A, X, S, R, A2, X2, S2, R2, y_true)


def setup(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    args = config.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    setup(args)
    main(args)