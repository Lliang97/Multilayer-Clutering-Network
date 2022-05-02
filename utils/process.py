import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
import sys
from scipy import sparse


def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    # print("prepare_adj:", adj)
    # data =  adj.tocoo().data
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    # print("tocoo_adj:", adj)
    adj = adj.astype(np.float32)
    # print("astype_adj:", adj)
    indices = np.vstack((adj.col, adj.row)).transpose()
    # print("indices", indices)
    # print(adj.row)
    # print(adj.col)
    return (indices, adj.data, adj.shape), adj.row, adj.col
    # return (indices, adj.data, adj.shape), adj.row, adj.col, data#adj.data


def prepare_graph_data1(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    # data =  adj.tocoo().data
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col
    # return (indices, adj.data, adj.shape), adj.row, adj.col, data#adj.data


def prepare_sparse_features(features):
    if not sp.isspmatrix_coo(features):
        features = sparse.csc_matrix(features).tocoo()
        features = features.astype(np.float32)
    indices = np.vstack((features.row, features.col)).transpose()
    return (indices, features.data, features.shape)


def conver_sparse_tf2np(input):
    # Convert Tensorflow sparse matrix to Numpy sparse matrix
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])),
                          shape=(input[layer][2][0], input[layer][2][1])) for layer in input]


###############################################
# This section of code adapted from tkipf/gcn #
###############################################


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_multiattribute(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix cl + citeseer + cp dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    nx_graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(nx_graph)
    # print(type(adj))
    # print("citeseer_adj:",adj)
    edges = nx_graph.edges()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    # print(sp.coo_matrix(adj))
    return sp.coo_matrix(adj), features.todense(), labels, idx_train, idx_val, idx_test


def load_multigraph(dataset):

    data = sio.loadmat('data/acm/acm.mat')
    # feature
    feature = data['feature']
    features = sp.csr_matrix(feature, dtype=np.float32)

    labels = data['label']
    num_nodes = data['label'].shape[0]

    data['PAP'] = sparse.coo_matrix(data['PAP'] + np.eye(num_nodes))
    data['PAP'] = data['PAP'].todense()
    data['PAP'][data['PAP'] > 0] = 1.0
    adj1 = sparse.coo_matrix(data['PAP'] - np.eye(num_nodes))
    data['PLP'] = sparse.coo_matrix(data['PLP'] + np.eye(num_nodes))
    data['PLP'] = data['PLP'].todense()
    data['PLP'][data['PLP'] > 0] = 1.0
    adj2 = sparse.coo_matrix(data['PLP'] - np.eye(num_nodes))

    PAP = np.stack((np.array(adj1.row), np.array(adj1.col)), axis=1)
    PLP = np.stack((np.array(adj2.row), np.array(adj2.col)), axis=1)
    # print(PAP)
    # print(PLP)

    #
    PAPedges = np.array(list(PAP), dtype=np.int32).reshape(PAP.shape)
    PAP_adj = sp.coo_matrix((np.ones(PAPedges.shape[0]), (PAPedges[:, 0], PAPedges[:, 1])), shape=(num_nodes, num_nodes), dtype=np.float32)
    PAP_adj = PAP_adj + PAP_adj.T.multiply(PAP_adj.T > PAP_adj) - PAP_adj.multiply(PAP_adj.T > PAP_adj)
    PAP_normalize_adj = normalize(PAP_adj)
    # print(PAP_normalize_adj)

    PLPedges = np.array(list(PLP), dtype=np.int32).reshape(PLP.shape)
    PLP_adj = sp.coo_matrix((np.ones(PLPedges.shape[0]), (PLPedges[:, 0], PLPedges[:, 1])), shape=(num_nodes, num_nodes), dtype=np.float32)
    PLP_adj = PLP_adj + PLP_adj.T.multiply(PLP_adj.T > PLP_adj) - PLP_adj.multiply(PLP_adj.T > PLP_adj)
    PLP_normalize_adj = normalize(PLP_adj)
    # print(PLP_normalize_adj)


    adj_list = [PAP_normalize_adj, PLP_normalize_adj]

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, features.todense(), labels, idx_train, idx_val, idx_test


def load_acm():

    data = sio.loadmat('data/acm/acm.mat')
    # feature
    feature = data['feature']
    features = sp.csr_matrix(feature, dtype=np.float32)

    labels = data['label']
    num_nodes = data['label'].shape[0]

    # adj1 = sparse.coo_matrix(data['PAP'])
    # adj2 = sparse.coo_matrix(data['PLP'])
    adj1 = sparse.coo_matrix(data['PAP'] + np.eye(num_nodes))
    adj2 = sparse.coo_matrix(data['PLP'] + np.eye(num_nodes))


    adj_list = [adj1, adj2]

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, features.todense(), labels, idx_train, idx_val, idx_test

def load_data(dataset):
    if dataset == 'acm' or dataset == 'dblp' or dataset == 'imdb':
        return load_multigraph(dataset)
    else:
        return load_multiattribute(dataset)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack(
        (adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape


def normalize(mx):
    """Row-normalize sparse matrix"""
    epsilon = 1e-5
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
