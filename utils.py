import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os

def pklLoad(fname):
    with open(fname, 'rb') as f:
        return pkl.load(f)

def pklSave(fname, obj):
    with open(fname, 'wb') as f:
        pkl.dump(obj, f)


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_action_zero_shot(dataset_str, w2v_type, split_ind, data_path = 'data', FLAGS=None):
    """Load data."""
    names = [w2v_type, 'labels', 'graph_all', 'graph_att', 'split_train', 'split_test', 'lookup_table']
    objects = []
    for i in range(len(names)):
        with open(data_path+"/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            print(data_path+"/ind.{}.{}".format(dataset_str, names[i]))
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    allx, ally, graph_all, graph_att, split_train, split_test, lookup_table_act_att = tuple(objects)
    zero_shot_train_classes = split_train[split_ind,:] -1
    zero_shot_test_classes = split_test[split_ind,:] -1

    features = allx  
    adj_all = nx.adjacency_matrix(nx.from_dict_of_lists(graph_all))
    adj_att = nx.adjacency_matrix(nx.from_dict_of_lists(graph_att))
    labels = np.array(ally) 

    idx_test = []
    idx_train = []
    y_train = []
    y_test = []

    for i in range(len(labels)):
        if labels[i] in zero_shot_train_classes:
            idx_train.append(i)
            y_train.append(labels[i])
        elif labels[i] in zero_shot_test_classes:
            if FLAGS.setting == "transductive":
                idx_train.append(i)      
                y_train.append(labels[i])
            idx_test.append(i)
            y_test.append(labels[i])
    idx_trainval = idx_train
    idx_val = idx_test
    y_trainval = y_train
    y_val = y_test


    train_mask = zero_shot_train_classes
    test_mask = zero_shot_test_classes
    label_num = len(train_mask)+len(test_mask)
    train_mask = sample_mask(train_mask, label_num)
    test_mask = sample_mask(test_mask, label_num)

    return adj_all, adj_att, features, y_train, y_val, idx_train, idx_val, train_mask, test_mask, lookup_table_act_att
def load_data_action_zero_shot_all(dataset_str, w2v_type, split_ind, data_path ):
    """Load data.
    Yahoo_100m-》w2v_type
    """
    names = [w2v_type, 'labels', 'graph_all', 'graph_att', 'split_train', 'split_test', 'lookup_table']
    objects = []
    for i in range(0,len(names)):
        with open(data_path+"/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            print(data_path+"/ind.{}.{}".format(dataset_str, names[i]))
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    allx, ally, graph_all, graph_att, split_train, split_test, lookup_table_act_att = tuple(objects)

    features = allx
    adj_all = nx.adjacency_matrix(nx.from_dict_of_lists(graph_all))
    adj_att = nx.adjacency_matrix(nx.from_dict_of_lists(graph_att))
    labels = np.array(ally)

    idx_val_s = {}
    idx_train_s = {}
    y_train_s = {}
    y_val_s = {}
    train_mask_s = {}
    test_mask_s = {}
    print('label max is {},min is {}'.format(max(labels),min(labels)))
    for split_inds in range(50):
        zero_shot_train_classes = split_train[split_inds,:] - 1
        zero_shot_test_classes = split_test[split_inds,:] - 1
        idx_test = []
        idx_train = []
        y_train = []
        y_test = []

        for i in range(len(labels)):
            if labels[i] in zero_shot_train_classes:
                idx_train.append(i)
                y_train.append(labels[i])
            elif labels[i] in zero_shot_test_classes:
                idx_test.append(i)
                y_test.append(labels[i])
        idx_trainval = idx_train
        idx_val = idx_test
        y_trainval = y_train
        y_val = y_test
        train_mask = zero_shot_train_classes
        test_mask = zero_shot_test_classes
        label_num = len(train_mask) + len(test_mask)
        train_mask = sample_mask(train_mask, label_num)
        test_mask = sample_mask(test_mask, label_num)
        idx_val_s[split_inds]  = idx_val
        idx_train_s[split_inds] = idx_train
        y_train_s[split_inds]   = y_train
        y_val_s[split_inds]    = y_val
        train_mask_s[split_inds]= train_mask
        test_mask_s[split_inds] = test_mask
        
    return adj_all, adj_att, features, y_train_s, y_val_s, idx_train_s, idx_val_s, train_mask_s, test_mask_s, lookup_table_act_att

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

def get_imageNet_input_data(data_set, time_interval, ini_seg_num, num_class, root):
    """preprocess the imageNet scores for all the data (train and test), merge
       the scores for a fixed time interval. The final segment length of a video
       is ini_seg_length/time_interval
       """
    save_file_name = root + '/input_data_imageNet_scores_' + data_set.lower() + '.pkl'
    saved_data = pklLoad(save_file_name)
    all_inds = saved_data[0]
    all_scores = saved_data[1]
    return all_inds, all_scores
