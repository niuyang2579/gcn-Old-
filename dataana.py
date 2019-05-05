from __future__ import division
from __future__ import print_function

# import time
import tensorflow as tf

from utils import *
# from models import GCN, MLP


# import networkx as nx
# import pandas as pd
# import scipy as sp
import numpy as np
from scipy import sparse
from keras.utils import to_categorical



def ana():
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

    # Load data   ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

    # print(adj)  #邻接矩阵nx.adjacency_matrix  图邻接矩阵
    # print(features)      #
    # print(y_train)      #
    # print(y_val)      #
    # print(y_test)      #
    # print(train_mask.dtype)      #
    # print(val_mask.type)      #
    # print(test_mask.type)      #

    # f=open('./dataana/features.txt','w')
    # f.write(str(features))
    # f.close()

    np.savetxt('./dataana/y_test.txt',y_test,fmt='%d')
    # print(features.toarray().shape)

def load():
    matrix=np.loadtxt('./rawdata/AdjacencyMatrix_G_rebuild_polblogs_edge.txt')
    print(matrix.shape)
    # adj=sp.sparse.csr_matrix(matrix)
    # label=np.loadtxt('./rawdata/label_matrix_polblogs_node.txt')
    # onehot_label = to_categorical(label)



if __name__ == '__main__':
    # load()
    ana()