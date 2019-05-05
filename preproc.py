#encoding:utf-8
import scipy as sp
import numpy as np
from scipy import sparse
from keras.utils import to_categorical
import csv
import xlrd

ATT_NUM = 21
FEA_NUM = 187
num_list = [3, 10, 12, 11, 18, 9, 6, 3, 4, 4, 7, 5, 11, 3, 3, 7, 5, 3, 17, 19, 27]
att = 1
# S = sparse.lil_matrix((1062796, 187))
S = sparse.lil_matrix((4000, 187))
# S = sparse.lil_matrix((1, 187))

def ismatch(features, word):
    features = features.split('/')
    for fea in features:
        if fea in word:
            return True
    return False

def labeltoMatrix(filename):
    dataset = csv.reader(open(filename, 'r', encoding='UTF-8'))
    S_label = sparse.lil_matrix((4000, 4))
    features = ["zenaty/vdana/vydata/manzel",
         "nejsem/nezenaty/nevydata/nemanzel/slobodn/zatial/volny/nezadany/vdov/window/nikdy",
    "mam pritel/vazny vztah/milovat/zalubeny",
    "other"]
    counter = 1
    for data in dataset:
        sig = 0
        if data[0] == counter:
            for ind in range(3):
                if ismatch(features[ind], data[1]):
                    S_label[int(data[0])-1, ind] = 1
            counter = counter + 1
        else:
            for ind in range(3):
                if ismatch(features[ind], data[1]):
                    S_label[int(data[0])-1, ind] = 1
            counter = int(data[0]) + 1
        for ind in range(3):
            if S_label[int(data[0])-1, ind] == 1:
                sig = 1
        if sig == 0:
            print("other")
            S_label[int(data[0])-1, 3] = 1
        print(int(data[0]), "is done!")
    # print(S_label.toarray())
    sparse.save_npz('./onehotdata4000/onehotlabel.npz', S_label.tocsr())

def getfeatures(filename):
    table = xlrd.open_workbook(filename).sheets()[0]
    nrows = table.nrows
    ncols = table.ncols
    a = table.row_values(0)
    for i in range(1, nrows):
        a_i = table.row_values(i)
        a = np.vstack((a, a_i))
    return a

def isother(features, data, att):
    for ind in range(num_list[att-1]):
        if ismatch(features[ind], data):
            return False
    return True

def decode(data, counter, features, att):
    for ind in range(num_list[att-1]):
        if features[ind] != 'other' and ismatch(features[ind], data):
            if att == 1:
                S[counter, ind] = 1
            else:
                S[counter, sum(num_list[0:att-1])+ind] = 1
        elif features[ind] == 'other':
            if isother(features, data, att) is True:
                if att == 1:
                    S[counter, ind] = 1
                else:
                    S[counter, sum(num_list[0:att-1])+ind] = 1

def kwtoMatrix(filename, feafilename):
    dataset = csv.reader(open(filename, 'r', encoding='UTF-8'))
    features = getfeatures(feafilename)
    # spamwriter = csv.writer(newfilename, dialect='excel')
    # spamwriter.writerow(data)
    counter = 1

    for data in dataset:
        if data[0] == str(counter):
            for att in range(1, 22):
                decode(data[att], counter-1, features[att-1], att)
            counter = counter+1
        else:
            for att in range(1, 22):
                decode(data[att], int(data[0])-1, features[att-1], att)
            counter = int(data[0]) + 1
        print(counter, "is done!")

    # sparse.save_npz('./onehotdata4000/feanoage.npz', S.tocsr())

    # csr_matrix_variable = sparse.load_npz('./onehotdata/onehot.npz')
    # print(csr_matrix_variable.toarray())

    # for fea in features:
    #     print(fea)

def agecvtcode(age):
    if age == 0: return 0
    if age > 0 and age <= 10: return 1
    if age > 10 and age <= 15: return 2
    if age > 15 and age <= 20: return 3
    if age > 20 and age <= 25: return 4
    if age > 25 and age <= 30: return 5
    if age > 30 and age <= 40: return 6
    if age > 40 and age <= 50: return 7
    if age > 50: return 8

def agetoMatrix(filename):
    dataset = csv.reader(open(filename, 'r', encoding='UTF-8'))
    S_age = sparse.lil_matrix((4000, 9))
    # S_age = sparse.lil_matrix((1, 9))
    counter = 1
    for data in dataset:
        if data[0] == str(counter):
            if data[1] == 'null':
                S_age[counter-1, 0] = 1
            else:
                S_age[counter-1, agecvtcode(int(data[1]))] = 1
            counter = counter+1
        else:
            if data[1] == 'null':
                S_age[int(data[0])-1, 0] = 1
            else:
                S_age[int(data[0])-1, agecvtcode(int(data[1]))] = 1
            counter = int(data[0])+1
        print(counter, "is done!")

    # sparse.save_npz('./onehotdata/onehotage.npz', S_age.tocsr())
    sparse.save_npz('./onehotdata4000/age.npz', S_age.tocsr())

def split():
    oh_fea = sparse.load_npz('./onehotdata4000/feanoage.npz')
    oh_age = sparse.load_npz('./onehotdata4000/age.npz')
    # oh_label = sparse.load_npz('./onehotdata/onehotlabel.npz')
    a = oh_fea.toarray()
    b = oh_age.toarray()
    c = np.hstack([a, b])
    features = sparse.csr_matrix(c)
    sparse.save_npz('./onehotdata4000/features.npz', features)

def adjload():
    # G = np.loadtxt('./socpokec/soc-pokec-relationships.txt')
    # rows = G[:, 0]-1
    # cols = G[:, 1]-1
    # ones = np.ones(len(G), np.uint32)
    # G_mat = sparse.coo_matrix((ones, (rows, cols)), shape=(1062796, 1062796))
    # print(G_mat.toarray())
    # sparse.save_npz('./onehotdata/adj.npz', sparse.csr_matrix(G))
    # node = []
    # for i in range(1, 1062797):
    #     node.append(i)
    # node_id = np.hstack(node)
    # np.savetxt('./onehotdata/node_id.txt', node_id, fmt='%d')
    G_mat = np.loadtxt('./soc4000/AdjacencyMatrix_soc-pokec-relationships-real_edge(4000).txt')
    adj = sp.sparse.csr_matrix(G_mat)
    sparse.save_npz('./onehotdata4000/adj.npz', adj)

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def input_data(train_nums, test_list):
    # adj
    G_mat = np.loadtxt('./rawdata/AdjacencyMatrix_G_rebuild_polblogs_edge.txt')
    adj = sp.sparse.csr_matrix(G_mat)

    # features
    # unit_mat = np.ones((1490, 1))
    # features = sp.sparse.csr_matrix(unit_mat)

    # label
    train_range = range(train_nums)
    val_range = range(train_nums, train_nums+500)

    labels = np.loadtxt('./rawdata/label_matrix_polblogs_node.txt')
    labels = to_categorical(labels)

    features = sp.sparse.csr_matrix(labels)

    train_mask = sample_mask(train_range, labels.shape[0])
    val_mask = sample_mask(val_range, labels.shape[0])
    test_mask = sample_mask(test_list, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # print(adj)  #邻接矩阵nx.adjacency_matrix  图邻接矩阵
    # print(features)      #
    # print(y_train)      #
    # print(y_val.shape)      #
    # print(y_test)      #
    # print(train_mask)      #
    # print(val_mask)      #
    # print(test_mask)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def datapre():
    adj = sparse.load_npz('./onehotdata4000/adj.npz')
    features = sparse.load_npz('./onehotdata4000/features.npz')
    y_train = np.loadtxt('./onehotdata4000/y_train.txt')
    y_val = np.loadtxt('./onehotdata4000/y_val.txt')
    y_test = np.loadtxt('./onehotdata4000/y_test.txt')
    train_mask = np.loadtxt('./onehotdata4000/train_mask.txt')
    val_mask = np.loadtxt('./onehotdata4000/val_mask.txt')
    test_mask = np.loadtxt('./onehotdata4000/test_mask.txt')
    print("Read Successfully!")
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

if __name__ == '__main__':
    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = input_data(200, range(1001,1490))
    # np.savetxt('./dataana/y_test.txt',y_test,fmt='%d')
    # kwtoMatrix('./soc4000/soc-pokec-profiles-input(4000).csv', './socpokec/features.xlsx')
    # agetoMatrix('./soc4000/soc-pokec-profiles-input-age(4000).csv')
    # labeltoMatrix('./soc4000/soc-pokec-profiles-marry(4000).csv')
    # split()
    adjload()
    # datapre()