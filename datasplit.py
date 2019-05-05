from scipy import sparse
import numpy as np
import csv

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

N = 4000
def traindata():
    dataset = csv.reader(open('./socpokec/soc-pokec-profiles-complete_per.csv', 'r', encoding='UTF-8'))
    train = []
    val = []
    test = []
    for data in dataset:
        if int(data[1])>50:
            train.append(int(data[0])-1)
        elif int(data[1])<=50 and int(data[1])>20:
            val.append(int(data[0])-1)
        else:
            test.append(int(data[0])-1)
    train_id = np.hstack(train)
    train_mask = sample_mask(train_id, N)
    np.savetxt('./onehotdata/train_mask.txt', train_mask, fmt='%d')
    val_id = np.hstack(val)
    val_mask = sample_mask(val_id, N)
    np.savetxt('./onehotdata/val_mask.txt', val_mask, fmt='%d')
    test_id = np.hstack(test)
    test_mask = sample_mask(test_id, N)
    np.savetxt('./onehotdata/test_mask.txt', test_mask, fmt='%d')

def counter():
    dataset = csv.reader(open('./socpokec/soc-pokec-profiles-complete_per.csv', 'r', encoding='UTF-8'))
    val = []
    test = []
    for data in dataset:
        if int(data[1]) <= 50 and int(data[1]) > 20:
            val.append(int(data[0]))
        elif int(data[1]) <= 20:
            test.append(int(data[0]))
    print(len(val))
    print(len(test))

def labeldel():
    label = sparse.load_npz('./onehotdata4000/onehotlabel.npz').toarray()
    mask = np.loadtxt('./onehotdata4000/test_mask.txt')
    tlabel = []
    zeros = [0, 0, 0, 0]
    for i, l in enumerate(label):
        if mask[i] == 1:
            tlabel.append(l)
        else:
            tlabel.append(zeros)
    np.savetxt('./onehotdata4000/y_test.txt', tlabel, fmt='%d')

def txtomask():
    train_id = np.loadtxt('./soc4000/soc-pokec-profiles-complete_trainID(4000).txt')
    train = []
    for ind in train_id:
        train.append(int(ind))
    train_mask = sample_mask(train, N)
    np.savetxt('./onehotdata4000/train_mask.txt', train_mask, fmt='%d')
    valTest = np.loadtxt('./soc4000/soc-pokec-profiles-complete_valTest(4000).txt')
    val = []
    test = []
    for i, id in enumerate(valTest):
        if i % 2 == 0:
            val.append(int(id))
        else:
            test.append(int(id))
    val_mask = sample_mask(val, N)
    np.savetxt('./onehotdata4000/val_mask.txt', val_mask, fmt='%d')
    test_mask = sample_mask(test, N)
    np.savetxt('./onehotdata4000/test_mask.txt', test_mask, fmt='%d')

def masktolabel():
    train_mask = np.loadtxt('./onehotdata4000/train_mask.txt', dtype=int)
    val_mask = np.loadtxt('./onehotdata4000/val_mask.txt', dtype=int)
    test_mask = np.loadtxt('./onehotdata4000/test_mask.txt', dtype=int)
    labels = sparse.load_npz('./onehotdata4000/onehotlabel.npz').toarray()
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    np.savetxt('./onehotdata4000/y_train.txt', y_train, fmt='%d')
    np.savetxt('./onehotdata4000/y_val.txt', y_val, fmt='%d')
    np.savetxt('./onehotdata4000/y_test.txt', y_test, fmt='%d')

if __name__ == '__main__':
    labeldel()