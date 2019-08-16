import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))

def res():
    # 'sopokec', 'citeseer', 'cora', 'pubmed'
    name = ['sopokec', 'citeseer', 'cora', 'pubmed']
    accuracy = np.array([0.61636, 0.70400, 0.80800, 0.77700])
    a = plt.bar(range(len(accuracy)), accuracy, tick_label = name)
    autolabel(a)
    plt.ylim([0, 1])
    plt.title('Accuracy on four datasets')
    plt.show()

def count(train_mask, val_mask, test_mask):
    train_mask = np.loadtxt('./onehotdata4000/train_mask.txt')
    val_mask = np.loadtxt('./onehotdata4000/val_mask.txt')
    test_mask = np.loadtxt('./onehotdata4000/test_mask.txt')
    tr = 0
    va = 0
    te = 0
    for train in train_mask:
        if train == 1:
            tr = tr + 1
    for val in val_mask:
        if val == 1:
            va = va + 1
    for test in test_mask:
        if test == 1:
            te = te + 1
    print(tr)
    print(va)
    print(te)
def lr():
    # learning rate: 0.01
    accuracy = [0.59793]

res()