import numpy as np
import data_loader as dl
import sys
from classification import DecisionTreeClassifier
from eval import Evaluator

def split_data(X, Y, k):
    N, C = X.shape
    if k <= 1:
        return [X], [Y]
    arr = np.arange(N)
    np.random.shuffle(arr)
    Xs = np.zeros((N, C))
    Ys = np.empty((N,), np.str)
    for i in range(N):
        Xs[i] = X[arr[i]]
        Ys[i] = Y[arr[i]]

    a, b = divmod(N, k)
    Xsplit = []
    Ysplit = []
    for i in range(k):
        if i < b:
            start = i*(a + 1)
            small = 1
        else:
            start = i*a + b
            small = 0
        Xsplit.append(Xs[start:(start + a + small)])
        Ysplit.append(Ys[start:(start + a + small)])

    return Xsplit, Ysplit


def train_and_test(trainX, trainY, testX, testY):
    tree = DecisionTreeClassifier()
    tree.train(trainX, trainY)
    predictions = tree.predict(testX)
    e = Evaluator()
    confusion = e.confusion_matrix(predictions, testY)
    accuracy = e.accuracy(confusion)
    return accuracy, tree


file = sys.argv[1]
k = int(sys.argv[2])
X, Y = dl.load_data(file)
Xsplit, Ysplit = split_data(X, Y, k)
accs = []
trees = []
for i in range(k):
    testX = Xsplit[i]
    testY = Ysplit[i]
    trainX = np.concatenate(Xsplit[:i] + Xsplit[i+1:])
    trainY = np.concatenate(Ysplit[:i] + Ysplit[i + 1:])
    acc , tree = train_and_test(trainX, trainY, testX, testY)
    accs.append(acc)
    trees.append(tree)

avg_acc = np.mean(accs)
stdev = np.std(accs)
print(avg_acc, u"\u00B1", stdev)
max_i = accs.index(max(accs))
max_tree = trees[max_i]
if len(sys.argv) > 3:
    dl.save_tree(max_tree, sys.argv[3])
if len(sys.argv) > 4:
    dl.save_tree(trees, sys.argv[4])
