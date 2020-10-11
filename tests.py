import pytest
import eval
import numpy as np
import classification
import tree
import data_loader as dl


def test_count_leaf_nodes():
    leaf = tree.TLeaf(['A','A','B'])
    t = tree.TNode(None, None, leaf, tree.TNode(None,None, leaf, leaf))
    assert(t.count_leaf_nodes() == 3)


def test_prune():
    in_X, in_y = dl.load_data('data/simple2.txt')
    h = (len(in_X) // 2)
    X_valid = np.split(in_X, [h])[0]
    X = np.split(in_X, [h])[1]
    y_valid = np.split(in_y, [h])[0]
    y = np.split(in_y, [h])[1]
    t = classification.DecisionTreeClassifier()
    t.train(X, y)
    assert(t.root.get_depth() == 17)
    assert(t.prune(X_valid, y_valid, alpha=0.01, in_place=False).root.get_depth() == 5)
    assert(t.prune(X_valid, y_valid, alpha=1, in_place=False).root.get_depth() == 1)


def test_stop_level():
    in_X, in_y = dl.load_data('data/simple2.txt')
    for stop_level in range(2, 10):
        t = classification.DecisionTreeClassifier()
        t.train(in_X, in_y, stop_level=stop_level)
        assert(t.root.get_depth() == stop_level)
