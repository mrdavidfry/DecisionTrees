from entropy import entropy, IG
import numpy as np
import data_loader as dl
import collections
import matplotlib.pyplot as plt
import matplotlib


def get_majority_element(labels):
    return max(collections.Counter(labels).items(), key=lambda tup: tup[1])[0]


class TNode(object):
    def copy(self):
        return TNode(
            self.rule,
            self.treshold,
            self.left.copy(),
            self.right.copy()
            )

    def __init__(self, rule, treshold, left, right):
        self.leaf = False
        self.left = left
        self.right = right
        self.rule = rule
        self.treshold = treshold

    def predict(self, x):
        return self.left.predict(x) if self.treshold >= x[self.rule] else self.right.predict(x)

    def to_string(self, structure):
        return (
            ''.join(['   │' if v else '    ' for v in structure[:-1]])
            + ('   ├───?' if structure[-1] else '   └───?')
            + ' x[{rule}] >= {treshold} \n'.format(rule=self.rule, treshold=self.treshold)
            + self.left.to_string(structure+[True])
            + self.right.to_string(structure+[False])
            )

    def count_leaf_nodes(self):
        # implemented for pruning
        return self.left.count_leaf_nodes() + self.right.count_leaf_nodes()

    def get_depth(self):
        return max(self.left.get_depth(), self.right.get_depth()) + 1

    def draw(self, fig = None, ax = None, level=0, xfrom=0, xto=1):
        if fig is None:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1,1,1)
        center = (xfrom + xto) / 2
        y = 0.975 - 0.05*level
        rect = plt.Rectangle((center, y), 0.1, 0.025,facecolor="red")
        ax.add_patch(rect)
        ax.text(center, y, "x[{}]<={}".format(self.rule, self.treshold),fontsize=12)
        self.left.draw(fig=fig, ax=ax, level=level+1, xfrom=xfrom, xto=center)
        self.right.draw(fig=fig, ax=ax, level=level+1, xfrom=center, xto=xto)

    def prune(self, X_valid, y_valid, root_classifier, parent=None, am_i_left = None, alpha=0):
        self.left.prune(X_valid, y_valid, root_classifier, self, am_i_left=True, alpha=alpha)
        self.right.prune(X_valid, y_valid, root_classifier, self, am_i_left=False, alpha=alpha)
        if self.left.leaf and self.right.leaf and parent is not None:
            # I am a potentially prunable node
            R_alpha_original = root_classifier.get_R_alpha(X_valid, y_valid, alpha)
            #this is an aggreate node from my two childrenn nodes merged
            leaf = TLeaf(np.concatenate([self.left.y_all, self.right.y_all]))
            #now I temporarily replace myself with a T
            if am_i_left:
                parent.left = leaf
            else:
                parent.right = leaf
            # parent.left = TLeaf(np.array(['A', 'C', 'B']))

            R_alpha_pruned = root_classifier.get_R_alpha(X_valid, y_valid, alpha)

            # if performance is actually worse, than replace with original
            if R_alpha_pruned > R_alpha_original:
                if am_i_left:
                    parent.left = self
                else:
                    parent.right = self

    def __str__(self):
        return self.to_string([False])

class TLeaf(object):
    def copy(self):
        return TLeaf(self.y_all.copy())

    def __init__(self, y):
        self.leaf = True
        self.y_all = y
        self.y = get_majority_element(y)

    def predict(self, x):
        return self.y

    def to_string(self, structure):
        counter = collections.Counter(self.y_all)
        return (
            ''.join(['   │' if v else '    ' for v in structure[:-1]])
            + ('   ├───>' if structure[-1] else '   └───>')
            + ' ' + str(self.y)
            + ' ' + str(dict(counter))
            + '\n'
            )

    def prune(self, X_valid, y_valid, root_classifier, parent=None, am_i_left = None, alpha=0):
        pass

    def draw(self, fig = None, ax = None, level=0, xfrom=0, xto=10):
        if fig is None:
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(1,1,1) 
        center = (xfrom + xto) / 2
        y = 0.975 - 0.05*level
        rect = plt.Rectangle((center, y), 0.1, 0.025,facecolor="green")
        ax.add_patch(rect)
        ax.text(center, y,"y=" + self.y,fontsize=12)

    def count_leaf_nodes(self):
        return 1

    def get_depth(self):
        return 0


def train(x, y, stop_level=None, rules_used=set()):

    def split(x, y, rule, treshold):
        discriminator = [treshold >= value for value in x[:, rule]]
        not_discriminator = [not b for b in discriminator]
        return (x[discriminator], y[discriminator]), (x[not_discriminator], y[not_discriminator])

    def evaluate_split(x, y, rule, treshold):
        (_, y1), (_, y2) = split(x, y, rule, treshold)
        return IG(y, [y1, y2])

    N, K = x.shape
    rules = set(range(K))

    if len(set(y)) == 1 or (stop_level == 0 and stop_level is not None):
        # TODO: this stop rule could be probably improved upon
        return TLeaf(y)

    rule_treshold_pairs_tmp = [[(rule, v) for v in set(x[:, rule])] for rule in (rules)]
    rule_treshold_pairs = [y for x in rule_treshold_pairs_tmp for y in x]

    result = max(
        ({'rule': rule, 'treshold': treshold, 'IG': evaluate_split(x, y, rule, treshold)} for rule, treshold in rule_treshold_pairs),
        key=lambda tup: tup['IG']
    )
    best_treshold = result['treshold']
    best_rule = result['rule']

    (xl, yl), (xr, yr) = split(x, y, best_rule,  best_treshold)

    if len(yl) == 0 or len(yr) == 0:
        # split does not actually split anything
        return TLeaf(y)

    left  = train(xl, yl, stop_level=None if stop_level is None else stop_level-1, rules_used=rules_used | {best_rule})
    right = train(xr, yr, stop_level=None if stop_level is None else stop_level-1, rules_used=rules_used | {best_rule})
    return TNode(best_rule, best_treshold, left, right)
