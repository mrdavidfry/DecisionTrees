##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the 
# DecisionTreeClassifier 
##############################################################################

import numpy as np
import tree
import eval


class DecisionTreeClassifier(object):
    """
    A decision tree classifier

    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained

    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X

    """

    def __init__(self):
        self.is_trained = False
        self.root = None

    def get_R_alpha(self, X_valid, y_valid, alpha=0):
        miss_rate = 1-eval.accuracy(eval.confusion_matrix(self.predict(X_valid), y_valid))
        weighted_complexity = self.root.count_leaf_nodes() * alpha
        R_alpha = miss_rate + weighted_complexity
        assert alpha >= 0
        return R_alpha

    def prune(self, X_valid, y_valid, alpha=0, in_place=False):
        if not in_place:
            new_DTClassifier = self.copy()
            return new_DTClassifier.prune(X_valid, y_valid, alpha=alpha, in_place=True)
        else:
            self.root.prune(X_valid, y_valid, root_classifier=self, parent=None, am_i_left=None, alpha=alpha)
            return self

    def copy(self):
        new_classifier = DecisionTreeClassifier()
        new_classifier.is_trained = self.is_trained
        new_classifier.root = self.root.copy() if self.root is not None else None
        return new_classifier


    def train(self, x, y, stop_level=None):
        """ Constructs a decision tree classifier from data

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array

        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance

        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################

        # set a flag so that we know that the classifier has been trained
        self.root = tree.train(x, y, stop_level=stop_level)
        self.is_trained = True

        return self

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the
            number of attributes)

        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels
        # feel free to change this if needed
        # predictions = np.zeros((x.shape[0],), dtype=np.object)
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################

        # remember to change this if you rename the variable
        predictions = [self.root.predict(features) for features in x]
        return np.array(predictions, dtype=np.object)

    def __str__(self):
        if self.is_trained:
            return str(self.root)
        else:
            return 'untrained tree'