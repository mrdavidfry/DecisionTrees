##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks:
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np


class Evaluator(object):
    """ Class to perform evaluation
    """

    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.

        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.

        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        #######################################################################
        #                 ** TASK 3.1: COMPLETE THIS METHOD **
        #######################################################################

        if not class_labels:
            class_labels = np.unique(annotation)
        imap = {class_labels[i]: i for i in range(len(class_labels))}

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        for i in range(len(prediction)):
            pred = prediction[i]
            truth = annotation[i]
            confusion[imap[truth], imap[pred]] += 1

        return confusion

    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """
        #######################################################################
        #                 ** TASK 3.2: COMPLETE THIS METHOD **
        #######################################################################

        return confusion.trace() / confusion.sum()

    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.

        Also returns the macro-averaged precision across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.
        """

        #######################################################################
        #                 ** TASK 3.3: COMPLETE THIS METHOD **
        #######################################################################

        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion), ))

        for i in range(len(confusion)):
            column = confusion[:, i]
            s = column.sum()
            p[i] = column[i] / s if s > 0 else 0

        macro_p = np.mean(p)

        return (p, macro_p)


    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.

        Also returns the macro-averaged recall across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged recall score across C classes.
        """

        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion), ))

        #######################################################################
        #                 ** TASK 3.4: COMPLETE THIS METHOD **
        #######################################################################

        for i in range(len(confusion)):
            row = confusion[i, :]
            r[i] = row[i] / np.sum(row)

        # You will also need to change this
        macro_r = np.mean(r)

        return (r, macro_r)

    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.

        Also returns the macro-averaged f1-score across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged f1 score across C classes.
        """

        precisions = self.precision(confusion)[0]
        recalls = self.recall(confusion)[0]
        f = np.array([2 * p * r / (p + r) for p, r in zip(precisions, recalls)])

        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################

        # You will also need to change this
        macro_f = np.mean(f)

        return (f, macro_f)

# for convenience we define the following helpers


def confusion_matrix(prediction, annotation, class_labels=None):
    e = Evaluator()
    return e.confusion_matrix(prediction, annotation, class_labels=None)


def accuracy(confusion):
    e = Evaluator()
    return e.accuracy(confusion)


def precision(confusion):
    e = Evaluator()
    return e.precision(confusion)


def recall(confusion):
    e = Evaluator()
    return e.recall(confusion)


def f1_score(confusion):
    e = Evaluator()
    return e.f1_score(confusion)
