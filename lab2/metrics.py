import numpy as np
from sklearn import metrics as met


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes = None) -> np.ndarray:
    """"
    Computes the confusion matrix from labels (y_true) and predictions (y_pred).
    The matrix columns represent the prediction labels and the rows represent the ground truth labels.
    The confusion matrix is always a 2-D array of shape `[num_classes, num_classes]`,
    where `num_classes` is the number of valid labels for a given classification task.
    The arguments y_true and y_pred must have the same shapes in order for this function to work

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    # TODO your code here - compute the confusion matrix
    # even here try to use vectorization, so NO for loops

    # 0. if the number of classes is not provided, compute it based on the y_true and y_pred arrays
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

    # 1. create a confusion matrix of shape (num_classes, num_classes) and initialize it to 0
    conf_mat = np.zeros((num_classes, num_classes))

    # 2. use argmax to get the maximal prediction for each sample
    # hint: you might find np.add.at useful: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
    indices = np.stack([y_true, y_pred])
    np.add.at(conf_mat, indices.tolist(), 1)

    # end TODO your code here
    return conf_mat


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None) -> float:
    """"
    Computes the precision score.
    For binary classification, the precision score is defined as the ratio tp / (tp + fp)
    where tp is the number of true positives and fp the number of false positives.

    For multiclass classification, the precision and recall scores are obtained by summing over the rows / columns
    of the confusion matrix.

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    # TODO your code here
    conf_mat = confusion_matrix(y_true, y_pred, num_classes)
    precision = conf_mat.diagonal() / (conf_mat.diagonal() + np.sum(conf_mat, axis=0))
    # end TODO your code here
    return np.average(precision)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None)  -> float:
    """"
    Computes the recall score.
    For binary classification, the recall score is defined as the ratio tp / (tp + fn)
    where tp is the number of true positives and fn the number of false negatives

    For multiclass classification, the precision and recall scores are obtained by summing over the rows / columns
    of the confusion matrix.

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    # TODO your code here
    conf_mat = confusion_matrix(y_true, y_pred, num_classes)
    recall = conf_mat.diagonal() / (conf_mat.diagonal() + np.sum(conf_mat, axis=1))
    # end TODO your code here
    return np.average(recall)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # TODO your code here
    # remember, use vectorization, so no for loops
    # hint: you might find np.trace useful here https://numpy.org/doc/stable/reference/generated/numpy.trace.html
    conf_mat = confusion_matrix(y_true, y_pred)
    acc_score = np.trace(conf_mat) / np.sum(conf_mat)
    # end TODO your code here
    return acc_score


if __name__ == '__main__':
    y_true = [1, 2, 3, 1, 2, 0, 4]
    y_pred = [4, 3, 0, 1, 2, 0, 1]
    print(confusion_matrix(y_true, y_pred))
    print(precision_score(y_true, y_pred))
    print(met.precision_score(y_true, y_pred, average='macro'))
    print(recall_score(y_true, y_pred))
    print(met.recall_score(y_true, y_pred, average='macro'))
    print(accuracy_score(y_true, y_pred))
    print(met.accuracy_score(y_true, y_pred))
    # TODO your tests here
    # add some test for your code.
    # you could use the sklean.metrics module (with macro averaging to check your results)
