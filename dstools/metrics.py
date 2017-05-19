import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import beta, binom_test
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.fixes import partition
from sklearn.utils.multiclass import type_of_target


def precision_at_k_score(y_true, y_score, sample_weight=None, n_tops=1):
    """Compute the precision for n_tops scored labels
    Count how many time the ``n_tops`` label with highest score are in the set
    of true labels. One minus precision at 1 is also called one error.
    It's averaged over the samples and can take a value between 0 and 1.
    The best performance achievable could be below 1.
    Parameters
    ----------
    y_true : array, shape = [n_samples, n_labels]
        True binary labels in binary indicator format.
    y_score : array, shape = [n_samples, n_labels]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    n_tops : int, optional
        The number of labels on which the precision is computed.
    Returns
    -------
    score : float
    """
    y_true = check_array(y_true, ensure_2d=False, accept_sparse='csr')
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score)

    y_true = csr_matrix(y_true)

    y_type = type_of_target(y_true)
    if y_type not in ("multilabel-indicator",):
        raise ValueError("{0} format is not supported".format(y_type))

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different shape")

    if n_tops < 1:
        raise ValueError("Number of top labels must >= 1, got %s"
                         % n_tops)

    n_samples, n_labels = y_true.shape

    y_true = csr_matrix(y_true)

    if n_tops > 1:
        row_top_k = partition(y_score, kth=n_labels - n_tops,
                              axis=1)[:, -n_tops]
    else:
        row_top_k = np.max(y_score, axis=1)

    # Here we take into account the ties
    y_thresholded = csr_matrix(y_score >= row_top_k.reshape((-1, 1)))

    n_ties = y_thresholded.sum(axis=1)
    score = (y_true.multiply(y_thresholded).sum(axis=1) / n_ties).mean()

    return score


def print_confusion_matrix(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def binom_interval(success, total, confint=0.95):
    '''from paulgb's binom_interval.py'''
    quantile = (1 - confint) / 2.
    lower = beta.ppf(quantile, success, total - success + 1)
    upper = beta.ppf(1 - quantile, success + 1, total - success)
    return lower, upper


def accuracy_confidence_interval(cm):
    return binom_interval(sum(np.diag(cm)), sum(cm))


def accuracy_p_value(cm):
    return binom_test(sum(np.diag(cm)), n=sum(cm))
