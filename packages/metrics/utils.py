import numpy as np
from . import metrics

def _thresholds(W):
    '''
    Returns a NumPy list of L2-norm values of W, expressed
    as a percent of the largest norm, in increasing order.

    Note: For improved interpretability in PR_Curve, this function
    returns the minimum threshold required for a certain L2-norm 
    to be included in thresholding. 
    '''
    # Constant
    EPSILON = 1e-6 # For numerical stability

    # Calculate norm
    W_norm = np.linalg.norm(W, axis=-1)

    # Calculate and sort thresholds
    thresholds = np.append(0, np.sort(W_norm / np.max(W_norm) + EPSILON, axis=None)[:-1])

    return thresholds


def ravel_without_diag(W):
    '''
    Return a ravelled np.array without the diagonal elements.

    Arguments:
        W: A causality matrix of shape (p x p).

    Returns:
        A 1-D NumPy array of length p^2 - p.
    '''
    # Obtain length of W
    p = len(W)
    return np.delete(W.ravel(), [(p + 1) * i for i in range(p)])


def metrics_list(W, W_truth, autocausation=True):
    '''
    Helper function to iterate over PRF
    function and return a sorted list of PRF tuples.
    '''
    # Calculate prec, recall and F-score metric for a
    # range of thresholds. Metrics is a list of 
    # (order, (prec, recall F-score)) tuple
    metrics = [metrics.PRF(W, W_truth, 
                   threshold=threshold, 
                   autocausation=autocausation) for threshold in _thresholds(W)]

    # Obtain a unique list of metrics, sorted by recall
    return sorted(metrics, key=lambda x: (x['recall'], -x['precision']))


def bounds(metrics):
    '''
    Return the lower and upper bounds of the recall domain.

    Arguments:
        metrics: A list of metrics dictionary

    Returns:
        a: Lower recall bound
        b: Upper recall bound
    '''
    recalls = list(map(lambda x: x['recall'], metrics))
    b = max(recalls)
    a = min(recalls)

    return a, b