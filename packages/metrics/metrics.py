import numpy as np

from . import utils

def fault_detected(W, pos, exceptions=[]):
    ''' (np.ndarray, int) -> (Bool)
    Post-processing to check whether 
    fault (variable at position `pos`) 
    is correctly identified.
    
    Arguments:
        W: Binary causality matrix
        pos: Fault index

    Returns:
        A Boolean.
    '''
    return (not np.any(np.delete(W[pos], [pos] + exceptions))) and np.any(W[:, pos])

def MSE(W, W_truth, threshold=0.1, autocausation=True):
    '''
    Calculate the mean squared error.

    Arguments:
        W:             Calculated causality of shape (p x p) or (p x p x K)
        W_truth:       Grouth truth causality of shape (p x p)
        threshold:     Value above which causality values are labelled as positive. 
                       Applicable only when `W` is a 3-dimensional matrix.
                       (i.e. an output of Granger Net)
        autocausation: Boolean on whether autocausation was included in the analysis.

    Returns:
        The MSE (NumPy float).
    '''
    # Reduce last dimension and binarize if W is from GN
    if len(W.shape) == len(W_truth.shape) + 1:
        W = np.linalg.norm(W, axis=-1)

        # Binarize values
        W = W >= threshold * np.max(W)

    if not autocausation:
        W = utils.ravel_without_diag(W)
        W_truth = utils.ravel_without_diag(W_truth)

    return np.mean((W - W_truth)**2)


def PRF(W, W_truth, threshold=None, autocausation=True):
    '''
    Calculate metrics by comparing thresholded values of 
    `W` against the ground truth matrix, `W_truth`.

    Value is considered positive if it is at least `threshold` of 
    the max value in the L2 norm of `W`.

    Arguments:
        W:             A NumPy matrix of calculated weights (p x p x K)
        W_truth:       Truth matrix (p x p)
        threshold:     Cutoff point
        autocausation: Boolean on whether autocausation was included in the analysis.

    Returns:
        A dictionary of threshold, precision, recall, F1-score
    '''
    # Constants
    EPSILON = 1e-10 # For numerical stability

    if len(W.shape) == len(W_truth.shape) + 1:
        # Obtain norm of W (p x p)
        W = np.linalg.norm(W, axis=-1)

        # Find the corresponding indices where values are positive
        # and negative 
        pos_idx = np.where(W >= threshold * np.max(W))
        neg_idx = np.where(W < threshold * np.max(W))

        # Set appropriate values to 0 and 1
        W[pos_idx] = 1
        W[neg_idx] = 0

    if not autocausation:
        W = utils.ravel_without_diag(W)
        W_truth = utils.ravel_without_diag(W_truth)

    # Assertion checks
    assert W.size == W_truth.size, \
    'Sizes of the norm of W and ground truth matrices are not in agreement.'

    assert set(np.unique(W_truth)) <= {0., 1.}, \
    'Ground truth matrix should contains only 0 and 1 binary values.'

    # Calculate true positive
    t_pos = np.sum(W * W_truth)

    # Calculate precision, recall and F-score
    prec = t_pos / (np.sum(W) + EPSILON)
    rec = t_pos / (np.sum(W_truth) + EPSILON)
    f_score = 2 / (1 / (prec + EPSILON) + 1 / (rec + EPSILON))

    return dict(threshold=threshold, precision=prec, recall=rec, f_score=f_score)


def AUCPR(W, W_truth, normalized=True, autocausation=True):
    '''
    Calculate the area under the precision-recall curve.
    
    If normalized is True, calculates the normalized value,
    which accounts for the subdomain and "unachieveable region". 

    This value should always be True, since it is impossible to 
    obtain a recall of 0 based on the definition applied in 
    thresholding the weights. 
    '''
    # Constants
    _ROUND = 4      # Number of decimal places to output
    EPSILON = 1e-6  # For numerical stability

    # Obtain a list of (prec, recall, F1) tuples
    metrics = utils.metrics_list(W, W_truth, autocausation=autocausation)

    # Calculate PR AUC
    AUC = 0.

    for i in range(len(metrics) - 1):
        AUC += 0.5 * (metrics[i + 1]['precision'] + metrics[i]['precision']) \
                * (metrics[i + 1]['recall'] - metrics[i]['recall'])

    # Return calculated AUC (absolute area) if normalized is False
    if not normalized:
        return round(AUC, _ROUND)

    # Calculate the maximum possible AUC: 1 x (b - a)
    # where a and b are the lower and upper domain bounds respectively
    a, b = utils.bounds(metrics)
    AUC_max = b - a

    # Calculate area occupied by "unachievable region"
    skew = np.sum(W_truth) / W_truth.size # Finds the skew of the data
    AUC_min = (b - a) - (1 - skew) / (skew + EPSILON) \
            * np.log((skew * b + (1 - skew)) / (skew * a + (1 - skew) + EPSILON))

    # Calculate normalised AUCPR
    AUCNPR = (AUC - AUC_min) / (AUC_max - AUC_min + EPSILON)

    return round(AUCNPR, _ROUND)