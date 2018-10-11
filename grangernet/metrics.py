import numpy as np

import plotly.offline as pyo
import plotly.graph_objs as go

import matplotlib.pyplot as plt

# Activate offline notebook plotting (for Jupyter Notebook)
pyo.init_notebook_mode()

def _ravel_without_diag(W):
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
        W = _ravel_without_diag(W)
        W_truth = _ravel_without_diag(W_truth)

    return np.mean((W - W_truth)**2)

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


def _metrics_list(W, W_truth, autocausation=True):
    '''
    Helper function to iterate over PRF
    function and return a sorted list of PRF tuples.
    '''
    # Calculate prec, recall and F-score metric for a
    # range of thresholds. Metrics is a list of 
    # (order, (prec, recall F-score)) tuple
    metrics = [PRF(W, W_truth, 
                   threshold=threshold, 
                   autocausation=autocausation) for threshold in _thresholds(W)]

    # Obtain a unique list of metrics, sorted by recall
    return sorted(metrics, key=lambda x: (x['recall'], -x['precision']))


def _bounds(metrics):
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
        W = _ravel_without_diag(W)
        W_truth = _ravel_without_diag(W_truth)

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
    metrics = _metrics_list(W, W_truth, autocausation=autocausation)

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
    a, b = _bounds(metrics)
    AUC_max = b - a

    # Calculate area occupied by "unachievable region"
    skew = np.sum(W_truth) / W_truth.size # Finds the skew of the data
    AUC_min = (b - a) - (1 - skew) / (skew + EPSILON) \
            * np.log((skew * b + (1 - skew)) / (skew * a + (1 - skew) + EPSILON))

    # Calculate normalised AUCPR
    AUCNPR = (AUC - AUC_min) / (AUC_max - AUC_min + EPSILON)

    return round(AUCNPR, _ROUND)


def PR_curve(W, W_truth, display_AUC=True, image=None, run_id=''):
    '''
    Display a Plotly precision-recall curve of the 
    data along with the "unachievable region".

    Inputs:
        image: Format of image to exported. Image will 
               not be exported when set to default None value. 
    '''
    # # Assertion checks
    # if image is not None:
    #     assert 'dst' in kwargs.keys(), 'Export destination not specified.'

    # Constants
    INTERVAL = 0.01  # Interval between each minimum PR points
    DP = 4 # Number of decimal places to round to
    M_TOP = 75 # Top margin
    M_BOT = 50 # Bottom margin
    M_L = 80 # Left margin
    M_R = 80 # Right margin


    # Convert W_truth into a NumPy array if not already so
    W_truth = np.array(W_truth)

    # Obtain sorted PRF metrics
    metrics = _metrics_list(W, W_truth)

    # Obtain minimum and maximum recalls (a & b respectively)
    a, b = _bounds(metrics)

    # Calculate skew of dataset
    skew = np.sum(W_truth) / W_truth.size

    # Create traces for PR curve and unachievable region
    curve = go.Scatter(
        x = [round(metric['recall'], DP) for metric in metrics],
        y = [round(metric['precision'], DP) for metric in metrics],
        text = ['threshold: {}'.format(round(metric['threshold'], 3)) for metric in metrics],
        mode = 'lines+markers',
        marker = dict(color = '#1f77b4'),
        name = 'PR Curve'
    )

    r = np.arange(a, b + INTERVAL, INTERVAL)

    min_curve = go.Scatter(
        x = np.round(r, DP),
        y = np.round(skew * r / (1 - skew + skew * r), DP),
        name = 'Unachievable Region',
        fill = 'tozeroy',
        fillcolor = 'rgb(225, 225, 225)',
        mode = 'none',
        hoverinfo = 'none',
    )

    data = [min_curve, curve]

    # Define title
    title = ''
    if image is None:
        title += '<b>Precision-Recall Curve</b>'
    if display_AUC:
        title += f'<br>(AUCNPR: {round(AUCPR(W, W_truth), 3)})'

    # Define layout
    layout = go.Layout(
        title = title,
        titlefont = dict(size=20 if image else 20),
        xaxis = dict(
            title = 'Recall',
            titlefont = dict(size=22 if image else 16),
            tickfont = dict(size=16 if image else 12),
            showgrid = False,
            dtick = 1 - a,
            tick0 = a,
            range = [a - INTERVAL, b + INTERVAL]
        ),
        yaxis = dict(
            title = 'Precision',
            titlefont = dict(size=22 if image else 16),
            tickfont = dict(size=16 if image else 12),
            range = [0, 1 + INTERVAL]
        ),
        legend = dict(
            orientation = 'h',
            x = 0.28,
            y = -0.23,
            traceorder = 'reversed'
        ),
        margin = dict(
            t = M_TOP,# if image is None else 20, 
            b = M_BOT,
        ),
        font = dict(
            family = 'Palatino',
        ),
        hovermode = 'closest',
        height = 300 + M_TOP + M_BOT,
        width = 300 * (b - a) + M_L + M_R,
        showlegend = False,
        annotations = [
            dict(
                x = 0.5 * (b + a),
                y = 0.4 * skew * (b - a) / (1 - skew + skew * (b - a)),
                text = '<b>Unachievable<br>Region</b>',
                showarrow = False,
                font = dict(size=18 if image else 14)
            ),
        ]
    )

    # Plot figure
    fig = go.Figure(data=data, layout=layout)
    return pyo.iplot(fig, 
        image = image, 
        filename = 'PR_Curve_{}'.format(run_id), 
        image_height = 300 + M_TOP + M_BOT,
        image_width = 300 * (b - a) + M_L + M_R)

