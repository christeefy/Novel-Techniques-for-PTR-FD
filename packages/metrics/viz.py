import numpy as np

import plotly.offline as pyo
import plotly.graph_objs as go

from ..metrics import utils
from .metrics import AUCPR


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
    metrics = utils.metrics_list(W, W_truth)

    # Obtain minimum and maximum recalls (a & b respectively)
    a, b = utils.bounds(metrics)

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