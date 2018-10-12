import plotly.offline as pyo
import plotly.graph_objs as go
from plotly import tools

# Initialise Jupyter notebook for offline plotting
pyo.init_notebook_mode()

def visualise_predictions(targetDict, predictionsDict):
    '''
    Create a scatterplot visualising predictions vs. target.
    
    Arguments:
        target:      Target values (N x P)
        predictions: Prediction values (N x P)
    '''
    figure = tools.make_subplots(1, 2, print_grid=False, 
                                        subplot_titles=['{0} → {1} (rho = {2})<br>{3:.3f}'\
                                                        .format(targetCol, 
                                                                dataCol, 
                                                                np.round(np.corrcoef(targetDict[targetCol][:,-1], 
                                                                                     predictionsDict[targetCol][:,-1])[0,-1], 3),
                                                                L2_score(targetDict[targetCol][:, -1], predictionsDict[targetCol][:, -1])),
                                                        '{0} → {1} (rho = {2})<br>{3:.3f}'\
                                                        .format(dataCol, 
                                                                targetCol, 
                                                                np.round(np.corrcoef(targetDict[dataCol][:,-1], 
                                                                                     predictionsDict[dataCol][:,-1])[0,-1], 3),
                                                                L2_score(targetDict[dataCol][:, -1], predictionsDict[dataCol][:, -1])),
                                                       ])
    
    for n in range(2):
        varCols = [dataCol, targetCol]
    
        unReconstrVar = varCols[n]
        reconstrVar = varCols[1 - n]
        
        target = targetDict[reconstrVar]
        predictions = predictionsDict[reconstrVar]
        
        trace = go.Scattergl(
            x = target[:,-1],
            y = predictions[:,-1],
            mode = 'markers',
            hoverinfo = 'text',
            text = [str(i) for i in range(len(target))]
        )

        combined_data = np.append(target, predictions)

        line_trace = go.Scattergl(
            x = [np.min(combined_data), np.max(combined_data)],
            y = [np.min(combined_data), np.max(combined_data)],
            mode = 'lines',
            hoverinfo = 'none',
            line = {
                'color': '#000000',
                'dash': 'dash',
                'width': 2
            }
        )
        
        figure.append_trace(trace, 1, n + 1)
        figure.append_trace(line_trace, 1, n + 1)

    figure['layout'].update(go.Layout(
        showlegend = False,
        height = 400,
        width = 1000,
        xaxis = {'title': varCols[1], 'domain': [0., 0.4]},
        xaxis2 = {'title': varCols[0], 'domain': [0.6, 1.]},
        yaxis = {'title': '{0} | M({1})'.format(varCols[1], varCols[0])},
        yaxis2 = {'title': '{0} | M({1})'.format(varCols[0], varCols[1])},
    ))
    
    pyo.iplot(figure)