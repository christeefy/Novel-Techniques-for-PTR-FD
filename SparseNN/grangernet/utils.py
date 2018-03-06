import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

from pathlib import Path

def causal_heatmap(W, var_names, mode, ord=2, dst=None, file_header=None, ext='png'):
    '''
    Visualise calculated weights as 
    a heatmap.
    
    When mode is 'joint', it outputs a 
    (p_effect x p_cause) heatmap. When it 
    mode is 'ind', it produces a (p_cause x K) 
    heatmap for each individual p_effect.
    
    Inputs:
        W:            A np array of size (p_effect x p_cause x K)
        df:           A pd dataframe containing column labels
        mode:         Visualisation mode {'joint', 'ind'}
        ord :         The order of the norm
        dst:          Destination to save file
        file_header:  String to be appended to each filename
        ext:          Format to save file
        
    Returns:
        A plt heatmap.
    '''
    assert mode in ['joint', 'ind']
    
    # Infer dimensions
    p, K = W[0].shape

    # Calculate norm on axis 2
    _W_norm = np.linalg.norm(W, ord=ord, axis=2)

    # Infer autocorrelation setting (Boolean) of analysis
    autocorrelation_setting = not np.all(np.diag(_W_norm) == 0)
    print('Autocorrelation during analysis: {}'.format(autocorrelation_setting))
    
    if mode == 'joint':        
        # Visualise causality
        plt.figure(figsize=(p, p))
        plt.imshow(_W_norm, cmap='Greys')
        plt.xlabel('Cause\n', fontsize=16)
        plt.xticks(range(p), var_names)
        plt.ylabel('Response', fontsize=16)
        plt.yticks(range(p), var_names)
        ax = plt.gca()
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        
        # Set white to 0
        plt.clim(vmin=0)
        
        # Output image to file if dst is not None
        if dst is not None:
            assert dst[-1] == '/'

        	# Create path if itdoes not exist
            Path(dst).mkdir(exist_ok=True, parents=True)

            plt.savefig(dst + file_header + '_overall.' + ext, bbox_inches='tight')
        
    elif mode == 'ind':
        for (var, row) in zip(var_names, W):
            plt.figure(figsize=(K, p))
            plt.imshow(row.T, cmap='Greys')
            plt.xlabel('Causes to {}\n'.format(var), fontsize=16)
            plt.xticks(range(p), var_names)
            plt.ylabel('Time Lag', fontsize=16)
            plt.yticks(range(K), range(1, K + 1))
            ax = plt.gca()
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()

            # Set white to 0
            plt.clim(vmin=0, vmax=np.max(_W_norm))

            # Output image to file if dst is not None
            if dst is not None:
                assert dst[-1] == '/'

                # Create path if itdoes not exist
                Path(dst).mkdir(exist_ok=True, parents=True)

                plt.savefig('{}{}_{}.{}'.format(dst, file_header, var, ext), bbox_inches='tight')

            print()
            plt.pause(0.001)
    

def causal_graph(W, var_names, threshold=0.1, dst=None, filename='graph'):
    '''
    Construct a causal graph using the graphviz module.

    Inputs:
        W:         FCNN layer 1 weights as a np array (p x p x K)
        var_names: List of variable names
        threshold: Minimum pct of max value of W to consider a positive causal connection
        dst:       File save location
        filename:  Filename
    '''
    # Calculate L2-norm of W
    _W_norm = np.linalg.norm(W, axis=-1)

    # Create causal directed graph
    dot = Digraph()

    # Remove margin and let nodes flow from left to right
    dot.graph_attr['margin'] = '0'
    dot.graph_attr['rankdir'] = 'LR'
    dot.graph_attr['layout'] = 'circo'

    # Create nodes
    for var in var_names:
        dot.node(var, var)

    # Create a function to zip np.where results
    zipper = lambda x: zip(x[0], x[1])

    # Create edges
    for (effect, cause) in zipper(np.where(_W_norm >= threshold * np.max(_W_norm))):
        # Obtain relative weight of element
        _weight = _W_norm[effect, cause] / np.max(_W_norm)
        dot.edge(var_names[cause], var_names[effect], penwidth=str(5 * _weight), arrowsize='1')

    # Save file (optional)
    if dst is not None:
        dot.render(filename, dst, view=False, cleanup=True)

    return dot


def save_results(ex_id, W, hparams, W_submod=None):
    '''
    Save computation results to Results/ex_id using np.savez. 
    W and hparams will be saved as 'W' and 'hparams'
    respectively. 
    '''
    if W_submod is None:
        np.savez('Results/{}.npz'.format(ex_id), W=W, hparams=hparams)
    else:
        np.savez('Results/{}.npz'.format(ex_id), W=W, W_submod=W_submod, hparams=hparams)
    

def load_results(ex_id):
    '''
    Load results from Results/ex_id.
    Returns subsequent W and hparams.
    '''
    file = np.load('Results/{}.npz'.format(ex_id))
    
    W = file['W']
    hparams = file['hparams'].item()
    
    if 'W_submod' in file.keys():
        W_submod = file['W_submod']
        return W, W_submod, hparams
    
    return W, hparams