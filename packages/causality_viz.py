import numpy as np
from graphviz import Digraph

import matplotlib.pyplot as plt

def causal_heatmap(W, var_names, mode='regular', ord=2, threshold=0.1, dst=None, file_header=None, ext='png'):
    '''
    Generates a heatmap visualization of a (p x p) heatmap based on causalities `W` of shape (p_effect x p_cause x K).

    ***Heatmap Modes***
    When mode is 'regular' (default), it displays W of shape (p x p) as a (p x p) heatmap. 
    
    When mode is 'joint', it converts W of shape (p x p x K) to a heatmap of shape (p x p)
    by norming the K-dimension.

    When mode is 'joint_threshold', its transformation is similar to 'joint' except that
    the (p x p) values are thresholded to binary values based on the supplied `threshold`.

    When mode is 'ind', it produces a
    (p_cause x K) heatmap for each individual p_effect (applicable only to Granger Net outputs).

    
    Arguments:
        W:            A np array of size (p_effect x p_cause x K)
        var_names:    List of variable names
        mode:         Visualisation mode. Valid choices are {'joint', 'ind', 'joint_threshold', 'regular'}
        ord :         The order of the norm
        threshold:    Threshold to binarize causality for 'joint_threshold' mode
        dst:          Destination to save file
        file_header:  String to be appended to each filename (only applicable if `dst` is specificed)
        ext:          Format to save file (only applicable if `dst` is specificed)
        
    Returns:
        A matplotlib heatmap.
    '''
    assert mode in ['joint', 'ind', 'joint_threshold', 'regular']

    # Enable LaTeX fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Convert var_names to be in LaTeX math-mode
    var_names = [f'${var}$' for var in var_names]


    if mode == 'regular':
        p = len(W)

        # Visualise causality
        plt.figure(figsize=(p, p))
        plt.imshow(W, cmap='Greys')
        plt.xlabel('Cause', fontsize=16)
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

            plt.savefig(dst + file_header + '.' + ext, bbox_inches='tight', dpi=200)

        plt.show()

        return
    
    # Infer dimensions
    p, K = W[0].shape

    # Calculate norm on axis 2
    _W_norm = np.linalg.norm(W, ord=ord, axis=-1)

    # Infer autocorrelation setting (Boolean) of analysis
    autocorrelation_setting = _has_autocorrelation(W)
    print('Autocorrelation during analysis: {}'.format(autocorrelation_setting))
    
    if mode == 'joint':        
        # Visualise causality
        plt.figure(figsize=(p, p))
        plt.imshow(_W_norm, cmap='Greys')
        plt.xlabel('Cause', fontsize=16)
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

            plt.savefig(dst + file_header + '.' + ext, bbox_inches='tight', dpi=200)

        # Show image
        plt.show()

    elif mode == 'joint_threshold':
        # Get tuple containing index of values above threshold
        idx_above = np.where(_W_norm >= threshold * np.max(_W_norm))
        idx_below = np.where(_W_norm < threshold * np.max(_W_norm))
        
        # Thresold appropriate values to 0 or 1
        _W_norm[idx_above] = 1
        _W_norm[idx_below] = 0

        # Visualise causality
        plt.figure(figsize=(p, p))
        plt.imshow(_W_norm, cmap='Greys')
        plt.xlabel('Cause', fontsize=16)
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

            plt.savefig(dst + file_header + '.' + ext, bbox_inches='tight', dpi=200)

        # Show image
        plt.show()
        
    elif mode == 'ind':
        for (var, row) in zip(var_names, W):
            plt.figure(figsize=(K, p))
            plt.imshow(row.T, cmap='Greys')
            plt.xlabel('Causes to {}'.format(var.split("$")[1]), fontsize=16)
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

                plt.savefig('{}{}_{}.{}'.format(dst, file_header, var, ext), bbox_inches='tight', dp=200)

            print()
            plt.show()

    

def causal_graph(W, var_names, norm_W=False, threshold=0.1, eastman=False, use_circo_layout=None, dst=None, filename='graph'):
    '''
    Construct a causal graph using the graphviz module.

    Arguments:
        W:                FCNN layer 1 weights as a np array (p x p x K)
        var_names:        List of variable names
        norm_W:           Boolean on whether `W` needs to be normed. 
        threshold:        Minimum pct of max value of W to consider a positive causal connection
        use_circo_layout: Boolean on whether to use circo layout. Default is None, 
                          which infers based on whether `W` contains autocorrelation.
        dst:              File save location
        filename:         Filename
    '''
    # Calculate L2-norm of W
    if norm_W:
        W = np.linalg.norm(W, axis=-1)

    # Create causal directed graph
    dot = Digraph()

    # Remove margin and let nodes flow from left to right
    dot.graph_attr['margin'] = '0'
    dot.graph_attr['rankdir'] = 'LR'
    if not _has_autocorrelation(W) or use_circo_layout:
        dot.graph_attr['layout'] = 'circo'

    # Create nodes
    for var in var_names:
        if eastman:
            dot.node(var, var, 
                     shape='doubleellipse' if 'OP' in var else 'ellipse',
                     fillcolor='#00A89D' if var.split('.')[0] in ['LC1', 'TC1', 'TC2'] else '#56C1FF',#'#062958',
                     style='filled',
                     penwidth='0',
                )
        else:
            dot.node(var, var)

    # Create a function to zip np.where results
    zipper = lambda x: zip(x[0], x[1])

    # Create edges
    for (effect, cause) in zipper(np.where(W >= threshold * np.max(W))):
        # Obtain relative weight of element
        _weight = W[effect, cause] / np.max(W)
        dot.edge(var_names[cause], var_names[effect], penwidth=str(1 * _weight), arrowsize='1')

    # Save file (optional)
    if dst is not None:
        dot.render(filename, dst, view=False, cleanup=True)

    return dot


def _has_autocorrelation(W):
    '''
    Checks whether causality matrix W of shape (p x p x K) was calculated with 
    or without the autocorrelation setting during training (applicable to Granger Net only). 

    Returns a Boolean.
    '''

    # Compute the L2 norm
    W_norm = np.linalg.norm(W, axis=-1)

    return not np.all(np.diag(W_norm) == 0)
