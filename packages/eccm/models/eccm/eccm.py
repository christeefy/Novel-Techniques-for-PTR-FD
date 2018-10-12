import pandas as pd
import itertools

from . import utils
from ... import graph
from ..ccm.utils import ccm_one_way
from ...private.utils import generate_delayed_df

def _eccm_base(df, source, target, cross_map_lags, embed_dim, delay=1):
    '''
    Performs CCM causality calculations for `target`→`source` iteratively for
    (-cross_map_lags, cross_map_lags).

    Returns:
        A list of causalities for all cross map lags.
    '''

    # Generate empty list to store lagged causality indices
    causality_list = []
    
    # Iterate through [-`cross_map_lags`, `cross_map_lags`]
    for lag in reversed(range(-cross_map_lags, cross_map_lags + 1)):
        # Generate delayed embedding df with appropriate cross map lag
        _df = generate_delayed_df(df, source, target, lag, embed_dim, delay)
        
        # Calculate causality of `target` → `source`
        _causality = ccm_one_way(_df, source, target)
        
        # Append value to list
        causality_list.append(_causality)
    
    return causality_list


def eccm(DF, cross_map_lags=5, use_all_points=False, criterion='Peak', p_val=0.05, verbose=False):
    '''
    Perform causality calculations using the Extended CCM algorithm.

    There are five stages to this algorithm:

    1. Causality estimation for a given variable pair at various cross map lags

    2. Selecting the best causality for a given variable pair
        To select the best causality for a variable pair at various time lags, two criterions are available. 
            - The peak criterion selects the largest causality with a negative cross map lag. 
            - The URC (Upper Right Corner) is a heuristic to select the causality corresponding 
            to a negative cross map lag at the upper right corner of a plateau, which occurs quite commonly in 
            some of the simulated examples. 

    3. Selecting significant causalities amongst all variable pairs.
        This is done using a statistical t-test on the Pearson correlation coefficients. 

    4. Prune graph for indirect causalities
        A graph pruning is also done to eliminate indirect causalities calculated by ECCM.

    5. Convert pruned graph into a causality matrix.

    Developer's Note:
    This function is effectively handles variable-pair manipulations and defers ECCM calculation
    for a given variable pair to the _eccm_base function.
    
    Arguments:
        DF:             A Pandas DataFrame containing the time series variables
        cross_map_lags: Range of cross map lags to calculate 
                        [-`cross_map_lags`, `cross_map_lags` + 1]
        use_all_points: Boolean. If True, use all points for ECCM calculation. 
                        Otherwise, use the last 25% of datapoints or 1,000 points, 
                        whichever is lower.
        criterion:      The criterion used to compare causalities calculated for different lags.
                        Valid choices include {'Peak', 'URC'}
        p_val:          Maximum Type I error in rejecting null hypothesis of no causality
        verbose:        Boolean to control verbosity during calculations 
    
    Returns:
        A tuple of
            causalities: A (p x p) NumPy matrix of binary causalities.
            g:                     The pruned graph
    '''
    
    assert criterion in ['Peak', 'URC']
    
    # Calculate length of datapoints to use
    N = len(DF) if use_all_points else min(int(0.25 * len(DF)), 1000)
    
    # Ensure that there are sufficient points to perform ECCM
    assert N > cross_map_lags + 3 # Default kNN hyperparameter = 3 in CCM function
    
    # Create empty DataFrame to store results
    causalitiesDF = pd.DataFrame()
    
    # Run Extended CCM algorithm
    p = len(DF.columns)
    for i, (source, target) in enumerate(itertools.permutations(DF.columns, 2)):
        if verbose:
            print(f'Processing {source} → {target} ({i + 1} of {p**2 - p}, {100 * (i + 1) / (p**2 - p):.1f}%)...')

        causalitiesDF['{} → {}'.format(target, source)] = \
            _eccm_base(DF[-N:], 
                   source=source, 
                   target=target,
                   embed_dim=len(DF.columns),
                   cross_map_lags=cross_map_lags)

    # Add 'Cross Map Lag' as DataFrame index
    causalitiesDF['xMap Lag'] = list(range(-cross_map_lags, cross_map_lags + 1))
    causalitiesDF.set_index('xMap Lag', inplace=True)
    
    # Calculate peaks
    peaksDF = utils._peak_causality_coordinates(causalitiesDF)
    
    # Filter peaks
    peaksDF = (
        peaksDF
        .where(peaksDF[f'{criterion} Sig.'] <= p_val)
        .where(peaksDF[f'{criterion} xMap Lag'] < 0)
        .dropna()
    )
    
    # Prune graph for indirect causalities
    g = graph.Graph(nodes=DF.columns, 
                         edges=peaksDF.index, 
                         dists=[-int(dist) for dist in peaksDF[f'{criterion} xMap Lag']])
    if verbose:
        print('Before pruning')
        display(g)
        
    g.prune(verbose=verbose)
    
    if verbose:
        print('After pruning')
        display(g)
    
    # Obtain causality matrix
    causalities, _ = g.adj_mat()
    
    return causalities, g