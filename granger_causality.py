import numpy as np
from scipy.stats import f

def _create_dataset_vector_output(df, max_lag):
    '''
    Create X and Y for a `var`, based on a `df`. 
    Lagged X values of up to `max_lag` are selected.

    If autocorrelate is False, past values of `var`
    will not be present in X. 

    Arguments:
        df: A pandas dataframe of shape (N x p)
    
    Returns:
        X: Input data of shape (N, p * max_lag)
        Y: Output labels of shape (N, p).
    '''
    # Reverse df rows. This makes going down 
    # the rows to be going back in time. 
    df = df[::-1]
    
    # Obtain output label
    Y = np.array(df.iloc[:-(max_lag + 1), :])

    # Create input data
    X = np.vstack([df[(i + 1):(max_lag + i + 1)].values.T.reshape(-1) for i in range(len(df) - 1 - max_lag)])
    
    return X, Y

def _calc_RSS(X, Y):
    '''
    Calculate the residual sum of squares (RSS) of the linear model 
    predictions of Y using inputs X.

    Linear coefficients of X are calculated using ordinary least squares minimization. 

    Arguments:
        X: NumPy array of shape (N x p)
        Y: NumPy array of shape (N x m)
    '''
    return np.linalg.lstsq(X, Y)[1]

def granger_causality(df, max_lag, pval=0.05):
    '''
    Calculates the causality between time series pairs in `df` using the 
    Granger Causality technique. 

    Arguments:
        df: A pandas dataframe of shape (N x p)
        max_lag: Number of past lagged inputs to include in the linear model (int)
        pval: Type 1 error for rejecting null hypothesis of no causality. 
    '''
    # Create numpy input and output arrays for VAR modelling from dataframe
    X, Y = _create_dataset_vector_output(df, K)

    assert len(X) == len(Y)

    # Infer data dimensions
    N, p = Y.shape
    
    # Calculate RSS for full model
    RSS_full = np.expand_dims(calc_RSS(X, Y), axis=1)
    
    # Calculate RSS for reduced models
    RSS = np.zeros((p, p))
    for i in range(p):
        RSS[:, i] = _calc_RSS(np.delete(X, slice(i * K, (i + 1) * K), axis=1), Y)
        
    # Calculate f-stats
    f_stats = ((RSS - RSS_full) / K) / (RSS_full / (N - p * K))
    
    # Binarize values based on significance
    causalities = f.sf(f_stats, K, N - p * K) <= pval
    
    return causalities1