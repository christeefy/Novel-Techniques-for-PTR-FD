import numpy as np
import pandas as pd

from ...private import utils

def _euclidean_dist(A, B=None):
    '''
    Calculate the Euclidean distance for rows in matrix A and rows in matrix B.
    If B is None, calculate distances for rows between matrix A.

    Arguments:
        A: A matrix (a x P)
        B: A matrix (b x k x P)
    
    Returns:
        A distance matrix (a x b), indicating the distance of all non-i-th point to the i-th point. 
    ''' 
    # Define input matrices with expanded dimensions
    A_expanded = np.expand_dims(A, 2)
    
    # Epsilon term for numerical stability
    epsilon = 1e-9
    
    # Calculate distance of each point and every other point
    if B is None:
        return np.sqrt(np.sum(np.square(A_expanded - np.transpose(A_expanded, (2, 1, 0))), axis=1)) + epsilon
    else:
        return np.sqrt(np.sum(np.square(np.transpose(A_expanded, (0,2,1)) - B), axis=2)) + epsilon


def _kNN(k, data):
    '''
    Return the nearest neighbours to each row in data in the form of a responsibility matrix.
    
    Arguments:
        k:    Number of nearest neighbours (scalar)
        data: Data to perform k-NN, a numpy array (N x P)
    
    Returns:
        A responsibility matrix (N x k), listing the indices of the k-nearest neighbours for each row
    '''

    def _responsibilities(k, distances):
        '''
        Finds the k-nearest neighbours to each point by index.
        
        Arguments:
            k:         Number of nearest neighbours (scalar)
            distances: A distance matrix (N x N)
        
        Returns:
            A responsibility matrix (N x k), listing the indices of the k-nearest neighbours for each row
        '''
        return np.argsort(distances)[:,1:(k + 1)]

    return _responsibilities(k, _euclidean_dist(data))


def _predict_target(data, target, resp):
    '''
    Performa a prediction of the target based on a weighting of contemporaneous neighbours of data.
    
    Arguments:
        data:   Data values (N x P)
        target: Target values to perform prediction (N x P)
        resp:   A responsibility matrix (N x k)
    
    Returns:
        An array of predicted target values (N)
    '''

    def _calculate_weights(data, resp):
        '''
        Calculate weights based on the k-nearest neighbours
        
        Arguments:
            data:             Data values (N x P)
            responsibilities: A responsibility matrix (N x k)
        
        Returns:
            A matrix of weights (N x k)
        '''
        # Obtain shape of responsibilities
        N, k = resp.shape

        # Calculate values for numerator
        for i in range(k):
            num = np.exp( - np.divide(_euclidean_dist(data, data[resp]), \
                                      _euclidean_dist(data, data[resp])[:,0][:, np.newaxis]))

        # Calculate denominator
        denom = np.sum(num, axis=1, keepdims=True)

        # Calculate and return weights
        return np.divide(num, denom)
    
    weights = _calculate_weights(data, resp)
    return np.sum(target[resp] * np.expand_dims(weights, axis=2), axis=1)


def _causality_index(target, prediction):
    '''
    Return a scalar causality index between `target` and `prediction`.

    Pearson correlation coefficient is used as a proxy for causality index.
    '''
    return np.corrcoef(target, prediction)[0,-1]


def ccm_one_way(DF, source, target, k=3):
    '''
    Calculates causality for `target`→`source`. 
    This helper function performs one-half of the 
    convergent cross-mapping (CCM) algorithm 
    described in paper.
    
    Arguments:
        DF:        A Pandas DataFrame
        source:    A string corresponding to data column in DF to perform k-NN
        target:    A string corresponding to target column in DF to perform prediction
        k:         Number of nearest neighbours (scalar)
    
    Returns:
        Calculated causality (float)
    '''
    
    # Define `source` and `target`
    _source = DF[source].values
    _target = DF[target].values

    # Find indices of k-nearest neighbours
    responsibilities = _kNN(k, _source)

    # Calculate predicted target values
    predictions = _predict_target(_source, _target, responsibilities)

    # Return the causality index
    return _causality_index(_target[:, -1], predictions[:, -1])