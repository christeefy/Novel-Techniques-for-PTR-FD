import numpy as np
import tensorflow as tf
import pandas as pd

def create_dataset(df, var, max_lag):
    '''
    Create X and Y for a var, based on a df. 
    X values will be selected up to `max_lag`. 
    
    Returns:
        X: Input data of shape (m, p * max_lag)
        Y: Output labels of shape (m,).
    '''
    assert var in df.columns
    
    # Reverse df rows. This makes going down 
    # the rows to be going back in time. 
    df = df[::-1]
    
    # Obtain output label
    Y = np.array(df[var][:-(max_lag + 1)])
    
    # Create input data
    X = np.vstack([df[(i + 1):(max_lag + i + 1)].values.T.reshape(-1) for i in range(len(df) - 1 - max_lag)])
    
    return X, Y
    

def extract_weights(W1, max_lag):
    '''
    Extract weights from W1 of size ((p * K) x n_H) 
    by computing the L2-norm for each weight along 
    the n_H axis.
    
    Returns:
        A np array of size p x K.
    '''
    # Compute the L2-normof W1 of size K x p
    W1_norm = np.linalg.norm(W1, axis=1).reshape(-1, max_lag, order='C')
    
    return W1_norm


def extract_weights_tf(W1, max_lag):
    '''
    Convert W1 ((p * K) x n_H) to
    an tensor of size (K x p) by computing 
    the L2-norm along the last axis, to be 
    supplied to an image summary in TensorFlow.
    '''
    return tf.transpose(tf.reshape(tf.norm(W1, axis=1), (1, -1, max_lag, 1)), perm=(0, 2, 1, 3))


def normalize_in_place(df):
    '''
    Normalize a Pandas dataframe in-place. 

    Normalization here is standardization:
        (df - mean(df)) / stddev(df)
    '''
    for key in df.columns:
        df[key] = (df[key] - np.mean(df[key])) / np.std(df[key])
    return df