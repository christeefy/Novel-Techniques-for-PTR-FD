import tensorflow as tf
import numpy as np

def L1_regularizer(W, lambda_, max_lag):
    '''
    A basic regularizer that ensure sparsity amongst all inputs (all 'pK' inputs) of the neural network.
    It computes the L1-norm of weights.

    Arguments:
        W: Encoder weights of shape (pK x n_H)
        lambda_: Regularization weight
        max_lag: Dummy variable to ensure consistent argument 
                 signature across all regularizers.

    Returns:
        A tf scalar.
    '''
    return lambda_ * tf.reduce_sum(tf.abs(W))

def hierarchical_L1_regularizer(W, lambda_, max_lag):
    '''
    A regularizer that ensure that encourages encoder weight sparsity of predictor groups and 
    penalizes more lagged inputs. 
    
    It computes the L1-norm of weights hierarchically — an additional sum if done over the lags.

    Arguments:
        W: Encoder weights of shape (pK x n_H)
        lambda_: Regularization weight
        max_lag: Maximum number of lagged values used as neural network inputs

    Returns:
        A tf scalar.
    '''
    # Infer dimensions of W
    _p_times_lag, n_H = W.shape.as_list()
    p = _p_times_lag // max_lag
    
    # Reshape W
    _W = tf.reshape(W, shape=(-1, max_lag, n_H))

    # Calculate the L1-norm
    _W = tf.abs(_W)
    
    # Optimization to parallelize hierarchical calculation analytically
    _W = tf.range(1, max_lag + 1, dtype=tf.float32) * tf.reduce_sum(_W, axis=[0, 2])
    
    return lambda_ * tf.reduce_sum(_W)

def L2_regularizer(W, lambda_, max_lag):
    '''
    A basic regularizer that prevents overfitting by minimizing
    all encoder weights simultaneously.

    It computes the L2-norm. 

    Arguments:
        W: Encoder weights of shape (pK x n_H)
        lambda_: Regularization weight
        max_lag: Dummy variable to ensure consistent argument 
                 signature across all regularizers.

    Returns:
        A tf scalar.
    '''
    return lambda_ * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(W), axis=-1)))  

def group_L2_regularizer(W, lambda_, max_lag):
    '''
    A regularizer that encourages sparsity amongst groups of predictors ('p' inputs) 
    of the neural network. If a predictor is non-causal, all lagged values inputs of
    that predictor are minimized simultaneously.

    It computes the L2-norms by groups of predictors.

    Arguments:
        W: Encoder weights of shape (pK x n_H)
        lambda_: Regularization weight
        max_lag: Dummy variable to ensure consistent argument 
                 signature across all regularizers.

    Returns:
        A tf scalar.
    '''
    return lambda_ * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(W), axis=[-1, -2])))

def hierarchical_L2_regularizer(W, lambda_, max_lag):
    '''
    A regularizer that ensure that encourages encoder weight sparsity of predictor groups and 
    penalizes more lagged inputs. 
    
    It computes the L2-norm of weights hierarchically — an additional sum if done over the lags.

    Arguments:
        W: Encoder weights of shape (pK x n_H)
        lambda_: Regularization weight
        max_lag: Maximum number of lagged values used as neural network inputs

    Returns:
        A tf scalar.
    '''

    # Infer dimensions of W
    _p_times_lag, n_H = W.shape.as_list()
    p = _p_times_lag // max_lag
    
    # Reshape W
    _W = tf.reshape(W, shape=(-1, max_lag, n_H))

    # Calculate the L2-norm
    _W = tf.square(_W)
    
    # Optimization to parallelize hierarchical calculation analytically
    _W = tf.range(1, max_lag + 1, dtype=tf.float32) * tf.reduce_sum(W, axis=[0, 2])
    
    return lambda_ * tf.reduce_sum(tf.sqrt(_W))