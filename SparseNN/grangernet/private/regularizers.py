import tensorflow as tf
import numpy as np

def L1_regularizer(W1, lambda_, max_lag):
    return lambda_ * tf.reduce_sum(tf.abs(W1))

def hierarchical_L1_regularizer(W1, lambda_, max_lag):
    # Infer dimensions of W1
    _p_times_lag, n_H = W1.shape.as_list()
    p = _p_times_lag // max_lag
    
    # Reshape W1
    _W1 = tf.reshape(W1, shape=(-1, max_lag, n_H))
    
    # Optimization to parallelize over W1[:, k:, :] for k in range(K)
    _W1 = tf.concat([1**(-k) * _W1 * np.concatenate([np.zeros(shape=(p, k, n_H)), 
                                           np.ones(shape=(p, max_lag - k, n_H))], axis=1) \
                     for k in range(max_lag)], axis=0)
    
    return lambda_ * tf.reduce_sum(tf.abs(_W1))

# def hierarchical_L1_regularizer(W1, lambda_, max_lag):
#     # Infer dimensions of W1
#     _p_times_lag, n_H = W1.shape.as_list()
#     p = _p_times_lag // max_lag
    
#     # Reshape W1
#     _W1 = tf.reshape(W1, shape=(-1, max_lag, n_H))
    
#     # Optimization to parallelize over W1[:, k:, :] for k in range(K)
#     _W1 = tf.concat([_W1 * np.concatenate([np.zeros(shape=(p, k, n_H)), 
#                                            np.ones(shape=(p, max_lag - k, n_H))], axis=1) \
#                      for k in range(max_lag)], axis=0)
    
#     return lambda_ * tf.reduce_sum(tf.abs(_W1))

def L2_regularizer(W1, lambda_, max_lag):
    return lambda_ * tf.reduce_sum(tf.sqrt(tf.reduce_sum(W1**2, axis=-1)))  

def group_L2_regularizer(W1, lambda_, max_lag):
    return lambda_ * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.reduce_sum(W1**2, axis=-1), axis=-1)))

def hierarchical_L2_regularizer(W1, lambda_, max_lag):
    # Infer dimensions of W1
    _p_times_lag, n_H = W1.shape.as_list()
    p = _p_times_lag // max_lag
    
    # Reshape W1
    _W1 = tf.reshape(W1, shape=(-1, max_lag, n_H))
    
    # Optimization to parallelize over W1[:, k:, :] for k in range(K)
    _W1 = tf.concat([_W1 * np.concatenate([np.zeros(shape=(p, k, n_H)), 
                                           np.ones(shape=(p, max_lag - k, n_H))], axis=1) \
                     for k in range(max_lag)], axis=0)
    
    return lambda_ * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.reduce_sum(_W1**2, axis=-1), axis=-1)))

def hierarchical_L1_regularizer_nonlinear(W, n_SUBMOD, lambda_, max_lag):
    # Infer dimensions of W
    n_H1, n_H2 = W.shape.as_list()
    p = n_H1 // max_lag // n_SUBMOD
    
    # Reshape W
    _W = tf.reshape(W, shape=(p, max_lag, n_SUBMOD, n_H2))
    
    # Optimization to parallelize over W[:, k:, :] for k in range(K)
    _W = tf.concat([_W * np.concatenate([np.zeros(shape=(p, k, n_SUBMOD, n_H2)), 
                                         np.ones(shape=(p, max_lag - k, n_SUBMOD, n_H2))], axis=1) \
                    for k in range(max_lag)], axis=0)
    
    return lambda_ * tf.reduce_sum(tf.abs(_W))

def hierarchical_L2_regularizer_nonlinear(W, n_SUBMOD, lambda_, max_lag):
    # Infer dimensions of W
    n_H1, n_H2 = W.shape.as_list()
    p = n_H1 // max_lag // n_SUBMOD
    
    # Reshape W
    _W = tf.reshape(W, shape=(p, max_lag, n_SUBMOD, n_H2))
    
    # Optimization to parallelize over W[:, k:, :] for k in range(K)
    _W = tf.concat([_W * np.concatenate([np.zeros(shape=(p, k, n_SUBMOD, n_H2)), 
                                         np.ones(shape=(p, max_lag - k, n_SUBMOD, n_H2))], axis=1) \
                    for k in range(max_lag)], axis=0)
    
    return lambda_ * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(_W**2, axis=-1), axis=-1), axis=-1)))