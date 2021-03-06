import numpy as np
import tensorflow as tf
import pandas as pd

def create_dataset(df, var=None, max_lag=10, autocausation=True, include_shuffled_copy=False):
    '''
    Create X and Y for a `var`, based on a `df`. 
    X values will be selected up to `max_lag`. 

    If autocausation is False, past values of `var`
    will not be present in X. 
    
    Returns:
        X: Input data of shape (m, p * max_lag)
        Y: Output labels of shape (m,).
    '''
    if var is not None:
        assert var in df.columns
    elif var is None:
        assert include_shuffled_copy == True
    
    # Reverse df rows. This makes going down 
    # the rows to be going back in time. 
    df = df[::-1]
    
    # Obtain output label
    if var is not None:
        Y = np.array(df[var][:-(max_lag + 1)])
    else:
        Y = df.iloc[:-(max_lag + 1)].values
    
    # Remove column if autocausation is not True
    if not autocausation:
        df = df.drop(var, axis=1)

    # Create input data
    X = np.vstack([df[(i + 1):(max_lag + i + 1)].values.T.reshape(-1) for i in range(len(df) - 1 - max_lag)])

    # Add shuffled data along dimension-1
    if include_shuffled_copy:
        # Create shuffling index
        shuffle_idx = np.arange(len(X))

        # Shuffle index in-place
        np.random.shuffle(shuffle_idx)

        # Stack additional values
        X = np.stack([X, X[shuffle_idx]], axis=1)
        Y = np.stack([Y, Y[shuffle_idx]], axis=1)

    return X, Y
    

def extract_weights(W1, max_lag, pos, autocausation=True):
    '''
    Extract weights from W1 of size ((p * K) x n_H) 
    by computing the L2-norm for each weight along 
    the n_H axis.
    
    Returns:
        A np array of size p x K.
    '''
    # Compute the L2-normof W1 of size K x p
    W1_norm = np.linalg.norm(W1, axis=1).reshape(-1, max_lag, order='C')
    
    # Add -1 to the appropriate index 
    # if autocausation is not True
    if not autocausation:
        W1_norm = np.insert(W1_norm, pos, values=0, axis=0)
    
    return W1_norm


def extract_weights_tf(W1, max_lag, pos, autocausation=True):
    '''
    Convert W1 ((p * K) x n_H) to
    an tensor of size (K x p) by computing 
    the L2-norm along the last axis, to be 
    supplied to an image summary in TensorFlow.

    Returns a (1 x K x p x 1) tensor.

    Note: 
        Output's first dimension is the batch dim
        Output's last dimension is the channel dim
    '''
    W_norm = tf.transpose(tf.reshape(tf.norm(W1, axis=1), (1, -1, max_lag, 1)), perm=(0, 2, 1, 3))

    # Add -1 to the appropriate column
    # if autocausation is not True
    if not autocausation:
        return tf.concat([W_norm[..., :pos, :], 
                          tf.constant(0, tf.float32, 
                                      shape=(1, W_norm.get_shape().as_list()[1], 1, 1)),
                          W_norm[..., pos:, :]], 
                        axis=2)
    return W_norm


def normalize_in_place(df):
    '''
    Normalize a Pandas dataframe in-place. 

    Normalization here is standardization:
        (df - mean(df)) / stddev(df)
    '''
    epsilon = 1e-6

    for key in df.columns:
        df[key] = (df[key] - np.mean(df[key])) / (np.std(df[key]) + epsilon)
    return df

def generate_batch_size_scheduler(total_epochs, final_bs, initial_bs=32, interpolation='exp_step'):
    '''
    Decorator function that creates a function that returns the batch size given the current epoch.
    
    Arguments:
        total_epochs: Total number of epochs
        initial_bs: Initial batch size (minimum)
        final_bs: Final batch size (maximum)
        interpolation: Method to interpolate between initial and final batch size.
                       Possible values include: {'step', 'exp_step', 'linear'}
    
    Return: 
        Function that calculates the batch size at a given epoch. 
    '''
    assert initial_bs <= final_bs, \
    'final_batch_size of size {} must be greater than or equal to initial batch size of size {}.'.format(final_bs, initial_bs)
    
    assert interpolation in ['none', 'step', 'exp_step', 'linear'], \
    'Invalid interpolation parameter.'
    
    def _batch_size_scheduler(epoch):
        '''
        Calculate the appropriate batch size given the current epoch.
        '''
        if interpolation == 'step':
            num_steps = 5 # Number of steps to take
            step_factor = (final_bs - initial_bs) // num_steps
            batch_size = initial_bs + step_factor * ((num_steps + 1) * epoch // total_epochs)
        elif interpolation == 'exp_step':
            num_steps = 5 # Number of steps to take
            base = 2 # Exponent base
            
            exp_step_factor = (np.log(final_bs / initial_bs) / np.log(base)) / num_steps
            batch_size = initial_bs * base**((num_steps + 1) * epoch // total_epochs * exp_step_factor)
        elif interpolation == 'linear':
            batch_size = initial_bs + (final_bs - initial_bs) * epoch // (total_epochs)
        elif interpolation == 'none':
            batch_size = initial_bs
        
        return int(batch_size)
    return _batch_size_scheduler