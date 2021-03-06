import tensorflow as tf
from ..private import regularizers, utils
from ..private.gpu import utils as gpu_utils

def _define_vars(input_shape, n_H):
    '''
    Helper function to define shared variables.
    '''

    # Define variables in each layers
    with tf.device('/cpu:0'):
        with tf.name_scope('VARIABLE'):
            with tf.variable_scope('LAYER1'):
                gpu_utils.create_vars_on_CPU('W1', (input_shape, n_H))
                gpu_utils.create_vars_on_CPU('b1', (1, n_H))

            with tf.variable_scope('OUTPUT'):
                gpu_utils.create_vars_on_CPU('W_OUTPUT', (n_H, 1))
                gpu_utils.create_vars_on_CPU('b_OUTPUT', (1, 1))


def _build_tower(inputs, targets, reg_mode, max_lag, lambda_):
    '''
    For each gpu, create a tower replica 
    and return the loss of that tower. 
    '''

    # Layer 1
    with tf.variable_scope('LAYER1'):
        W1 = tf.get_variable('W1')
        b1 = tf.get_variable('b1')

        ACT1 = tf.nn.relu(inputs @ W1 + b1, name='ACTV1')

    # Output layer
    with tf.variable_scope('OUTPUT'):
        W_out = tf.get_variable('W_OUTPUT')
        b_out = tf.get_variable('b_OUTPUT')

        Y_pred = tf.add(ACT1 @ W_out, b_out, name='PREDICTION')

    # Define the cost function
    with tf.variable_scope('LOSS'):
        with tf.variable_scope('RECONSTRUCTION_LOSS'):
            reconstruction_loss = tf.reduce_mean(tf.square(targets - Y_pred))

            # Add summary for reconstruction loss
            tf.summary.scalar('reconstruction_loss', reconstruction_loss)

            # Add tensor to collection to compute average reconstruction loss
            tf.add_to_collection('RECONSTRUCTION_LOSS', reconstruction_loss)

    return reconstruction_loss


def build_graph(input_shape, max_lag, lambda_, reg_mode, num_GPUs, pos, lambda_output, autocausation=True, n_H=32):
    '''
    Builds a fully connected neural network with one hidden layer. 
    Hidden layer uses a ReLU activation function. 
    Output layer has no activation function.

    Regularization is applied to the layer 1 weights only.

    Training is done using the Adam Optimizer.

    Summaries are applied to:
        W1 (image summary)
        total_cost (scalar summary)

    This implementation assumes there is soft device placement 
    for a single GPU.

    Arguments:
        input_shape: Input dimensions (int)
        lambda_:     Regularization weight (float)
        reg_mode:    Regularization mode. 
                     Valid options include ['L1', 'L2', 'gL2', 'hL1', 'hL2']
        max_lag:     Maximum lag in calculations
        n_H:         Number of units in hidden layer
        var_names:   List of names of time series variables
        num_GPUs:    Number of GPUs available on machine 
                     (if CPU only, then this value is 1).

    Returns (in order):
        placeholders: _X and _Y
        tensors:      W1, avg_loss
        optimizer:    opt
    '''
    assert reg_mode in ['L1', 'L2', 'gL2', 'hL1', 'hL2']
    
    # Define a global step counter
    global_step = tf.get_variable('global_step', [], trainable=False,
                                  initializer=tf.constant_initializer(0))

    # Create placeholders
    with tf.device('/cpu:0'), tf.name_scope('PLACEHOLDERS'):
        _X = tf.placeholder(tf.float32, shape=(None, *input_shape), name='INPUTS')
        _Y = tf.placeholder(tf.float32, shape=(None, 1), name='TARGETS')

        _X_split = tf.split(_X, num_GPUs, name='SPLITTED_INPUTS')
        _Y_split = tf.split(_Y, num_GPUs, name='SPLITTED_TARGETS')
    
    # Define variables
    _define_vars(*input_shape, n_H=n_H)

    # Define optimizer
    opt = tf.train.AdamOptimizer()

    # Obtain handle on W1
    with tf.variable_scope('LAYER1', reuse=True):
        W1 = tf.get_variable('W1')

        # Get image summary of W2
        with tf.variable_scope('IMAGE_SUMMARY'):
            tf.summary.image('W1', utils.extract_weights_tf(W1, max_lag, pos, autocausation=autocausation))

    # Create variable to store gradients and losses
    tower_grads, tower_losses = [], []

    # Compute regularization losses
    reg_func = {
        'L1': regularizers.L1_regularizer,
        'L2': regularizers.L2_regularizer,
        'gL2': regularizers.group_L2_regularizer,
        'hL1': regularizers.hierarchical_L1_regularizer,
        'hL2': regularizers.hierarchical_L2_regularizer,
    }

    with tf.name_scope('REGULARIZATION_LOSS'):
        with tf.name_scope('W1'):
            reg_loss_W1 = reg_func[reg_mode](W1, lambda_, max_lag)
            tf.summary.scalar('reg_loss_W1', reg_loss_W1)

        with tf.name_scope('W_out'):
            # Get handle on W_out
            with tf.variable_scope('OUTPUT', reuse=True):
                W_out = tf.get_variable('W_OUTPUT')

            reg_loss_W_output = reg_func['L2'](W_out, lambda_output, max_lag)
            tf.summary.scalar('reg_loss_W_output', reg_loss_W_output)

    # Build tower for each GPU
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        for gpu in range(num_GPUs):
            with tf.device('/gpu:{}'.format(gpu)):
                with tf.name_scope('Tower_{}'.format(gpu + 1)):
                    # Build tower and compute loss
                    loss = _build_tower(_X_split[gpu], _Y_split[gpu],
                                        reg_mode, max_lag, lambda_)

                    # Store losses
                    tower_losses.append(tf.add_n([loss, reg_loss_W1, reg_loss_W_output]))

                    # Compute gradients
                    grad = opt.compute_gradients(tower_losses[-1])
                    tower_grads.append(grad)

    with tf.device('/cpu:0'):
        # Average gradients
        with tf.name_scope('AVERAGE_GRADIENTS'):
            avg_grads = gpu_utils.average_gradients(tower_grads)

        # Include average losses in scalar summary
        with tf.name_scope('AVERAGE_LOSS'):
            avg_loss = tf.reduce_mean(tf.stack(tower_losses, axis=0))
            tf.summary.scalar('avg_loss', avg_loss)

            tf.summary.scalar('avg_reconstruction_loss', 
                              tf.reduce_mean(tf.get_collection('RECONSTRUCTION_LOSS')))

        # Update variables
        train_op = opt.apply_gradients(avg_grads, global_step=global_step)

    return _X, _Y, W1, avg_loss, train_op, loss