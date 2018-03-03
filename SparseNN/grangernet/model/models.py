import tensorflow as tf
from ..private import regularizers, utils

def build_granger_net(input_shape, lambda_, reg_mode, max_lag, n_H=32):
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

    Inputs:
        input_shape: Input dimensions (int)
        lambda_:     Regularization weight (float)
        reg_mode:    Regularization mode. 
                     Valid options include ['L1', 'L2', 'gL2', 'hL1', 'hL2']
        max_lag:     Maximum lag in calculations
        n_H:         Number of units in hidden layer

    Returns (in order):
        placeholders: _X and _Y
        tensors:      W1, total_cost
        optimizer:    opt
    '''
    assert reg_mode in ['L1', 'L2', 'gL2', 'hL1', 'hL2']

    # Define regularization function mapping
    reg_func = {
        'L1': regularizers.L1_regularizer,
        'L2': regularizers.L2_regularizer,
        'gL2': regularizers.group_L2_regularizer,
        'hL1': regularizers.hierarchical_L1_regularizer,
        'hL2': regularizers.hierarchical_L2_regularizer,
    }
    
    # Create placeholders
    _X = tf.placeholder(tf.float32, shape=(None, *input_shape), name='X')
    _Y = tf.placeholder(tf.float32, shape=(None, 1), name='Y')
    

    # Hidden layer
    with tf.variable_scope('LAYER1'):
        W1 = tf.Variable(tf.random_normal((_X.get_shape().as_list()[1], n_H)), name='W1')
        b1 = tf.Variable(tf.random_normal((1, n_H)), name='b1')

        with tf.device('/gpu:0'):
            ACT1 = tf.nn.relu(_X @ W1 + b1, name='ACTV1')
        
        # Add image summary for W1
        tf.summary.image('W1', utils.extract_weights_tf(W1, max_lag), max_outputs=1)
        
    # Output layer
    with tf.variable_scope('OUTPUT'):
        W_out = tf.Variable(tf.random_normal((n_H, _Y.get_shape().as_list()[1])), name='W_OUTPUT')
        b_out = tf.Variable(tf.random_normal((1, 1)), name='b_OUTPUT')

        with tf.device('/gpu:0'):
            Y_pred = tf.add(ACT1 @ W_out, b_out, name='PREDICTION')
        
    # Define the cost function
    with tf.variable_scope('LOSS'), tf.device('/gpu:0'):
        with tf.variable_scope('RECONSTRUCTION_LOSS'):
            reconstruction_loss = tf.reduce_mean((_Y - Y_pred)**2)

            # Add summary for reconstruction loss
            tf.summary.scalar('reconstruction_loss', reconstruction_loss)
        with tf.variable_scope('REGULARIZATION_LOSS'):
            reg_loss = reg_func[reg_mode](W1, lambda_, max_lag)

        # Calculate total loss
        total_loss = tf.add(reconstruction_loss, reg_loss, name='TOTAL_LOSS')

        # Add summaries
        tf.summary.scalar('total_loss', total_loss)
    
    # Define Adam Optimzer to minimise total_loss
    opt = tf.train.AdamOptimizer().minimize(total_loss)
    
    return _X, _Y, W1, total_loss, opt