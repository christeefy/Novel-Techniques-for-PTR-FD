import tensorflow as tf
import numpy as np
import pandas as pd
import datetime

from ..models import granger_net

from ..private import utils
from ..private.gpu import utils as gpu_utils 

def analyze(df, max_lag, run_id='', lambda_=0.1, reg_mode='hL1', epochs=2000, \
            initial_batch_size=32, batch_size_interpolation='exp_step', \
            early_stopping=True, autocorrelate=True):
    # Assertion checks
    assert isinstance(df, pd.DataFrame), 'Make sure first positional argument is a Pandas dataframe'
    assert epochs >= 101, 'Epochs must be at least 101 for summaries to work'
    assert initial_batch_size <= len(df) - max_lag - 1, 'Given the data, batch size cannot exceed {}'.format(len(df) - max_lag - 1)

    # Log start time
    START_TIME = datetime.datetime.now()
    START_TIME_DIR = START_TIME.strftime('%b %d, %Y/%I.%M%p')

    # Standardize values in df
    utils.normalize_in_place(df)
    
    # Infer number of variables
    p = len(df.columns)
    
    # Get number of GPUs (Use 1 if there are not GPUs)
    num_GPUs = max(1, gpu_utils.get_num_gpus())
    
    # Create empty causality array
    W = []
    
    # Define a shuffling index
    shuffle_idx = np.arange(gpu_utils.get_truncation_idx(len(df) - max_lag - 1, num_GPUs))
    
    for (i, var) in enumerate(df.columns):
        print('Computing causality of {} (variable {} of {})...'.format(var, i + 1, p))
        
        # Obtain data and target for NN
        X, Y = utils.create_dataset(df, var, max_lag, 
                                    autocorrelate=autocorrelate)
        
        # Create early stopping variables
        early_stop = {
            'epoch': 0,
            'loss': 1e8
        }
        
        # Truncate data to be evely split amongst all GPUs
        even_split_idx = gpu_utils.get_truncation_idx(len(X), num_GPUs)
        X = X[:even_split_idx]
        Y = Y[:even_split_idx]
        
        assert len(shuffle_idx) == len(X), 'shuffle_idx of len {} while X of len {}'.format(len(shuffle_idx), len(X))

        # Reset default graph
        tf.reset_default_graph()
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Build model
            _X, _Y, W1, _loss, optimizer = granger_net.build_graph(X[0].shape, max_lag, lambda_, reg_mode, 
                                                                   num_GPUs=num_GPUs, 
                                                                   pos=i,
                                                                   autocorrelate=autocorrelate)
            
            # Create summary writer
            summary_writer = tf.summary.FileWriter('Logs/{}/{}/{}'.format(run_id, START_TIME_DIR, var),
                                                   tf.get_default_graph())
            merged = tf.summary.merge_all()

            # Initialise all TF variables
            tf.global_variables_initializer().run()
            
            # Calculate initial loss prior to training
            summary = sess.run(merged, feed_dict={
                _X: X[:(initial_batch_size * num_GPUs)], 
                _Y: Y[:(initial_batch_size * num_GPUs)][:, np.newaxis]
            })
            summary_writer.add_summary(summary, 0)
            
            # Define batch size scheduler
            batch_size_scheduler = utils.generate_batch_size_scheduler(epochs, 
                                                                       initial_bs=initial_batch_size, 
                                                                       final_bs=max(initial_batch_size, len(X) // 10),
                                                                       interpolation=batch_size_interpolation)

            # Train model
            for epoch in range(epochs):
                # Shuffle index
                np.random.shuffle(shuffle_idx)

                # Calculate batch size for current epoch
                batch_size = batch_size_scheduler(epoch)
                
                for batch in range(len(X) // batch_size // num_GPUs):
                    # Perform gradient descent
                    loss, _ = sess.run([_loss, optimizer], feed_dict={
                        _X: X[shuffle_idx][(batch * batch_size * num_GPUs):((batch + 1) * batch_size * num_GPUs)], 
                        _Y: Y[shuffle_idx][(batch * batch_size * num_GPUs):((batch + 1) * batch_size * num_GPUs)][:, np.newaxis]
                    })
                    
                # Perform summary logging and print statements
                if (epoch + 1) % min(50, epochs // 100) == 0:
                    if (epoch + 1) % min(1000, epochs // 5) == 0:
                        print('At epoch {}'.format(epoch + 1))

                    summary = sess.run(merged, feed_dict={
                        _X: X[shuffle_idx][:(batch_size * num_GPUs)], 
                        _Y: Y[shuffle_idx][:(batch_size * num_GPUs)][:, np.newaxis]
                    })

                    summary_writer.add_summary(summary, epoch + 1)
                        
                # Check for early stopping
                if loss < early_stop['loss']:
                    early_stop['loss'] = loss
                    early_stop['epoch'] = epoch + 1
                elif (epoch + 1) - early_stop['epoch'] >= (epochs // 10 if early_stopping else epochs + 1):
                    print('Exited due to early stopping.')
                    break
                        
            # Ensure pending summaries are written to disk
            summary_writer.flush()
            print()
            
            # Obtain weights and append to main array
            W1_ = sess.run(W1)
            W.append(utils.extract_weights(W1_, max_lag, pos=i, autocorrelate=autocorrelate))
            
    # Create a np array from W
    W = np.array(W)

    print('Analysis completed in {} mins {} secs'.format(*divmod((datetime.datetime.now() - START_TIME).seconds, 60)))
    
    return W