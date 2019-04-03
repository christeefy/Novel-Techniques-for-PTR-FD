import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm, tqdm_notebook

from ..models import granger_net as granger_net_model

from ..private import utils as private_utils
from ..private.gpu import utils as gpu_utils

from ... import causality_viz
from ...utils import in_ipynb

import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _parse_arguments():
    # Build argument parser
    parser = argparse.ArgumentParser('Script to invoke the Granger Net function for process topology reconstruction.')
    parser.add_argument('csv', help='Location of csv file containing data for process topology reconstruction.')
    parser.add_argument('max_lag', help='Number of past time series values to consider as inputs.', type=int)
    parser.add_argument('-r', '--run_id', help='Run identifier. Default value is None. Include value to enable TensorBoard logging.', default=None)
    parser.add_argument('-l', '--lambda_', help='Encoder weights regularization parameter. Default value is 0.1.', default=0.1)
    parser.add_argument('-m', '--reg_mode', help='Regularization mode. Default mode is hL1', type=str, default='hL1')
    parser.add_argument('-n', '--n_H', help='Number of hidden units at the encoder layer. Default value is 32.', type=int, default=32)
    parser.add_argument('-e', '--epochs', help='Maximum number of epochs for training. Default value is 3000.', type=int, default=3000)
    parser.add_argument('-s', '--early_stopping', help='Whether to perform early stopping during training. Default value is True.', type=bool, default=True)
    parser.add_argument('-a', '--autocausation', help='Whether to evaluate autocausation. Default value is True', type=bool, default=True)
    parser.add_argument('-b', '--initial_batch_size', help='Initial batch size. Default value is 32.', type=int, default=32)
    parser.add_argument('-i', '--batch_size_interpolation', help='Method to interpolate from initial and final batch size. Default value is "exp_step".', default='exp_step', choices=['exp_step', 'linear', 'step'])
    parser.add_argument('-t', '--threshold', help='Percentage of maximum causality value to use as threshold to convert heatmap into binary heatmap. Binary heatmap is generated if provided, otherwise grayscale heatmap.', type=float)

    # Parse arguments
    args = parser.parse_args()

    return args


def granger_net(df, max_lag, norm=True, run_id=None, lambda_=0.1, lambda_output=0., reg_mode='hL1', n_H=32, epochs=3000, \
            early_stopping=True, autocausation=True, \
            initial_batch_size=32, batch_size_interpolation='exp_step'):
    '''
    Script to perform Granger Net calculations.

    Arguments:
        df:                       Pandas dataframe containing time series data
        max_lag:                  Maximum number of lagged values to consider as inputs (int)
        run_id:                   (Optional) Run identifier (str). Default value is None. Include value to enable TensorBoard logging.
        lambda_:                  Regularization weight to apply to decoder weights. L2-regularizatoin is used. Default value is zero. (float)
        lambda_output:            Regularization weight to apply to encoder weights (float)
        reg_mode:                 Regularization scheme to apply to encoder weights (str). Possible values include: {'L1', 'L2', 'gL2', 'hL1', 'hL2'}.
        n_H:                      Number of hidden units of the encoder layer (int)
        epochs:                   Maximum number of epochs for training (int)
        early_stopping:           Boolean on whether to prematurely end training if loss value does not improve after 10% of `epochs`.
        autocausation:            Boolean on whether to evaluate autocausation
        initial_batch_size:       Initial batch size to use for training (int)
        batch_size_interpolation: Method to interpolate from initial and final batch size (10% of length of `df`).
                                  Possible values include: {'step', 'exp_step', 'linear'}.

    Returns:
        The numpy array of shape (p x p x max_lag) containing the trained encoder weights.
    '''
    # Assertion checks
    assert isinstance(df, pd.DataFrame), 'Make sure first positional argument is a Pandas dataframe'
    assert epochs >= 101, 'Epochs must be at least 101 for summaries to work'
    assert initial_batch_size <= len(df) - max_lag - 1, 'Given the data, batch size cannot exceed {}'.format(len(df) - max_lag - 1)

    # Log start time
    START_TIME = datetime.datetime.now()
    START_TIME_DIR = START_TIME.strftime('%b %d, %Y/%I.%M%p')

    # Standardize values in df
    private_utils.normalize_in_place(df)

    # Infer number of variables
    p = len(df.columns)

    # Get number of GPUs (Use 1 if there are not GPUs)
    num_GPUs = max(1, gpu_utils.get_num_gpus())

    # Create empty causality array
    W = []

    # Define a shuffling index
    shuffle_idx = np.arange(gpu_utils.get_truncation_idx(len(df) - max_lag - 1, num_GPUs))

    # Choose appropriate tqdm function
    tqdm_func = tqdm_notebook if in_ipynb() else tqdm

    for (i, var) in tqdm_func(enumerate(df.columns),
                              desc='Cycling through variables',
                              otal=len(df.columns),
                              leave=False):

        # Obtain data and target for NN
        X, Y = private_utils.create_dataset(df, var, max_lag,
                                    autocausation=autocausation)

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
            _X, _Y, W1, _loss, optimizer, reconstruction_loss = granger_net_model.build_graph(X[0].shape, max_lag, lambda_, reg_mode,
                                                                   num_GPUs=num_GPUs,
                                                                   pos=i,
                                                                   autocausation=autocausation,
                                                                   lambda_output=lambda_output,
                                                                   n_H=n_H)

            # Create summary writer
            if run_id is not None:
                LOG_DIR = 'Logs/{}/{}/{}'.format(run_id, START_TIME_DIR, var)
                summary_writer = tf.summary.FileWriter(LOG_DIR,
                                                       tf.get_default_graph())
                merged = tf.summary.merge_all()

            # Initialise all TF variables
            tf.global_variables_initializer().run()

            # Create saver to W1
            saver = tf.train.Saver({'W1': W1}, max_to_keep=1)

            if run_id is not None:
                # Calculate initial loss prior to training
                summary = sess.run(merged, feed_dict={
                    _X: X[:(initial_batch_size * num_GPUs)],
                    _Y: Y[:(initial_batch_size * num_GPUs)][:, np.newaxis]
                })
                summary_writer.add_summary(summary, 0)

            # Define batch size scheduler
            batch_size_scheduler = private_utils.generate_batch_size_scheduler(epochs,
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

                # Log summaries
                if (epoch + 1) % min(50, epochs // 100) == 0:
                    if run_id is not None:
                        summary = sess.run(merged, feed_dict={
                            _X: X[shuffle_idx][:(batch_size * num_GPUs)],
                            _Y: Y[shuffle_idx][:(batch_size * num_GPUs)][:, np.newaxis]
                        })

                        summary_writer.add_summary(summary, epoch + 1)

                # Check for early stopping
                if loss < early_stop['loss']:
                    early_stop['loss'] = loss
                    early_stop['epoch'] = epoch + 1

                    # Save best W1 values
                    _best_W1 = sess.run(W1)

                elif (epoch + 1) - early_stop['epoch'] >= (epochs // 10 if early_stopping else epochs + 1):
                    #print('Exited due to early stopping.')
                    break

            if run_id is not None:
                # Log summary upon completing training
                summary = sess.run(merged, feed_dict={
                    _X: X[shuffle_idx][:(batch_size * num_GPUs)],
                    _Y: Y[shuffle_idx][:(batch_size * num_GPUs)][:, np.newaxis]
                })

                summary_writer.add_summary(summary, epoch + 1)

                # Ensure pending summaries are written to disk
                summary_writer.flush()

            # Obtain weights and append to main array
            W.append(private_utils.extract_weights(_best_W1, max_lag, pos=i, autocausation=autocausation))

    # Create a np array from W
    W = np.array(W)

    if norm:
        W = np.linalg.norm(W, axis=-1)

    return W


if __name__ == '__main__':
    # Parse arguments
    args = _parse_arguments()

    # Obtain data from csv
    df = pd.read_csv(args.csv)

    # Run Granger Net algorithm
    W = granger_net(df=df, **{k: v for k, v in vars(args).items() if k not in ['csv', 'threshold']})

    # Visualize granger net
    print('Generating causal heatmap...')
    if args.threshold is None:
        causality_viz.causal_heatmap(W, df.columns, mode='joint')
    else:
        causality_viz.causal_heatmap(W, df.columns, mode='joint_threshold', threshold=args.threshold)