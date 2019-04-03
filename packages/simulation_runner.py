from pathlib import Path
from typing import Callable, Tuple, Optional

import numpy as np
import pandas as pd

from tqdm import tqdm, tqdm_notebook
from functools import partial
from multiprocessing import cpu_count, Pool

from packages.utils import in_ipynb
from packages.metrics import PRF, MSE, fault_detected, AUCPR
from packages.metrics.utils import is_fault, metrics_list


def evaluate_simulations(N: int,
                         problem: str,
                         datagen: Callable,
                         causality_func: Callable,
                         dst: Path = Path('Results/'),
                         parallel=True,
                         chunksize: int = 1) -> None:
    '''
    Wrapper function to evaluate multiple simulations, potentially in parallel.
    For each simulation run, the process topology is calculated and
    subsequently evaluated against the corresponding ground truth.

    Arguments:
        N {int} -- If of type `int`, number of simulations to perform.
                   If iterable, they refer to the simulation seeds.
        problem {str} -- The particular example to simulate.
                         {'ex1', 'ex2', 'ex3', 'pred_prey', 'noniso'}
        datagen {Callable} -- A function to generate data.
        causality_func {Callable} -- A function to invoke PTR calculations.

    Keyword Arguments:
        dst {Path} -- Location to save results. (default: {Path('Results/')})
        parallel {bool} -- Whether to process simulations in parallel.
                           (default: {True})
        chunksize {int} -- For parallel processing, number of chunks to process
                           per process. (default: {1})

    Raises:
        AssertionError -- If invalid `problem` provided.
        AssertionError -- If invalid `causality_func` provided.
        TypeError -- If invalid `N` provided.
    '''

    # Parse inputs
    PROBLEMS = ('ex1', 'ex2', 'ex3', 'pred_prey', 'noniso')
    TECHNIQUES = ('GC', 'GN', 'ECCM')
    technique_keys = dict(zip(('granger_causality', 'granger_net', 'eccm'),
                              TECHNIQUES))
    technique = technique_keys[causality_func.__name__]
    dst = Path(dst)

    # Error checking
    if problem not in PROBLEMS:
        raise AssertionError('Invalid `problem` provided. '
                             f'Valid values are {PROBLEMS}.')
    if technique not in TECHNIQUES:
        raise AssertionError('Invalid causality function provided of type '
                             f'{causality_func.__name__}. Valid values are '
                             f'{TECHNIQUES}.')

    # Create iterables based on input type of N
    if isinstance(N, int):
        iters = range(N)
    elif isinstance(N, list) or isinstance(N, tuple):
        iters = N
    else:
        raise TypeError

    # Create appropriate tqdm func
    tqdm_func = tqdm_notebook if in_ipynb() else tqdm

    # Get truth matrix
    W_truth = np.load(f'Results/Ground_Truths/{problem}_gt.npz')['W_truth']

    if parallel:
        # Process operations in parallel
        with Pool(cpu_count()) as p:
            func = partial(_single_pass_eval,
                           causality_func=causality_func,
                           datagen=datagen,
                           technique=technique,
                           W_truth=W_truth)
            res = list(tqdm_func(p.imap(func, iters, chunksize=chunksize),
                                 desc=causality_func.__name__,
                                 total=len(iters)))
    else:
        # Otherwise, sequentially on a single CPU core
        res = []
        for i in tqdm_func(iters,
                           desc=causality_func.__name__,
                           total=len(iters)):
            res.append(_single_pass_eval(i,
                                         causality_func=causality_func,
                                         datagen=datagen,
                                         technique=technique,
                                         W_truth=W_truth))

    # Collect results and save into a csv file
    df = pd.DataFrame(res, columns=('Threshold', 'Precision', 'Recall', 'F1',
                                    'MSE', 'FaultDetected', 'AUCNPR'))
    Path(dst / f'Ex{PROBLEMS.index(problem) + 1}').mkdir(exist_ok=True,
                                                         parents=True)
    df.to_csv(dst / f'Ex{PROBLEMS.index(problem) + 1}' / f'{technique}.csv',
              index=False)


def _single_pass_eval(seed: int,
                      causality_func: Callable,
                      technique: str,
                      datagen: Callable,
                      W_truth: np.ndarray) -> Tuple[Optional[float],
                                                    float,
                                                    float,
                                                    float,
                                                    float,
                                                    Optional[float]]:
    '''
    Performs a single simulation. This includes (1) process topology
    reconstruction and (2) metrics evaluation.

    Returns:
        The metrics — threshold, MSE, precision, recall, fault_detected, AUCNPR
    '''

    # Generate data
    ex, _ = datagen(seed=seed)

    # Obtain causality matrix
    W = causality_func(ex)

    # Calculate metrics
    if technique != 'GN':
        # For GC and ECCCM
        threshold, prec, rec, F = PRF(W, W_truth).values()
        mse = MSE(W, W_truth)
        fault = fault_detected(W, is_fault(W_truth))
        aucnpr = None
    else:
        # For Granger Net
        # NOTE: W is of shape (K x K x n_hmod)
        # This step is done instead of `PRF` to obtain
        # the threshold that maximises the F1-score
        threshold, prec, rec, F = (
            sorted(metrics_list(W, W_truth),
                   key=lambda x: x['f_score'],
                   reverse=True)[0].values())

        # Obtain a thresholded binary causality matrix
        W_norm = np.linalg.norm(W, axis=-1)
        W_norm = W_norm >= (threshold * np.max(W_norm))

        # Calculate remaining metrics
        mse = MSE(W, W_truth, threshold=threshold)
        fault = fault_detected(
            W=W_norm,
            pos=is_fault(W_truth))
        aucnpr = AUCPR(W, W_truth, normalized=True)

    return (threshold, prec, rec, F, mse, fault, aucnpr)
