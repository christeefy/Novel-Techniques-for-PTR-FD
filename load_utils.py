import numpy as np
from pathlib import Path

def save_results(filename, dst, **kwargs):
    '''
    Save computation results to dst/filename as a npz file. 
    W and hparams will be saved with keys 'W' and 'hparams'
    respectively. 

    Arguments:
        filename: Name of saved npz file.
        dst:      Parent directory of saved npz file.
        **kwargs: Key-value pairs of items to save. 
    '''

    # Create dst folder if it does not exist
    Path(dst).mkdir(exist_ok=True, parents=True)

    np.savez(f'{dst}/{filename}.npz', **kwargs)


def load_results(src):
    '''
    Load saved results of a npz file at `src`.

    Arguments:
        src: Location of npz file.

    Returns:
        A dictionary of saved values.
    '''
    return dict(np.load(src))