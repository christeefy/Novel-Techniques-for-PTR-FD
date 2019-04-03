from functools import partial, update_wrapper


def is_interactive():
    '''
    Checks if Python is run in a Jupyter Notebook environment.
    Returns a Boolean.
    '''
    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            return True
        return False
    except NameError:
        return False


def curry(func, **kwargs):
    return update_wrapper(partial(func, **kwargs), func)


def in_ipynb():
    try:
        get_ipython()
        return True
    except NameError:
        return False
