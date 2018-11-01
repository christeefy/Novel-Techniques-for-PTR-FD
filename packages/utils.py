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