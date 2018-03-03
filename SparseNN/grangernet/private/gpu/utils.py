from tensorflow.python.client import device_lib

def get_num_gpus():
    '''
    Obtain the number the GPUs available on machine.
    '''
    return len([device for device in device_lib.list_local_devices() if device.device_type == 'GPU'])

def get_truncation_idx(N, gpus):
    '''
    Obtain the index to truncate the data so that
    all partitions are of equal size. 
    '''
    quotient = N // gpus
    return quotient * gpus