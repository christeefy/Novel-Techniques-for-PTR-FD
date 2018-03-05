
import tensorflow as tf
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

def create_vars_on_CPU(name, shape, initializer=None):
    '''
    Helper function to initialise a variable on the CPU. 
    '''
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, tf.float32, 
                              initializer=initializer if initializer is not None else tf.truncated_normal_initializer())
    return var

def average_gradients(tower_grads):
    '''
    Average the gradient of all towers before updating variables on CPU.
    
    Inputs:
        tower_grads: A list of list of (grad, var) tuples. Inner list 
                     is for each gpu tower, while outer list is for each 
                     trainable var.
                     
    Returns:
        averaged_grads: A list of (avg_grad, var) tuples.
    '''
    averaged_grads = []
    for grads_var in zip(*tower_grads):
        # Grads_var is of type [(gpu0, var0), ..., (gpuN, var0)]
        
        # Concat and average extracted grads from grads_var
        grads = tf.concat([tf.expand_dims(grads, axis=0) for grads, _ in grads_var ], axis=0)
        avg_grad = tf.reduce_mean(grads, axis=0)
        
        # Append a copy of var to the average grads
        averaged_grads.append((avg_grad, grads_var[0][1]))
    return averaged_grads