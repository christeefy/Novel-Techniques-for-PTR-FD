import numpy as np
import pandas as pd
from scipy.integrate import odeint
import datetime

def generate_ex1(N, seed=int(datetime.datetime.now().timestamp())):
    '''
    Generate data for Example 1 from Ping's thesis.

    Arguments:
        N: Time-series length (int)
        seed: Random seed value (int)

    Returns:
        A tuple containing 
            - a Pandas DataFrame of shape (N x p)
            - system coefficients
    '''
    # Set random seed
    np.random.seed(seed)
    
    # Define an empty dict
    ex = {}
    
    # Convert n to int
    N = int(N)

    # Generate source variables
    ex['X'] = np.random.normal(0, 1, size=(N,))
    ex['v1'] = np.random.normal(0, 0.1, size=(N,))
    ex['v2'] = np.random.normal(0, 0.1, size=(N,))

    # Initialise dependent variables
    ex['Y'] = np.zeros((N,))
    ex['Y'][0] = 3.2
    ex['Z'] = np.zeros((N,))

    # Generate random coefficients
    k1, k2, k3 = np.random.uniform(size=(3,))

    # Generate dependent variables
    for t in range(N - 1):
        ex['Y'][t + 1] = k1 * ex['X'][t] + k2 * ex['Y'][t] + ex['v1'][t]
        ex['Z'][t + 1] = k3 * ex['Y'][t] + ex['v2'][t]

    # Map coeffs to dict
    coeffs = {k: v for (k, v) in zip(['k1', 'k2', 'k3'], (k1, k2, k3))}
        
    return pd.DataFrame(ex), coeffs


def generate_ex2(N, seed=int(datetime.datetime.now().timestamp())):
    '''
    Generate data for Example 2 from Ping's thesis.

    Input:
        N: Time-series length (int)
        seed: Random seed value (int)

    Returns:
        A tuple containing 
            - a Pandas DataFrame of shape (N x p)
            - system coefficients
    '''
    # Set random seed
    np.random.seed(seed)
    
    # Define an empty dict
    ex = {}
    
    # Convert n to int
    N = int(N)

    # Generate random parameters
    a = np.random.uniform(2, 4.5)
    b = np.random.uniform(4.6, 6)
    k1, k2, k3 = np.random.uniform(size=(3,))
    k4 = np.random.randint(2, 5)
    k5 = np.random.uniform(10, 30)

    # Generate source variables
    ex['X'] = np.random.uniform(a, b, size=(N,))
    ex['v1'] = np.random.normal(0, 0.05, size=(N,))
    ex['v2'] = np.random.normal(0, 0.05, size=(N,))

    # Initialise dependent variables
    ex['Y'] = np.zeros((N,))
    ex['Y'][0] = 0.2
    ex['Z'] = np.zeros((N,))

    # Generate dependent variables
    for k in range(N - 1):
        ex['Y'][k + 1] = 1 - 2 * abs(k1 - (k2 * ex['X'][k] + k3 * (abs(ex['Y'][k])**0.5))) + ex['v1'][k]
        ex['Z'][k + 1] = 5 * (ex['Y'][k] + 7.2)**k4 + k5 * (abs(ex['X'][k]))**0.5 + ex['v2'][k]

    # Map coeffs to dict
    coeffs = {k: v for (k, v) in zip(['a', 'b', 'k1', 'k2', 'k3', 'k4', 'k5'], (a, b, k1, k2, k3, k4, k5))}
        
    return pd.DataFrame(ex), coeffs


def generate_ex3(src, trial):
    '''
    Convert a csv containing simulated data for Example 3 of Ping's thesis 
    into a Pandas DataFrame.

    Arguments:
        src: Parent directory (folder) containing csv files for Example 3. 
        trial: Simulation id (int)

    Returns:
        A Pandas DataFrame of shape (N x p)
    '''
    assert trial in range(1, 11)
    return pd.read_csv(f'{src}/ex3_{trial}.csv', delimiter=',')


def nonisothermal_CSTR(src, trial):
	'''
	Convert a csv containing simulated values of a nonisothermal
    CSTR from Marlin's Process Control textbook into a Pandas
    DataFrame.

    Arguments:
        src: Parent directory (folder) containing csv files for the nonisothermal CSTR. 
        trial: Simulation id (int)

    Returns:
        A Pandas DataFrame of shape (N x p)
	'''
	return pd.read_csv(f'{src}/cstr_{trial}.csv', delimiter=',')


def rossler(t, initial_state=[9.1901e-4, 1.4984e-3, 0.58254, 0.13225], d_mode='constant', **kwargs):
    '''
    Generate a Rossler System consisting of 
    chemical species concentrations A, B, C and D. 

    Inputs:
        t:             List of time values where values are required
        initial_state: Initial states
        d_mode:        Mode of simulating input D. Valid choices 
                       include ['constant', 'periodic']

        **kwargs:
            amp:  Periodic amplitude
            freq: Periodic frequency
    '''
    def _rossler_reaction(state, t):
        # Obtain states
        a = state[0]
        b = state[1]
        c = state[2]
        d = state[3]

        # Define constants
        # Define constants
        k1 = 2.
        k2 = 0.4
        k3 = 1.0
        K = 1e-4
        k4 = 2e-3
        k5 = 0.5
        k6 = 2e-4
        k7 = 5e-3
        k8 = 6.8e-3

        # Generate dependent variables
        a_prime = k1 * a * c - k2 * a - k3 * a * b / (a + K) + k4 * d
        b_prime = k2 * a - k5 * b + k6
        c_prime = k7 - k1 * a * c - k8 * c
        
        if d_mode == 'constant':
            d_prime = 0
        elif d_mode == 'periodic':
            d_prime = kwargs['amp'] * kwargs['freq'] * np.cos(kwargs['freq'] * t)

        return [a_prime, b_prime, c_prime, d_prime]


    # Define valid d_modes
    assert d_mode in ['constant', 'periodic']
    if d_mode == 'periodic':
        assert {'amp', 'freq'} <= kwargs.keys(), 'Missing arguments'

    rosslerDF = pd.DataFrame(odeint(_rossler_reaction, initial_state, t), 
                             columns=['A', 'B', 'C', 'D'])

    return rosslerDF


def predator_prey_4_species(N, seed):
    '''
    Generate a four-species predator-prey system based on 
    ECCM's paper. 

    Arguments:
        N: Time-series length (int)
        seed: Random seed value (int)

    Returns:
        A tuple containing 
            - a Pandas DataFrame of shape (N x p)
            - system coefficients
    '''
    np.random.seed(seed + 100)

    ex = {
        'y1': np.zeros((N,)),
        'y2': np.zeros((N,)),
        'y3': np.zeros((N,)),
        'y4': np.zeros((N,)),
    }

    # Generate initial conditions
    for i in range(4):
        ex[f'y{i + 1}'][0] = 0.4

    # Generate random parameters
    r1, r2, r3, r4 = np.random.uniform(3.5, 4, size=(4,))

    # Generate gamma parameters
    r_21, r_32, r_43 = np.random.uniform(0.3, 0.4, size=(3,))

    for t in range(N - 1):
        ex['y1'][t + 1] = ex['y1'][t] * (r1 - r1 * ex['y1'][t])
        ex['y2'][t + 1] = ex['y2'][t] * (r2 - r2 * ex['y2'][t] - r_21 * ex['y1'][t])
        ex['y3'][t + 1] = ex['y3'][t] * (r3 - r3 * ex['y3'][t] - r_32 * ex['y2'][t])
        ex['y4'][t + 1] = ex['y4'][t] * (r4 - r4 * ex['y4'][t] - r_43 * ex['y3'][t])

    # Map coeffs to dict
    coeffs = {k: v for (k, v) in zip(['r1', 'r2', 'r3', 'r4', 'r_21', 'r_32', 'r_43'], (r1, r2, r3, r4, r_21, r_32, r_43))}        

    return pd.DataFrame(ex), coeffs


def eastman(PVs_only=False, oscillating_only=False, OSI=False):
    '''
    Load the Eastman dataset.

    Arguments:
        PVs_only: Load variables that are only PVs
        oscillating_only: Load variables that are known to 
                          share a common oscillation frequency.
        OSI: Load variables that match the OSI threshold as per Yuan's paper.
    '''
    if not ((not oscillating_only) and (not OSI)): 
        assert oscillating_only != OSI

    # Read data
    eastman_df = pd.read_csv('../Data/eastman.csv')

    if PVs_only:
        return (
            eastman_df[[col for col in eastman_df.columns if 'PV' in col]]
            .sort_index(axis=1, inplace=False))

    # Define key for oscillating variables
    if oscillating_only:
        osc_key = ['PC2.PV',
                   'TC1.PV', 'TC2.PV',
                   'FC1.PV', 'FC5.PV', 'FC8.PV',
                   'LC1.PV', 'LC2.PV']
        return (
            eastman_df[osc_key]
            .sort_index(axis=1, inplace=False))

    elif OSI:
        osc_key = [
        'LC1.PV', 'LC1.OP', 
        'FC1.PV', 'FC5.PV', 'FC5.OP', 'FC8.PV', 'FC8.OP',
        'LC2.PV', 'LC2.OP', 
        'TC1.PV', 'TC1.OP', 'TC2.PV', 'TC2.OP'
       ]

        return (
            eastman_df[osc_key]
            .sort_index(axis=1, inplace=False))

    return eastman_df.sort_index(axis=1, inplace=False)