import numpy as np
import pandas as pd
from scipy.integrate import odeint
import datetime

def generate_ex1(N, seed=int(datetime.datetime.now().timestamp())):
    '''
    Generate data for Example 1.

    Input:
        n: Time-series length

    Returns:
        A Pandas DataFrame
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

    # Generate dependent variables
    for k in range(N - 1):
        ex['Y'][k + 1] = 0.8 * ex['X'][k] + 0.5 * ex['Y'][k] + ex['v1'][k]
        ex['Z'][k + 1] = 0.6 * ex['Y'][k] + ex['v2'][k]
        
    return pd.DataFrame(ex)


def generate_ex2(N, seed=int(datetime.datetime.now().timestamp())):
    '''
    Generate data for Example 2.

    Input:
        n: Time-series length

    Returns:
        A Pandas DataFrame
    '''
    # Set random seed
    np.random.seed(seed)
    
    # Define an empty dict
    ex = {}
    
    # Convert n to int
    N = int(N)

    # Generate source variables
    ex['X'] = np.random.uniform(4, 5, size=(N,))
    ex['v1'] = np.random.normal(0, 0.05, size=(N,))
    ex['v2'] = np.random.normal(0, 0.05, size=(N,))

    # Initialise dependent variables
    ex['Y'] = np.zeros((N,))
    ex['Y'][0] = 0.2
    ex['Z'] = np.zeros((N,))

    # Generate dependent variables
    for k in range(N - 1):
        ex['Y'][k + 1] = 1 - 2 * abs(0.5 - (0.8 * ex['X'][k] + 0.4 * (abs(ex['Y'][k])**0.5))) + ex['v1'][k]
        ex['Z'][k + 1] = 5 * (ex['Y'][k] + 7.2)**2 + 10 * (abs(ex['X'][k]))**0.5 + ex['v2'][k]
        
    return pd.DataFrame(ex)


def generate_ex3():
    '''
    Create a DataFrame based on pre-simulated data.
    '''
    return pd.read_csv('Data/ex3.csv', delimiter=',')


def generate_ex4(N, seed=int(datetime.datetime.now().timestamp())):
    '''
    Generate data for Example 2.

    Input:
        n: Time-series length

    Returns:
        A Pandas DataFrame
    '''
    # Set random seed
    np.random.seed(seed)
    
    # Define an empty dict
    ex = {}
    
    # Convert n to int
    N = int(N)

    # Generate source variables
    ex['X'] = np.random.uniform(4, 5, size=(N,))
    ex['v1'] = np.random.normal(0, 0.05, size=(N,))
    ex['v2'] = np.random.normal(0, 0.05, size=(N,))

    # Initialise dependent variables
    ex['Y'] = np.zeros((N,))
    ex['Y'][0] = 0.2
    ex['Z'] = np.zeros((N,))

    # Generate dependent variables
    for k in range(N - 1):
        ex['Y'][k + 1] = ex['X'][k]**2
        ex['Z'][k + 1] = ex['Y'][k]**(1/2)
        
    return pd.DataFrame(ex)


def nonisothermal_CSTR():
	'''
	Loads the simulated values from MATLAB Simulink of 
	a nonisothermal CSTR from Marlin's Process Control 
	textbook.
	'''
	return pd.read_csv('../Data/nonisothermal_CSTR.csv', delimiter=',')


def rossler(t, initial_state=[9.1901e-4, 1.4984e-3, 0.58254, 0.13225], d_mode='constant', **kwargs):
    '''
    Returns a rossler system consisting of 
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



def predator_prey(gamma_xy=0, gamma_yx=0.32, r_x=3.7, r_y=3.8, N=1000, randomise=False):
    '''
    Generate values for a 2-species predator-prey model.
    '''
    def switch(a, b):
        '''
        Switches the values of 'a' and 'b'
        '''
        return b, a
    
    def generate_random_periods(N, low, high):
        '''
        Generate an array containing cumulative period values of when the coupling parameters should be switch.
        Inputs:
            N:    Length of output
            low:  Lower bound for random beta sampling
            high: Upper bound for random beta sampling
        '''
        # Generate an array of random integers between low and high sampled from a beta distribution
        temp = np.array([])
        while np.sum(temp) < N:
            temp = np.append(temp, np.random.randint(low, high))
        
        # Create cumulative period length counter
        n = 0
        for i in range(len(temp) - 1):
            temp[i + 1] += temp[i]
        temp[-1] = N
        
        return temp
      
    
    ###################
    # Function begins #
    ###################
    
    # Add 20 time points to N to account for burn-in
    N += 20
    
    ex = {
        'X': np.zeros((N,)),
        'Y': np.zeros((N,)),
        'gamma_xy': np.zeros((N,)),
        'gamma_yx': np.zeros((N,)),
    }
    
    ex['X'][0] = np.random.uniform(0, .1)
    ex['Y'][0] = np.random.uniform(0, .1)
    
    # If is_random = True, generate array of random time indices to switch gamma parameters:
    if randomise == True:
        periods = generate_random_periods(N, 50, 200)
        ex['periods'] = periods
    
    for k in range(N - 1):
        if (randomise == True) and (k == periods[0] - 1):
            gamma_xy, gamma_yx = switch(gamma_xy, gamma_yx)
            periods = np.delete(periods, 0)
            
        ex['X'][k + 1] = ex['X'][k] * (r_x - r_x * ex['X'][k] - gamma_xy * ex['Y'][k])
        ex['Y'][k + 1] = ex['Y'][k] * (r_y - r_y * ex['Y'][k] - gamma_yx * ex['X'][k])
        ex['gamma_xy'][k + 1] = gamma_xy
        ex['gamma_yx'][k + 1] = gamma_yx
        
    ex['X'] = ex['X'][20:]
    ex['Y'] = ex['Y'][20:]
    ex['gamma_xy'] = ex['gamma_xy'][20:]
    ex['gamma_yx'] = ex['gamma_yx'][20:] 
        
    return pd.DataFrame(ex)

def eastman(PVs_only=False, oscillating_only=False):
    '''
    Load the Eastman dataset.

    Inputs:
        PVs_only: Load variables that are only PVs
        oscillating_only: Load variables that are known to 
                          share a common oscillation frequency.
    '''
    # Read data
    eastman_df = pd.read_csv('../Data/eastman.csv')

    # Sort columns (in-place)
    eastman_df.sort_index(axis=1, inplace=True)

    if PVs_only is False:
        return eastman_df

    # Define key for oscillating variables
    osc_var = ['PC2.PV',
               'TC1.PV', 'TC2.PV',
               'FC1.PV', 'FC5.PV', 'FC8.PV',
               'LC1.PV', 'LC2.PV']
        
    # Define empty dataframe
    filtered = pd.DataFrame()
    
    for field, series in eastman_df.iteritems():
        # Filter out non-PV columns
        if ('PV' not in field):
            continue
        
        if (oscillating_only == True) and (field in osc_var):
            filtered = pd.concat([filtered, series], axis=1)
            continue
        elif (oscillating_only == False):
            filtered = pd.concat([filtered, series], axis=1)

    # Sort columns of filtered df
    filtered.sort_index(axis=1, inplace=True)
    
    return filtered