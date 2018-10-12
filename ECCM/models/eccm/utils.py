import pandas as pd
import numpy as np
from scipy.stats import t

def _calc_significance(rhos, N):
    ''' (pd.Series, int) -> pd.Series
    Calculate the p-value of the Pearson correlation coefficient `rho` 
    using Student's t-distribution.
    '''
    return t.sf(rhos * np.sqrt(N / (1 - rhos**2)), N)

def _URC_cross_map_lags(DF):
    '''
    Calculate the critical cross map lag of each causality series 
    baed on the Upper Right Corner (URC) criteria.
    
    Critical cross map lag is defined as the cross map lag before
    the lag corresponding to the maximum discrete concativity.
    Concativity can be calculated using discrete first- and second-
    order derivatives.
    
    Inputs:
        DF: An (N x P) DataFrame containing causalities for P 
            variable pairs at N cross map lags.
    
    Returns:
        A (P* x 1) DataFrame containing the critical URC cross map
        lags, where 0 <= P* <= P. 
    '''
    # Concatenate first- and second-order derivatives into a 
    # DataFrame with keys 'delta' and 'doubleDelta'
    _derivAllVarsDF = pd.concat([DF.diff(2).shift(-1)/2, DF.shift(-1) + DF.shift(1) - 2 * DF], 
                                axis=1, 
                                keys=['delta', 'doubleDelta'])
    
    # Create a list to store critical URC cross maps
    URCLags = []
    
    # Iterate through every variable pair
    for key in DF.columns:
        # Create a (N x 2) DataFrame containing the 
        # first- and second- order derivative of 
        # a single variable pair
        _derivSingleVarDF = _derivAllVarsDF.loc[:, _derivAllVarsDF.columns.get_level_values(1) == key]
        
        # Calculate critical cross map lag
        # 1. Filter values in _df which are strictly nonpositive
        # 2. Drop rows that contain do not match filter criteria
        # 3. Select 'doubleDelta' column
        # 4. Obtain index of minimum 'doubleDelta' less one,
        #    as a Pandas Series
        try:
            critXMapLag = ((_derivSingleVarDF[_derivSingleVarDF.apply(lambda x: x <= 0)])
                           .dropna())['doubleDelta'].idxmin()
        except ValueError:
            critXMapLag = pd.Series(None, index=[key])
        
        # Append critical lags to `URCLags`
        URCLags.append(critXMapLag)

    # Create DataFrame of critical lags
    return pd.DataFrame(pd.concat(URCLags), columns=['URC xMap Lags'])

def _peak_causality_coordinates(DF):
    '''
    Calculate the coordinate of causality peak
    for each variable pair in DF. 

    Returns a Pandas DataFrame. 
    '''
    # Create a new Pandas DataFrame
    peaksDF = pd.DataFrame()

    # Calculate peak causalities
    peaksDF['Peak Causality'] = DF.max().round(3)

    # Obtain peak cross map lag
    peaksDF['Peak xMap Lag'] = DF.idxmax()
    
    # Calculate signifance
    peaksDF['Peak Sig.'] = _calc_significance(peaksDF['Peak Causality'], 
                                              len(DF - peaksDF['Peak xMap Lag'] - len(DF.columns) + 1))
    
    # Calculate Upper Right Corner (URC) cross map lag
    peaksDF['URC xMap Lag'] = _URC_cross_map_lags(DF)
    
    # Calculate URC Causality
    URCCausalities = []
    for (index, value) in peaksDF['URC xMap Lag'].iteritems():
        # If cross map lag is None, causality is also None
        URCCausalities.append(None if pd.isnull(value) else round(DF.loc[value, index], 3))
    peaksDF['URC Causality'] = pd.Series(URCCausalities, index=peaksDF.index)
    
    # Calculate signifance
    peaksDF['URC Sig.'] = _calc_significance(peaksDF['URC Causality'], 
                                              len(DF - peaksDF['URC xMap Lag'] - len(DF.columns) + 1))

    # Rearrange columns and return DataFrame
    return peaksDF[['Peak Causality', 'Peak xMap Lag', 'Peak Sig.',
                    'URC Causality', 'URC xMap Lag', 'URC Sig.']]