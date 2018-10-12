from .utils import ccm_one_way
from ...private.utils import generate_delayed_df

def ccm(df, a, b, k=3):
    '''
    Perform convergent cross-mapping (CCM) algorithm described in paper.
    Calculates the causalities for `a`→`b` and `b`→`a`
    
    Arguments:
        DF:   A Pandas DataFrame
        a:    Variable name in DF to test for causality (str)
        b:    Variable name in DF to test for causality (str)
        k:    Number of nearest neighbours (scalar)
    
    Returns:
        A (float, float) tuple of causalities for `a`→`b` and `b`→`a`.
    '''

    # Generate a lagged dataframe
    _df = generate_delayed_df(df, b, a, cross_map_lag=0, embed_dim=len(df.columns))
    # Calculate `a`→`b`
    a_to_b = ccm_one_way(_df, b, a)

    # Generate a lagged dataframe
    _df = generate_delayed_df(df, a, b, cross_map_lag=0, embed_dim=len(df.columns))
    # Calculate `a`→`b`
    b_to_a = ccm_one_way(_df, a, b)

    return (a_to_b, b_to_a)


