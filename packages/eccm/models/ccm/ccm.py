import argparse
import pandas as pd

from .utils import ccm_one_way
from ...private.utils import generate_delayed_df


def parse_arguments():
    # Build argument parser
    parser = argparse.ArgumentParser('Script for invoking CCM function for process topology reconstruction.')
    
    parser.add_argument('csv', help='Location of csv file containing data for process topology reconstruction.')
    parser.add_argument('a', help='First variable to perform causality analysis.')
    parser.add_argument('b', help='Second variable to perform causality analysis.')
    parser.add_argument('-k', help='Number of nearest-neighbours in simplex projection calculation. Default value is 3.', default=3, type=int)

    # Parse arguments
    args = parser.parse_args()

    return args


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


if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()

    # Extract csv
    df = pd.read_csv(args.csv)

    # Reconstruct process topology
    a_to_b, b_to_a = ccm(df, **{k: v for k, v in vars(args).items() if k != 'csv'})

    # Output results
    print(f'Causality of {args.a} to {args.b}: {a_to_b}')
    print(f'Causality of {args.b} to {args.a}: {b_to_a}')