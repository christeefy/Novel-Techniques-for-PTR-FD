import pandas as pd

def generate_delayed_df(df, source, target, cross_map_lag, embed_dim, delay=1):
    '''
    Generate a delayed dataframe of `embed_dim` dimensions, each phase-shifted by `delay` delay.

    Input:
        df:             Pandas dataframe
        source:         Column of source variable (str)
        target:         Column of target variable (str)
        cross_map_lag:  Lags to shift sourceCol by (positive value: `source` maps to past `target`)
        embed_dim:      Embedding dimensions (int)
        delay:          Number of samples between each time series point (int)

    Returns:
        A multiLevel dataframe.
    '''
    # Obtain rows of dataframe
    N = len(df)

    assert embed_dim > 1
    assert delay >= 1

    # Define empty dataframe
    output = pd.DataFrame()

    # Define empty list for column headers
    column_headers = []

    # Create `embed_dim` duplicates of each time-series. 
    # Shift each duplicated time-series by 'delay' sampling intervals
    for field, series in df[[source, target]].iteritems():
        for i in range(embed_dim):
            output = pd.concat([output, series.shift(- i * delay - (cross_map_lag if field == source else 0))], 
                               axis=1)

            # Create multiLevel header
            column_headers.append((field, str(i)))

    # Add column headers to dataframe
    output.columns = pd.MultiIndex.from_tuples(column_headers)

    # Removes rows containing NaN 
    # Reset index to start from 0
    processed_output = (output
                        .dropna()
                        .reset_index(drop=True))

    return processed_output