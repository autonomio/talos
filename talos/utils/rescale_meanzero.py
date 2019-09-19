def rescale_meanzero(x):

    '''Rescales an array to mean-zero.

    x | array | the dataset to be rescaled
    '''

    import wrangle
    import pandas as pd

    return wrangle.df_rescale_meanzero(pd.DataFrame(x)).values
