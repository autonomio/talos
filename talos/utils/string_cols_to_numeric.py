def isnumber(value):

    '''Checks if a string can be converted into
    a float (or int as a by product). Helper function
    for string_cols_to_numeric'''

    try:
        float(value)
        return True
    except ValueError:
        return False


def string_cols_to_numeric(data, destructive=False):

    '''Takes in a dataframe and attempts to convert numeric columns
    into floats or ints respectively.'''

    if destructive is False:
        data = data.copy(deep=True)

    for col in data.columns:

        if data[col].apply(isnumber).sum() == len(data):
            try:
                data[col] = data[col].astype(int)
            except:  # intentionally silent
                try:
                    data[col] = data[col].astype(float)
                except:  # intentionally silent
                    data[col] = data[col]
        else:
            data[col] = data[col]

    return data
