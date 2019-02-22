from types import FunctionType
from inspect import isclass


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
        # avoid common non-numeric cases
        if data[col].dtype != 'O':
            continue
        if isclass(data[col][0]):
            continue
        if isinstance(data[col][0], FunctionType):
            continue
        if data[col].apply(isnumber).sum() == 0:
            continue

        # perform the conversion to int or float
        try:
            data[col] = data[col].astype(int)
        except ValueError:
            data[col] = data[col].astype(float)

    return data
