def create_param_space(data, no_of_metrics=2):

    '''Takes as input experiment log dataframe and returns
    ParamSpace object

    data | DataFrame | Talos experiment log as pandas dataframe
    no_of_metrics | int | number of metrics in the dataframe

    '''

    from talos.parameters.ParamSpace import ParamSpace

    params = {}

    for col in data.iloc[:, no_of_metrics:].columns:
        params[col] = data[col].unique().tolist()

    param_keys = sorted(list(params.keys()))

    return ParamSpace(params, param_keys)
