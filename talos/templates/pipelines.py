def breast_cancer(round_limit=2, random_method='uniform_mersenne'):

    '''Performs a Scan with Iris dataset and simple dense net'''
    import talos as ta
    scan_object = ta.Scan(ta.templates.datasets.breast_cancer()[0],
                          ta.templates.datasets.breast_cancer()[1],
                          ta.templates.params.breast_cancer(),
                          ta.templates.models.breast_cancer,
                          'test',
                          round_limit=round_limit)

    return scan_object


def cervical_cancer(round_limit=2, random_method='uniform_mersenne'):

    '''Performs a Scan with Iris dataset and simple dense net'''
    import talos as ta
    scan_object = ta.Scan(ta.templates.datasets.cervical_cancer()[0],
                          ta.templates.datasets.cervical_cancer()[1],
                          ta.templates.params.cervical_cancer(),
                          ta.templates.models.cervical_cancer,
                          'test',
                          round_limit=round_limit)

    return scan_object


def iris(round_limit=2, random_method='uniform_mersenne'):

    '''Performs a Scan with Iris dataset and simple dense net'''
    import talos as ta
    scan_object = ta.Scan(ta.templates.datasets.iris()[0],
                          ta.templates.datasets.iris()[1],
                          ta.templates.params.iris(),
                          ta.templates.models.iris,
                          'test',
                          round_limit=round_limit)

    return scan_object


def titanic(round_limit=2, random_method='uniform_mersenne'):

    '''Performs a Scan with Iris dataset and simple dense net'''
    import talos as ta
    scan_object = ta.Scan(ta.templates.datasets.titanic()[0][:50],
                          ta.templates.datasets.titanic()[1][:50],
                          ta.templates.params.titanic(),
                          ta.templates.models.titanic,
                          'test',
                          random_method=random_method,
                          round_limit=round_limit)

    return scan_object
