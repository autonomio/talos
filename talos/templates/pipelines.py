def _pipeline(name, round_limit, random_method, debug=False):
    import talos
    x, y = getattr(talos.templates.datasets, name)()
    params = getattr(talos.templates.params, name)(debug=debug) if name == 'titanic' else getattr(talos.templates.params, name)()
    return talos.Scan(x, y, params, getattr(talos.templates.models, name), 'test',
                      round_limit=round_limit, random_method=random_method)


def breast_cancer(round_limit=2, random_method='uniform_mersenne'):
    return _pipeline('breast_cancer', round_limit, random_method)


def cervical_cancer(round_limit=2, random_method='uniform_mersenne'):
    return _pipeline('cervical_cancer', round_limit, random_method)


def iris(round_limit=2, random_method='uniform_mersenne'):
    return _pipeline('iris', round_limit, random_method)


def titanic(round_limit=2, random_method='uniform_mersenne', debug=False):
    return _pipeline('titanic', round_limit, random_method, debug)
