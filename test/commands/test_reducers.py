def test_reducers():

    print("\n >>> start reducers...")

    import talos

    x, y = talos.templates.datasets.iris()
    p = talos.templates.params.iris()
    model = talos.templates.models.iris

    x = x[:50]
    y = y[:50]

    for strategy in ['trees',
                     'forrest',
                     'correlation',
                     'gamify',
                     'local_strategy']:

        talos.Scan(x=x,
                   y=y,
                   params=p,
                   model=model,
                   experiment_name='test_iris',
                   round_limit=2,
                   reduction_method=strategy,
                   reduction_interval=1)

    print('finised reducers \n')

    # # # # # # # # # # # # # # # # # #
