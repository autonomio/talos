def test_templates():

    import talos as ta

    x, y = ta.templates.datasets.titanic()
    x = x[:50]
    y = y[:50]
    model = ta.templates.models.titanic
    p = ta.templates.params.titanic()
    ta.Scan(x, y, p, model, round_limit=2)

    x, y = ta.templates.datasets.iris()
    x = x[:50]
    y = y[:50]
    model = ta.templates.models.iris
    p = ta.templates.params.iris()
    ta.Scan(x, y, p, model, round_limit=2)

    x, y = ta.templates.datasets.cervical_cancer()
    x = x[:50]
    y = y[:50]
    model = ta.templates.models.cervical_cancer
    p = ta.templates.params.cervical_cancer()
    ta.Scan(x, y, p, model, round_limit=2)

    x, y = ta.templates.datasets.breast_cancer()
    x = x[:50]
    y = y[:50]
    model = ta.templates.models.breast_cancer
    p = ta.templates.params.breast_cancer()
    ta.Scan(x, y, p, model, round_limit=2)

    x, y = ta.templates.datasets.icu_mortality(50)

    ta.templates.pipelines.breast_cancer(random_method='quantum')
    ta.templates.pipelines.cervical_cancer(random_method='sobol')
    ta.templates.pipelines.iris(random_method='uniform_crypto')
    ta.templates.pipelines.titanic(random_method='korobov_matrix')
