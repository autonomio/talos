def test_templates():

    print("\n >>> start templates ...")

    import talos

    x, y = talos.templates.datasets.titanic()
    x = x[:50]
    y = y[:50]
    model = talos.templates.models.titanic
    p = talos.templates.params.titanic()
    talos.Scan(x, y, p, model, 'test', round_limit=2)

    x, y = talos.templates.datasets.iris()
    x = x[:50]
    y = y[:50]
    model = talos.templates.models.iris
    p = talos.templates.params.iris()
    talos.Scan(x, y, p, model, 'test', round_limit=2)

    x, y = talos.templates.datasets.cervical_cancer()
    x = x[:50]
    y = y[:50]
    model = talos.templates.models.cervical_cancer
    p = talos.templates.params.cervical_cancer()
    talos.Scan(x, y, p, model, 'test', round_limit=2)

    x, y = talos.templates.datasets.breast_cancer()
    x = x[:50]
    y = y[:50]
    model = talos.templates.models.breast_cancer
    p = talos.templates.params.breast_cancer()
    talos.Scan(x, y, p, model, 'test', round_limit=2)

    x, y = talos.templates.datasets.icu_mortality(50)
    x, y = talos.templates.datasets.telco_churn(.3)
    x, y, x1, y1 = talos.templates.datasets.mnist()
    x, y = talos.templates.datasets.breast_cancer()
    x, y = talos.templates.datasets.cervical_cancer()
    x, y = talos.templates.datasets.titanic()

    talos.templates.pipelines.breast_cancer(random_method='quantum')
    talos.templates.pipelines.cervical_cancer(random_method='sobol')
    talos.templates.pipelines.iris(random_method='uniform_crypto')
    talos.templates.pipelines.titanic(random_method='korobov_matrix')

    print("finish templates \n")
