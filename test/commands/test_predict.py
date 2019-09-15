def test_predict():

    print("\n >>> start Predict()...")

    import sys
    sys.path.insert(0, '/Users/mikko/Documents/GitHub/talos')
    import talos

    x, y = talos.templates.datasets.iris()
    p = talos.templates.params.iris()
    model = talos.templates.models.iris

    x = x[:50]
    y = y[:50]

    scan_object = talos.Scan(x=x,
                             y=y,
                             params=p,
                             model=model,
                             experiment_name='test_iris', round_limit=2)

    predict = talos.Predict(scan_object)

    _preds = predict.predict(x, 'val_acc', False)
    _preds = predict.predict_classes(x, 'val_acc', False)

    print('finised Predict() \n')

    # # # # # # # # # # # # # # # # # #
