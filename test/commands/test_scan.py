def test_scan():

    print("\n >>> start Scan()...")

    import talos

    from keras.losses import binary_crossentropy
    from keras.optimizers import Adam
    from keras.activations import relu, elu
    from keras.layers import Dense
    from keras.models import Sequential

    p = {'activation': [relu, elu],
         'optimizer': ['Nadam', Adam],
         'losses': ['logcosh', binary_crossentropy],
         'shapes': ['brick', 'funnel', 'triangle'],
         'first_neuron': [16],
         'hidden_layers': ([0, 1, 2, 3]),
         'dropout': (.05, .35, .1),
         'epochs': [50]}

    def iris_model(x_train, y_train, x_val, y_val, params):

        model = Sequential()
        model.add(Dense(params['first_neuron'],
                        input_dim=4,
                        activation=params['activation']))

        talos.utils.hidden_layers(model, params, 3)

        model.add(Dense(3, activation='softmax'))

        if isinstance(params['optimizer'], str):
            opt = params['optimizer']
        else:
            opt = params['optimizer']()

        model.compile(optimizer=opt,
                      loss=params['losses'],
                      metrics=['acc', talos.utils.metrics.f1score])

        out = model.fit(x_train, y_train,
                        batch_size=25,
                        epochs=params['epochs'],
                        validation_data=[x_val, y_val],
                        verbose=0)

        return out, model

    x, y = talos.templates.datasets.iris()

    scan_object = talos.Scan(x=x,
                             y=y,
                             params=p,
                             model=iris_model,
                             experiment_name='testingq',
                             val_split=0.3,
                             random_method='uniform_mersenne',
                             round_limit=15,
                             reduction_method='spearman',
                             reduction_interval=10,
                             reduction_window=9,
                             reduction_threshold=0.01,
                             reduction_metric='val_acc',
                             minimize_loss=False,
                             boolean_limit=lambda p: p['first_neuron'] * p['hidden_layers'] < 220
                             )

    x = x[:50]
    y = y[:50]

    p['epochs'] = [5]

    # minimal settings
    talos.Scan(x=x,
               y=y,
               x_val=x,
               y_val=y,
               params=p,
               model=iris_model,
               experiment_name='iris_test',
               fraction_limit=.05)

    # config invoked
    talos.Scan(x=x,
               y=y,
               params=p,
               model=iris_model,
               experiment_name="testing2",
               x_val=x,
               y_val=y,
               random_method='latin_suduko',
               seed=3,
               performance_target=['acc', 0.01, False],
               round_limit=3,
               disable_progress_bar=True,
               print_params=True,
               clear_session=False)

    talos.Scan(x=x,
               y=y,
               params=p,
               model=iris_model,
               experiment_name="testing3",
               x_val=None,
               y_val=None,
               val_split=0.3,
               random_method='sobol',
               seed=5,
               performance_target=['val_acc', 0.1, False],
               fraction_limit=None,
               time_limit="2099-09-09 09:09",
               boolean_limit=None,
               reduction_method='spearman',
               reduction_interval=2,
               reduction_window=2,
               reduction_threshold=0.2,
               reduction_metric='loss',
               minimize_loss=True,
               clear_session=False)

    print('finised Scan() \n')

    # # # # # # # # # # # # # # # # # #

    print("\n >>> start Scan() object ...")

    # the create the test based on it

    _keras_model = scan_object.best_model()
    _keras_model = scan_object.best_model('loss', True)

    scan_object.evaluate_models(x_val=scan_object.x,
                                y_val=scan_object.y,
                                task='multi_label')

    scan_object.evaluate_models(x_val=scan_object.x,
                                y_val=scan_object.y,
                                task='multi_label',
                                n_models=3,
                                metric='val_loss',
                                folds=3,
                                shuffle=False,
                                asc=True)

    print('finised Scan() object \n')

    # # # # # # # # # # # # # # # # # #


    return scan_object
