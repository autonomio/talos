def test_latest():

    print('\n >>> start Latest Features... \n')

    import talos
    from keras.models import Sequential
    from keras.layers import Dense

    x, y = talos.templates.datasets.iris()

    p = {'activation': ['relu', 'elu'],
         'optimizer': ['Nadam', 'Adam'],
         'losses': ['logcosh'],
         'shapes': ['brick'],
         'first_neuron': [16, 32, 64, 128],
         'hidden_layers': [0, 1, 2, 3],
         'dropout': [.2, .3, .4],
         'batch_size': [20, 30, 40, 50],
         'epochs': [10]}

    def iris_model(x_train, y_train, x_val, y_val, params):

        model = Sequential()
        model.add(Dense(params['first_neuron'],
                        input_dim=4,
                        activation=params['activation']))

        talos.utils.hidden_layers(model, params, 3)

        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer=params['optimizer'],
                      loss=params['losses'], metrics=['acc'])

        out = model.fit(x_train,
                        y_train,
                        callbacks=[talos.utils.ExperimentLogCallback('testing_latest', params)],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        validation_data=[x_val, y_val],
                        verbose=0)

        return out, model

    scan_object = talos.Scan(x, y,
                             model=iris_model,
                             params=p,
                             experiment_name='testing_latest',
                             round_limit=5,
                             reduction_method='gamify',
                             save_weights=False)

    print('finised Latest Features \n')
