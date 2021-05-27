def recover_best_model():

    import os

    from talos.utils import recover_best_model
    import talos
    import tensorflow

    experiment_log = 'test_q/' + os.listdir('test_q')[0]   

    x, y = talos.templates.datasets.iris()
    input_model = talos.templates.models.iris

    x = x[:50]
    y = y[:50]

    # define the input model
    def iris_model(x_train, y_train, x_val, y_val, params):

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

        model = Sequential()
        model.add(Dense(params['first_neuron'], input_dim=4, activation=params['activation']))

        talos.utils.hidden_layers(model, params, 3)

        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer=params['optimizer'], loss=params['losses'], metrics=['acc'])

        out = model.fit(x_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        validation_data=(x_val, y_val),
                        verbose=0)

        return out, model

    recover_best_model(x_train=x,
                       y_train=y,
                       x_val=x,
                       y_val=y,
                       experiment_log=experiment_log,
                       input_model=iris_model,
                       metric='acc',
                       x_cross=x,
                       y_cross=y,
                       n_models=5,
                       task='multi_label')
