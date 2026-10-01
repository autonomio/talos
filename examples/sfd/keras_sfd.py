"""Standalone Keras template; set KERAS_BACKEND before importing Keras."""
backend = 'keras'

def params():
    return {'neurons': [8, 16], 'learning_rate': [.01, .03], 'epochs': [3], 'batch_size': [16]}


def prep(data, round_params):
    # The caller supplies data. A CLI entry point can use a user-owned loader here.
    if data is None:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        x, y = load_iris(return_X_y=True)
        x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=.2, random_state=17)
        return {'x_train': x_train, 'y_train': y_train, 'x_val': x_val, 'y_val': y_val}
    return data


def model(data, round_params):
    import keras
    network = keras.Sequential([keras.layers.Input(shape=(4,)),
                                keras.layers.Dense(round_params['neurons'], activation='relu'),
                                keras.layers.Dense(3, activation='softmax')])
    network.compile(optimizer=keras.optimizers.Adam(learning_rate=round_params['learning_rate']),
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = network.fit(data['x_train'], data['y_train'], validation_data=(data['x_val'], data['y_val']),
                          epochs=round_params['epochs'], batch_size=round_params['batch_size'], verbose=0)
    return {'_model': network, '_history': history, 'backend': 'keras'}
