def test_rest(scan_object):

    print('\n >>> start testing the rest... \n')

    import talos
    import random

    name = str(hash(random.random()))

    talos.Deploy(scan_object, name, 'val_acc')
    talos.Restore(name + '.zip')

    x, y = talos.templates.datasets.breast_cancer()
    x = x[:50]
    y = y[:50]

    callbacks = [talos.utils.live(),
                 talos.utils.early_stopper(10),
                 talos.utils.ExperimentLogCallback('test', {})]

    metrics = [talos.utils.metrics.f1score,
               talos.utils.metrics.fbeta,
               talos.utils.metrics.mae,
               talos.utils.metrics.mape,
               talos.utils.metrics.matthews,
               talos.utils.metrics.mse,
               talos.utils.metrics.msle,
               talos.utils.metrics.precision,
               talos.utils.metrics.recall,
               talos.utils.metrics.rmae,
               talos.utils.metrics.rmse,
               talos.utils.metrics.rmsle]

    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile('adam', 'logcosh', metrics=metrics)

    talos.utils.gpu_utils.force_cpu()

    model.fit(x, y, callbacks=callbacks)

    model.fit_generator(talos.utils.generator(x, y, 10), 5)

    model.fit_generator(talos.utils.SequenceGenerator(x, y, 10))

    print('finised testing the rest \n')
