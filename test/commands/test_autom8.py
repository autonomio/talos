def test_autom8():

    import talos
    import wrangle

    from keras.optimizers import Adam

    print('\n >>> start AutoParams()... \n')

    p = talos.autom8.AutoParams()
    p.params
    p = talos.autom8.AutoParams(p.params)
    p.resample_params(5)

    p.activations(['relu'])
    p.batch_size(20, 50, 2)
    p.dropout(0, 0.22, 0.04)
    p.epochs(5, 10, 1)
    p.kernel_initializers(['zeros'])
    p.last_activations(['softmax'])
    p.layers(0, 2, 1)
    p.losses([talos.utils.metrics.f1score])
    p.lr([0.01])
    p.networks(['dense'])
    p.neurons(1, 5, 1)
    p.optimizers([Adam])
    p.shapes(['brick'])
    p.shapes_slope(0, .2, .01)

    p.resample_params(1)

    print('finised AutoParams() \n')

    # # # # # # # # # # # # # # # #

    print('\n >>> start AutoModel(), AutoScan() and AutoPredict()... \n')

    x, y = wrangle.utils.create_synth_data('binary', 50, 10, 1)
    p.losses(['binary_crossentropy'])
    auto = talos.autom8.AutoScan('binary', 'testinga', 1)
    scan_object = auto.start(x, y, params=p.params)
    talos.autom8.AutoPredict(scan_object, x, y, x, 'binary')

    x, y = wrangle.utils.create_synth_data('multi_label', 50, 10, 4)
    p.losses(['categorical_crossentropy'])
    auto = talos.autom8.AutoScan('multi_label', 'testingb', 1)
    auto.start(x, y, params=p.params)
    talos.autom8.AutoPredict(scan_object, x, y, x, 'multi_label')

    x, y = wrangle.utils.create_synth_data('multi_class', 50, 10, 3)
    p.losses(['sparse_categorical_crossentropy'])
    auto = talos.autom8.AutoScan('multi_class', 'testingc', 1)
    auto.start(x, y, params=p.params)
    talos.autom8.AutoPredict(scan_object, x, y, x, 'multi_class')

    x, y = wrangle.utils.create_synth_data('continuous', 50, 10, 1)
    p.losses(['mae'])
    auto = talos.autom8.AutoScan('continuous', 'testingd', 1)
    auto.start(x, y, params=p.params)
    talos.autom8.AutoPredict(scan_object, x, y, x, 'continuous')

    print('finised AutoModel(), AutoScan() and AutoPredict() \n')

    # # # # # # # # # # # # # # # #
