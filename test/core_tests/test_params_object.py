import talos as ta


def test_params_object():

    '''Tests the object from Params()'''

    print('Start testing Params object...')

    p = ta.Params()

    # without arguments

    p.activations()
    p.batch_size()
    p.dropout()
    p.epochs()
    p.kernel_initializers()
    p.layers()
    p.neurons()
    p.lr()
    p.optimizers()
    p.shapes()
    p.shapes_slope()
    p.automated()

    p = ta.Params(replace=False)

    # with arguments
    p.activations()
    p.batch_size(10, 100, 5)
    p.dropout()
    p.epochs(10, 100, 5)
    p.kernel_initializers()
    p.layers(12)
    p.neurons(10, 100, 5)
    p.lr()
    p.optimizers('multi_label')
    p.shapes()
    p.shapes_slope()
    p.automated('sloped')

    return "Finished testing Params object!"
