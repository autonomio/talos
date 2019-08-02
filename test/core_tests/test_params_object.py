import talos as ta


def test_params_object():

    '''Tests the object from Params()'''

    print('Start testing Params object...')

    p = ta.autom8.AutoParams()

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

    p = ta.autom8.AutoParams(replace=False)

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

    return "Finished testing Params object!"
