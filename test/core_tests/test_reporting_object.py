import talos as ta


def test_reporting_object(scan_object):

    '''Tests all the attributes available in the Reporting() object'''

    print('Start testing Reporting object...')

    r = ta.Reporting(scan_object)
    r.best_params()
    r.correlate()
    r.data
    r.high()
    r.low()

    print(r.data)
    print(r.data.dtypes)

    r.plot_bars('first_neuron', 'val_acc', 'batch_size', 'hidden_layers')
    r.plot_box('first_neuron')
    r.plot_corr('val_loss')
    r.plot_hist()
    r.plot_kde('val_acc')
    r.plot_line()
    r.plot_regs()
    r.rounds()
    r.rounds2high()
    r.table()

    return "Finished testing Reporting object!"
