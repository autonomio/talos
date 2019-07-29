import talos as ta


def test_reporting_object(scan_object):

    '''Tests all the attributes available in the Reporting() object'''

    print('Start testing Reporting object...')

    r = ta.Reporting(scan_object)
    r.best_params('val_acc')
    r.correlate('val_acc')
    r.data
    r.high('val_acc')
    r.low('val_acc')

    print(r.data)
    print(r.data.dtypes)

    r.plot_bars('first_neuron', 'val_acc', 'batch_size', 'hidden_layers')
    r.plot_box('first_neuron', 'val_acc')
    r.plot_corr('val_loss')
    r.plot_hist('val_acc')
    r.plot_kde('val_acc')
    r.plot_line('val_acc')
    r.plot_regs('val_acc', 'val_loss')
    r.rounds()
    r.rounds2high('val_acc')
    r.table('val_acc')

    return "Finished testing Reporting object!"
