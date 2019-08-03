def test_analyze(scan_object):

    import talos

    '''Tests all the attributes available in the Reporting() object'''

    print('\n >>> Start Analyze()... \n')

    # for now test with old name
    r = talos.Reporting(scan_object)

    # read from file
    r = talos.Reporting('test.csv')

    # and then from scan object
    r = talos.Analyze(scan_object)

    # test the object properties
    r.best_params('val_loss', ['val_acc'])
    r.correlate('val_loss', ['val_acc'])
    r.data
    r.high('val_acc')
    r.low('val_acc')

    r.plot_bars('first_neuron', 'val_acc', 'dropout', 'hidden_layers')
    r.plot_box('first_neuron', 'val_acc')
    r.plot_corr('val_loss', ['val_acc'])
    r.plot_hist('val_acc')
    r.plot_kde('val_acc')
    r.plot_line('val_acc')
    r.plot_regs('val_acc', 'val_loss')
    r.rounds()
    r.rounds2high('val_acc')
    r.table('val_loss', ['val_acc'])

    print('finish Analyze() \n')
