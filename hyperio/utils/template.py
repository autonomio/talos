def value_cols():

    _vc_out = ['round_epochs', 'train_peak', 'val_peak',
               'train_acc', 'val_acc',
               'train_loss', 'val_loss',
               'train_score', 'val_score']

    return _vc_out


def rep_cols():

    _rp_out = ['first_neuron',
               'batch_size',
               'round_epochs',
               'lr',
               'optimizer',
               'activation',
               'last_activation',
               'loss']
    return _rp_out


def df_cols():

    '''This goes to both main.py and report.py'''

    _dc_out = ['train_peak',
               'val_peak',
               'train_acc',
               'val_acc',
               'train_loss',
               'val_loss',
               'train_score',
               'val_score',
               'round_epochs',
               'batch_size',
               'epochs',
               'dropout',
               'hidden_layers',
               'normalized_lr',
               'first_neuron',
               'loss',
               'optimizer',
               'activation',
               'last_activation',
               'embed_dims',
               'weight_regul']

    return _dc_out


def old_col_fix():

    '''This is to deal with the old df format'''

    return ['activation', 'hidden_layers']
