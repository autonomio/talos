def metric_names():

    '''These are used as a shorthand for filtering out columns
    that should not be included as depedent variables for optimizing.'''

    return ['round_epochs', 'val_loss', 'val_acc', 'val_fmeasure_acc',
            'val_recall', 'val_precision', 'val_matthews_correlation',
            'loss', 'acc', 'fmeasure_acc', 'recall', 'precision',
            'matthews_correlation']
