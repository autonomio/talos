def metric_names():

    '''These are used as a shorthand for filtering out columns
    that should not be included as depedent variables for optimizing.'''

    return ['round_epochs',
            'loss',
            'val_loss',
            'acc',
            'val_acc',
            'fmeasure_acc',
            'val_fmeasure_acc',
            'recall_acc',
            'val_recall_acc',
            'precision_acc',
            'val_precision_acc',
            'matthews_correlation_acc',
            'val_matthews_correlation_acc',
            'val_root_mean_squared_error',
            'root_mean_squared_error',
            'val_mean_squared_error',
            'mean_squared_error',
            'val_mean_average_error',
            'mean_average_error',
            ]
