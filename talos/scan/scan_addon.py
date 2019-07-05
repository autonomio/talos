def func_best_model(scan_object, metric='val_acc', asc=False):

    '''Picks the best model based on a given metric and
    returns the index number for the model.

    NOTE: for loss 'asc' should be True'''

    import warnings as warnings

    warnings.simplefilter('ignore')

    from ..utils.best_model import best_model, activate_model
    model_no = best_model(scan_object, metric, asc)
    out = activate_model(scan_object, model_no)

    return out


def func_evaluate(scan_object,
                  x_val,
                  y_val,
                  n=10,
                  metric='val_acc',
                  folds=5,
                  shuffle=True,
                  average='binary',
                  asc=False):

    '''
    For creating scores from kfold cross-evaluation and
    adding them to the data frame.

    '''
    import warnings as warnings
    from tqdm import tqdm
    import numpy as np

    warnings.simplefilter('ignore')

    picks = scan_object.data.sort_values(metric,
                                         ascending=asc).index.values[:n]

    if n > len(scan_object.data):
        data_len = len(scan_object.data)
    else:
        data_len = n

    out = []

    pbar = tqdm(total=data_len)

    from ..commands.evaluate import Evaluate
    for i in range(len(scan_object.data)):
        if i in list(picks):
            evaluate_object = Evaluate(scan_object)
            temp = evaluate_object.evaluate(x_val, y_val,
                                            model_id=i,
                                            metric=metric,
                                            folds=folds,
                                            shuffle=shuffle,
                                            asc=asc)
            out.append([np.mean(temp), np.std(temp)])
            pbar.update(1)
        else:
            out.append([np.nan, np.nan])

    pbar.close()

    scan_object.data['eval_f1score_mean'] = [i[0] for i in out]
    scan_object.data['eval_f1score_std'] = [i[1] for i in out]
