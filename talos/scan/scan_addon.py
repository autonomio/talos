def func_best_model(scan_object,
                    metric='val_acc',
                    asc=False,
                    saved=False,
                    custom_objects=None):

    '''Picks the best model based on a given metric and
    returns the index number for the model.

    scan_object | object | The object that is returned from Scan()
                           upon completion.
    metric | str | The metric to be used for picking the best model.
    asc | bool | Make this True for metrics that are to be minimized
                 (e.g. loss), and False when the metric is to be
                 maximized (e.g. acc).
    saved | bool | if a model saved on local machine should be used.
    custom_objects | dict | if the model has a custom object, pass it here.

    NOTE: for loss 'asc' should be True'''

    import warnings as warnings

    warnings.simplefilter('ignore')

    from ..utils.best_model import best_model, activate_model
    model_no = best_model(scan_object, metric, asc)
    out = activate_model(scan_object, model_no, saved, custom_objects)

    return out


def func_evaluate(scan_object,
                  x_val,
                  y_val,
                  task,
                  n_models=10,
                  metric='val_acc',
                  folds=5,
                  shuffle=True,
                  asc=False,
                  saved=False,
                  custom_objects=None):

    '''K-fold Cross Evaluator

    For creating scores from kfold cross-evaluation and
    adding them to the data frame.

    scan_object | python class | The class object returned by Scan() upon
                                 completion of the experiment.
    x_val | array or list of arrays | Input data (features) in the same format
                                      as used in Scan(), but should not be the
                                      same data (or it will not be much of
                                      validation).
    y_val | array or list of arrays | Input data (labels) in the same format
                                      as used in.
                                      Scan(), but should not be the same data
                                      (or it will not be much of validation).
    task | string | 'binary', 'multi_class', 'multi_label', or 'continuous'.
    n_models | int | The number of models to be evaluated. If set to 10,
                     then 10 models with the highest metric value are
                     evaluated. See below.
    metric | str | The metric to be used for picking the models to be
                   evaluated.
    folds | int | The number of folds to be used in the evaluation.
    shuffle | bool | If the data is to be shuffled or not. Set always
                     to False for timeseries but keep in mind that you
                     might get periodical/seasonal bias.
    asc | bool | Set to True if the metric is to be minimized.
    saved | bool | if a model saved on local machine should be used
    custom_objects | dict | if the model has a custom object, pass it here

    '''
    import warnings as warnings
    from tqdm import tqdm
    import numpy as np

    warnings.simplefilter('ignore')

    picks = scan_object.data.sort_values(metric,
                                         ascending=asc).index.values[:n_models]

    if n_models > len(scan_object.data):
        data_len = len(scan_object.data)
    else:
        data_len = n_models

    out = []

    pbar = tqdm(total=data_len)

    from ..commands.evaluate import Evaluate

    for i in range(len(scan_object.data)):

        if i in list(picks):

            evaluate_object = Evaluate(scan_object)
            temp = evaluate_object.evaluate(x_val,
                                            y_val,
                                            task=task,
                                            model_id=i,
                                            metric=metric,
                                            folds=folds,
                                            shuffle=shuffle,
                                            asc=asc,
                                            saved=saved,
                                            custom_objects=custom_objects)

            out.append([np.mean(temp), np.std(temp)])
            pbar.update(1)

        else:
            out.append([np.nan, np.nan])

    pbar.close()

    if task == 'continuous':
        heading = 'eval_' + 'mae'
    else:
        heading = 'eval_' + 'f1score'

    scan_object.data[heading + '_mean'] = [i[0] for i in out]
    scan_object.data[heading + '_std'] = [i[1] for i in out]

    print(">> Added evaluation score columns to scan_object.data")
