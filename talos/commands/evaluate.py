import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
from talos.backends import backend_for
from talos.utils.best_model import best_model, activate_model
from talos.utils.validation_split import kfold
from .predict import classes


def score(y, predictions, task, average=None):
    truth = np.asarray(y)
    if task in ('continuous', 'regression'):
        return mean_absolute_error(truth, predictions)
    if task == 'multi_label' and truth.ndim > 1:
        if np.all(np.sum(truth, axis=-1) == 1):
            truth = np.argmax(truth, axis=-1)
            predictions = classes(predictions, 'multi_class')
            average = average or 'macro'
        else:
            predictions = classes(predictions, 'multilabel')
            average = average or 'macro'
    elif task in ('multi_class', 'multiclass'):
        if truth.ndim > 1 and truth.shape[-1] > 1:
            truth = np.argmax(truth, axis=-1)
        truth = truth.reshape(-1)
        predictions = classes(predictions, 'multi_class')
        average = average or 'macro'
    elif task in ('multilabel', 'multi_label_independent'):
        predictions = classes(predictions, task)
        average = average or 'macro'
    elif task == 'binary':
        truth = truth.reshape(-1)
        predictions = classes(predictions, task).reshape(-1)
        average = average or 'binary'
    else:
        raise ValueError('Unknown evaluation task: ' + str(task))
    return f1_score(truth, predictions, average=average, zero_division=0)


class Evaluate:
    def __init__(self, scan_object):
        self.scan_object = scan_object
        self.data = scan_object.data

    def evaluate(self, x, y, task, metric, model_id=None, folds=5, shuffle=True,
                 asc=False, saved=False, custom_objects=None, multi_input=False,
                 print_out=False, average=None, model_factory=None, seed=None):
        if model_id is None:
            model_id = best_model(self.scan_object, metric, asc)
        model = activate_model(self.scan_object, model_id, saved, custom_objects, model_factory)
        adapter = backend_for(model)
        kx, ky = kfold(x, y, folds, shuffle, multi_input, seed)
        out = []
        for features, labels in zip(kx, ky):
            predictions = adapter.predict(model, features)
            if isinstance(labels, (list, dict)):
                label_values = list(labels.values()) if isinstance(labels, dict) else labels
                pred_values = list(predictions.values()) if isinstance(predictions, dict) else predictions
                result = np.mean([score(a, b, task, average) for a, b in zip(label_values, pred_values)])
            else:
                result = score(labels, predictions, task, average)
            out.append(float(result))
        if print_out:
            print('mean : %.2f \n std : %.2f' % (np.mean(out), np.std(out)))
        return out


def evaluate_models(scan_object, x_val, y_val, task, n_models=10, metric='val_acc',
                    folds=5, shuffle=True, asc=False, saved=False, custom_objects=None,
                    average=None, model_factory=None, multi_input=None, seed=None):
    if multi_input is None:
        multi_input = isinstance(x_val, list)
    picks = scan_object.data.sort_values(metric, ascending=asc, kind='stable').index[:n_models]
    heading = 'eval_mae' if task in ('continuous', 'regression') else 'eval_f1score'
    scan_object.data[heading + '_mean'] = np.nan
    scan_object.data[heading + '_std'] = np.nan
    evaluator = Evaluate(scan_object)
    for model_id in picks:
        values = evaluator.evaluate(x_val, y_val, task, metric, model_id=model_id,
                                    folds=folds, shuffle=shuffle, asc=asc, saved=saved,
                                    custom_objects=custom_objects, multi_input=multi_input,
                                    average=average, model_factory=model_factory, seed=seed)
        scan_object.data.loc[model_id, heading + '_mean'] = np.mean(values)
        scan_object.data.loc[model_id, heading + '_std'] = np.std(values)
