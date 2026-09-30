import numpy as np
from talos.backends import backend_for
from talos.utils.best_model import best_model, activate_model


def classes(predictions, task):
    values = np.asarray(predictions)
    if task == 'binary':
        if values.ndim > 1 and values.shape[-1] == 2:
            return np.argmax(values, axis=-1)
        return (values >= .5).astype(int)
    if task in ('multi_class', 'multiclass', 'multi_label'):
        return np.argmax(values, axis=-1) if values.ndim > 1 else values.astype(int)
    if task in ('multilabel', 'multi_label_independent'):
        return (values >= .5).astype(int)
    if task in ('continuous', 'regression'):
        return values
    raise ValueError('Unknown prediction task: ' + str(task))


class Predict:
    def __init__(self, scan_object):
        self.scan_object = scan_object
        self.data = scan_object.data

    def predict(self, x, metric, asc, model_id=None, saved=False, custom_objects=None, model_factory=None, **kwargs):
        if model_id is None:
            model_id = best_model(self.scan_object, metric, asc)
        model = activate_model(self.scan_object, model_id, saved, custom_objects, model_factory)
        return backend_for(model).predict(model, x, **kwargs)

    def predict_classes(self, x, metric, asc, task, model_id=None, saved=False, custom_objects=None, model_factory=None):
        return classes(self.predict(x, metric, asc, model_id, saved, custom_objects, model_factory), task)
