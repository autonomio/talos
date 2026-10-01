from talos.commands.evaluate import evaluate_models as func_evaluate
from talos.utils.best_model import best_model, activate_model


def func_best_model(scan_object, metric='val_acc', asc=False, saved=False, custom_objects=None):
    return activate_model(scan_object, best_model(scan_object, metric, asc), saved, custom_objects)
