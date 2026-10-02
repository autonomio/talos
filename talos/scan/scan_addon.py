"""Retain historical model evaluation and best-model activation exports."""

from talos.commands.evaluate import evaluate_models as func_evaluate
from talos.utils.best_model import activate_model, best_model

__all__ = ['activate_model', 'best_model', 'func_best_model', 'func_evaluate']


def func_best_model(scan_object, metric='val_acc', asc=False, saved=False, custom_objects=None):
    return activate_model(scan_object, best_model(scan_object, metric, asc), saved, custom_objects)
