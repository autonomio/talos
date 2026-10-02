"""Compute multiclass classification metrics from caller predictions."""

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import sklearn.metrics

from talos.metrics.safe_ovr_auc import safe_ovr_auc


class _SkMetricsModule(Protocol):

    '''Typed facade over the sklearn.metrics surface used for multiclass metrics.'''

    def accuracy_score(self, y_true: Any, y_pred: Any) -> float: ...

    def precision_score(self, y_true: Any, y_pred: Any, *, average: str) -> float: ...

    def recall_score(self, y_true: Any, y_pred: Any, *, average: str) -> float: ...


def _sk_metrics() -> _SkMetricsModule:

    '''Return sklearn.metrics behind the typed facade.'''

    return sklearn.metrics


def multiclass_metrics(data: dict[str, Any],
                       preds: list[int] | npt.NDArray[np.integer[Any]],
                       probs: list[list[float]] | npt.NDArray[np.floating[Any]],
                       average: str = 'macro') -> dict[str, Any]:

    '''
    Compute multiclass classification metrics from predictions and probabilities.

    Args:
        data (dict): Data dictionary with 'y_test' key containing true class labels
        preds (list): Predicted class labels
        probs (list): Predicted class probabilities
        average (str): Averaging strategy for precision and recall

    Returns:
        dict: Dictionary containing precision, recall, auc, and accuracy metrics
    '''

    round_results = {'precision': round(_sk_metrics().precision_score(data['y_test'], preds, average=average), 3),
                     'recall': round(_sk_metrics().recall_score(data['y_test'], preds, average=average), 3),
                     'auc': round(safe_ovr_auc(data['y_test'], np.asarray(probs)), 3),
                     'accuracy': round(_sk_metrics().accuracy_score(data['y_test'], preds), 3)}

    return round_results
