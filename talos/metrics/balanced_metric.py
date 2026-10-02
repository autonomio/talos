"""Weight class-sensitive model metrics for imbalanced outcomes."""

from typing import Any, Protocol

import numpy as np
import sklearn.metrics


class _SkMetricsModule(Protocol):

    '''Typed facade over the sklearn.metrics surface used for the balanced metric.'''

    def precision_score(self, y_true: Any, y_pred: Any, *, zero_division: Any) -> float: ...


def _sk_metrics() -> _SkMetricsModule:

    '''Return sklearn.metrics behind the typed facade.'''

    return sklearn.metrics


def balanced_metric(y_true: Any, y_pred: Any) -> float:

    '''
    Compute balanced precision metric that accounts for positive prediction rate.

    Calculates precision * sqrt(positive_rate) to balance signal quality
    with prediction frequency. Higher scores indicate better balance between
    accurate predictions and sufficient positive predictions.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels

    Returns:
        float: Balanced metric score (0.0 if no positive predictions)
    '''

    if np.sum(y_pred) == 0:
        return 0.0

    prec = _sk_metrics().precision_score(y_true, y_pred, zero_division=0)
    positive_rate = np.sum(y_pred) / len(y_pred)

    return prec * np.sqrt(positive_rate)
