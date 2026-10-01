from typing import Any
from typing import Protocol
from typing import cast

import numpy as np
import numpy.typing as npt
import sklearn.metrics

_BINARY_CLASS_COUNT = 2


class _SkMetricsModule(Protocol):

    '''Typed facade over the sklearn.metrics surface used for binary metrics.'''

    def roc_auc_score(self, y_true: Any, y_score: Any) -> Any: ...

    def accuracy_score(self, y_true: Any, y_pred: Any) -> float: ...

    def precision_score(self, y_true: Any, y_pred: Any, *, zero_division: Any) -> float: ...

    def recall_score(self, y_true: Any, y_pred: Any, *, zero_division: Any) -> float: ...

    def confusion_matrix(self, y_true: Any, y_pred: Any, *, labels: list[int]) -> npt.NDArray[np.integer[Any]]: ...


def _sk_metrics() -> _SkMetricsModule:

    '''Return sklearn.metrics behind the typed facade.'''

    return sklearn.metrics


def binary_metrics(data: dict[str, Any],
                   preds: list[int] | npt.NDArray[np.integer[Any]],
                   probs: list[float] | npt.NDArray[np.floating[Any]]) -> dict[str, Any]:

    '''
    Compute binary classification metrics from predictions and probabilities.

    Degenerate one-class outcomes never crash: fpr is NaN when y_test has no
    negatives, and auc is NaN when y_test contains a single class, mirroring
    the safe_ovr_auc convention.

    Args:
        data (dict): Data dictionary with 'y_test' key containing true binary labels
        preds (list): Predicted binary class labels
        probs (list): Predicted class probabilities

    Returns:
        dict: Dictionary containing recall, precision, fpr, auc, and accuracy metrics
    '''

    y_test = np.asarray(data['y_test'])
    negatives = int((y_test == 0).sum())
    false_positives = int(_sk_metrics().confusion_matrix(y_test, preds, labels=[0, 1])[0, 1])
    fpr = float('nan') if negatives == 0 else round(false_positives / negatives, 3)
    auc = (float('nan') if np.unique(y_test).size < _BINARY_CLASS_COUNT
           else round(cast(float, _sk_metrics().roc_auc_score(y_test, probs)), 3))

    round_results = {'recall': round(_sk_metrics().recall_score(y_test, preds, zero_division=0), 3),
                     'precision': round(_sk_metrics().precision_score(y_test, preds, zero_division=0), 3),
                     'fpr': fpr,
                     'auc': auc,
                     'accuracy': round(_sk_metrics().accuracy_score(y_test, preds), 3)}

    return round_results
