from typing import Any
from typing import Protocol
from typing import cast

import numpy as np
import numpy.typing as npt
import sklearn.metrics


class _SkMetricsModule(Protocol):

    '''Typed facade over the sklearn.metrics surface used for one-vs-rest AUC.'''

    def roc_auc_score(self, y_true: Any, y_score: Any) -> Any: ...


def _sk_metrics() -> _SkMetricsModule:

    '''Return sklearn.metrics behind the typed facade.'''

    return sklearn.metrics


def _probability_column_index(label: object, present: npt.NDArray[Any], n_columns: int) -> int | None:
    if n_columns == len(present):
        return int(np.where(present == label)[0][0])
    if isinstance(label, (np.integer, int)) and not isinstance(label, (np.bool_, bool)):
        index = int(cast('int | np.integer[Any]', label))
        if 0 <= index < n_columns:
            return index
    return None


def safe_ovr_auc(y_true: npt.NDArray[Any] | list[int] | list[str],
                 probs: npt.NDArray[np.floating[Any]] | list[list[float]]) -> float:

    '''
    Compute one-vs-rest AUC safely handling missing classes.

    Args:
        y_true (np.ndarray | list[int] | list[str]): True class labels, shape (n_samples,)
        probs (np.ndarray | list[list[float]]): Predicted probabilities, shape (n_samples, n_classes)

    Returns:
        float: Mean AUC across all valid class comparisons, or NaN if no valid AUC calculations can be made
    '''

    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    present = np.unique(y_true)
    aucs: list[float] = []
    for c in present:
        pos = (y_true == c)
        neg = ~pos
        column_index = _probability_column_index(c, present, probs.shape[1])
        if pos.any() and neg.any() and column_index is not None:
            aucs.append(cast(float, _sk_metrics().roc_auc_score(pos, probs[:, column_index])))
    return float('nan') if not aucs else float(np.mean(aucs))
