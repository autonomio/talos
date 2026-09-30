from typing import Any
from typing import Protocol

import numpy as np
import numpy.typing as npt
import sklearn.metrics


class _SkMetricsModule(Protocol):

    '''Typed facade over the sklearn.metrics surface used for regression metrics.'''

    def mean_absolute_error(self, y_true: Any, y_pred: Any) -> float: ...

    def root_mean_squared_error(self, y_true: Any, y_pred: Any) -> float: ...

    def mean_absolute_percentage_error(self, y_true: Any, y_pred: Any) -> float: ...

    def r2_score(self, y_true: Any, y_pred: Any) -> float: ...


def _sk_metrics() -> _SkMetricsModule:

    '''Return sklearn.metrics behind the typed facade.'''

    return sklearn.metrics


def continuous_metrics(data: dict[str, Any],
                       preds: list[float] | npt.NDArray[np.floating[Any]]) -> dict[str, Any]:

    '''
    Compute regression metrics from continuous predictions.

    NOTE: This function is experimental and may change in future versions.

    Args:
        data (dict): Data dictionary with 'y_test' key containing true continuous values
        preds (list): Predicted continuous values

    Returns:
        dict: Dictionary containing bias, mae, rmse, r2, and mape metrics
    '''

    y_test = np.asarray(data['y_test'])
    preds_array = np.asarray(preds)
    if preds_array.shape != y_test.shape and preds_array.size == y_test.size:
        preds_array = preds_array.reshape(y_test.shape)

    bias = np.mean(preds_array - y_test)
    mae = _sk_metrics().mean_absolute_error(y_test, preds_array)
    rmse = _sk_metrics().root_mean_squared_error(y_test, preds_array)
    r2 = _sk_metrics().r2_score(y_test, preds_array)
    mape = _sk_metrics().mean_absolute_percentage_error(y_test, preds_array) * 100

    return {
        'bias': round(bias, 3),
        'mae': round(mae, 3),
        'rmse': round(rmse, 3),
        'r2': round(r2, 3),
        'mape': round(mape, 3),
    }
