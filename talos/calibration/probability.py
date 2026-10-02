"""Fit scikit-learn probability calibration from caller validation data."""

from typing import Any, Protocol

import numpy.typing as npt
import polars as pl


class _Calibrator(Protocol):

    '''Typed facade over the sklearn CalibratedClassifierCV surface used here.'''

    def fit(self, X: Any, y: Any) -> Any: ...


def _calibrated_classifier_cv(clf: Any, method: str) -> _Calibrator:

    '''Create a frozen-estimator CalibratedClassifierCV behind the typed facade.'''

    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:
        return CalibratedClassifierCV(clf, method=method, cv='prefit')
    return CalibratedClassifierCV(FrozenEstimator(clf), method=method)


def sklearn_probability_calibrator(clf: Any,
                                    x_val: npt.NDArray[Any] | pl.DataFrame,
                                    y_val: npt.NDArray[Any] | pl.Series,
                                    method: str = 'isotonic') -> Any:

    '''
    Fit isotonic or sigmoid calibration on a pre-fitted classifier.

    Args:
        clf: Pre-fitted classifier with predict_proba method
        x_val (np.ndarray or pl.DataFrame): Validation features for calibration fitting
        y_val (np.ndarray or pl.Series): Validation labels for calibration fitting
        method (str): Calibration method ('isotonic' or 'sigmoid')

    Returns:
        Any: Fitted calibrated classifier with predict_proba method
    '''

    return _calibrated_classifier_cv(clf, method).fit(x_val, y_val)
