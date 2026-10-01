from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import polars as pl


class CalibratorProtocol(Protocol):

    '''Protocol for probability calibration functions.'''

    def __call__(self,
                 clf: Any,
                 x_val: npt.NDArray[Any] | pl.DataFrame,
                 y_val: npt.NDArray[Any] | pl.Series,
                 **params: Any) -> Any:
        ...


class ThresholdOptimizerProtocol(Protocol):

    '''Protocol for threshold optimisation functions.'''

    def __call__(self,
                 y_val: npt.NDArray[Any] | pl.Series,
                 val_proba: npt.NDArray[np.floating[Any]],
                 **params: Any) -> tuple[float, float]:
        ...


class CalibrationConfigProtocol(Protocol):

    '''Structural type for a resolved calibration configuration.'''

    calibration_func: CalibratorProtocol | None
    calibration_params: dict[str, Any]
    threshold_func: ThresholdOptimizerProtocol | None
    threshold_params: dict[str, Any]


def fit_calibrator(model: Any,
                   config: CalibrationConfigProtocol,
                   x_val: npt.NDArray[Any] | pl.DataFrame,
                   y_val: npt.NDArray[Any] | pl.Series) -> tuple[Any, float | None, float | None]:

    '''
    Fit calibrator on validation data.

    Args:
        model (Any): Fitted classifier with predict_proba method
        config (CalibrationConfigProtocol): Resolved calibration configuration
        x_val (np.ndarray or pl.DataFrame): Validation features
        y_val (np.ndarray or pl.Series): Validation labels

    Returns:
        tuple: (fitted_calibrator, optimal_threshold, val_score)
            val_score is None when no threshold_func is configured
    '''

    fitted = (config.calibration_func(model, x_val, y_val, **config.calibration_params)
              if config.calibration_func is not None else model)
    probabilities = np.asarray(fitted.predict_proba(x_val))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError('Classifier must return class probabilities')
    if probabilities.shape[1] != 2:
        if config.threshold_func is not None:
            raise ValueError('Threshold optimization requires binary classification')
        return fitted, None, None
    val_proba = probabilities[:, 1]

    if config.threshold_func is not None:
        classes = np.asarray(getattr(fitted, 'classes_', [0, 1]))
        binary_labels = (np.asarray(y_val) == classes[1]).astype(np.int8)
        threshold, score = config.threshold_func(binary_labels, val_proba, **config.threshold_params)
    else:
        threshold, score = 0.5, None

    return fitted, threshold, score


def apply_calibrated_predict(model: Any,
                              config: CalibrationConfigProtocol,
                              data: dict[str, Any]) -> dict[str, Any]:

    '''
    Apply calibration and threshold optimisation to a fitted model's predictions.

    Args:
        model: Fitted classifier with predict_proba method
        config (CalibrationConfigProtocol): Resolved calibration configuration
        data (dict): Data dictionary with x_val, y_val, x_test keys

    Returns:
        dict: Results with '_preds', '_probs', 'optimal_threshold' and 'val_score'
            (val_score is None when no threshold_func is configured)
    '''

    fitted, threshold, score = fit_calibrator(model, config, data['x_val'], data['y_val'])
    probabilities = np.asarray(fitted.predict_proba(data['x_test']))
    classes = np.asarray(getattr(fitted, 'classes_', np.arange(probabilities.shape[1])))
    if threshold is None:
        preds = classes[probabilities.argmax(axis=1)]
        output_probabilities = probabilities
    else:
        output_probabilities = probabilities[:, 1]
        preds = classes[(output_probabilities >= threshold).astype(int)]
    return {'_preds': preds, '_probs': output_probabilities, 'optimal_threshold': threshold, 'val_score': score}
