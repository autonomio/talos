from pathlib import Path
import numpy as np
import pandas as pd


class Log:
    """Analyze generic experiment tables and explicitly supplied predictions."""
    from talos.log._experiment_parameter_correlation import experiment_parameter_correlation
    from talos.log._read_from_file import read_from_file

    def __init__(self, uel_object=None, file_path=None, *, predictions=None, targets=None,
                 inverse_scaler=None, cols_to_multilabel=None):
        if (uel_object is None) == (file_path is None):
            raise ValueError('Supply one experiment result or artifact path')
        if file_path is not None:
            path = Path(file_path)
            if path.is_dir():
                path /= 'results.csv'
            if path.suffix == '.parquet':
                import polars as pl
                self.experiment_log = pd.DataFrame(pl.read_parquet(path).to_dict(as_series=False))
            else:
                self.experiment_log = self.read_from_file(path)
            self.result = None
        else:
            self.result = uel_object
            table = getattr(uel_object, 'experiment_log', None)
            if table is None:
                table = uel_object.data
            self.experiment_log = pd.DataFrame(table.to_dict(as_series=False)) if hasattr(table, 'to_pandas') else table.copy()
        if cols_to_multilabel:
            self.experiment_log = pd.get_dummies(self.experiment_log, columns=cols_to_multilabel, dtype=int)
        self.data = self.experiment_log
        self.predictions, self.targets = predictions, targets
        self.inverse_scaler = inverse_scaler

    def permutation_prediction_performance(self, permutation_id=None, *, predictions=None,
                                           targets=None, task='classification', threshold=0.5):
        from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
        prediction, target = self._prediction_pair(permutation_id, predictions, targets)
        if task == 'regression':
            return {'mae': mean_absolute_error(target, prediction),
                    'mse': mean_squared_error(target, prediction), 'r2': r2_score(target, prediction)}
        if task == 'multilabel':
            labels = (prediction >= threshold).astype(int) if np.issubdtype(prediction.dtype, np.floating) else prediction
        elif prediction.ndim > 1 and prediction.shape[-1] > 1:
            labels = prediction.argmax(axis=-1)
        elif np.issubdtype(prediction.dtype, np.floating):
            labels = (prediction.reshape(-1) >= threshold).astype(int)
        else:
            labels = prediction.reshape(-1)
        if task != 'multilabel' and target.ndim > 1 and target.shape[-1] > 1:
            target = target.argmax(axis=-1)
        return {'accuracy': accuracy_score(target, labels)}

    def permutation_confusion_metrics(self, permutation_id=None, *, predictions=None,
                                     targets=None, labels=None, threshold=0.5, task='classification'):
        from sklearn.metrics import confusion_matrix, classification_report
        prediction, target = self._prediction_pair(permutation_id, predictions, targets)
        if task == 'multilabel':
            from sklearn.metrics import multilabel_confusion_matrix
            prediction = (prediction >= threshold).astype(int) if np.issubdtype(prediction.dtype, np.floating) else prediction
            return {'confusion_matrix': multilabel_confusion_matrix(target, prediction, labels=labels),
                    'report': classification_report(target, prediction, output_dict=True, zero_division=0)}
        if prediction.ndim > 1 and prediction.shape[-1] > 1:
            prediction = prediction.argmax(axis=-1)
        elif np.issubdtype(prediction.dtype, np.floating):
            prediction = (prediction.reshape(-1) >= threshold).astype(int)
        if target.ndim > 1 and target.shape[-1] > 1:
            target = target.argmax(axis=-1)
        return {'confusion_matrix': confusion_matrix(target.reshape(-1), prediction.reshape(-1), labels=labels),
                'report': classification_report(target.reshape(-1), prediction.reshape(-1), labels=labels,
                                                output_dict=True, zero_division=0)}

    def experiment_confusion_metrics(self, **kwargs):
        if not isinstance(self.predictions, dict):
            return self.permutation_confusion_metrics(**kwargs)
        return {key: self.permutation_confusion_metrics(key, **kwargs) for key in self.predictions}

    def _prediction_pair(self, permutation_id, predictions, targets):
        prediction = self.predictions if predictions is None else predictions
        target = self.targets if targets is None else targets
        if isinstance(prediction, dict):
            prediction = prediction[permutation_id]
        if isinstance(target, dict):
            target = target[permutation_id]
        if prediction is None or target is None:
            raise ValueError('Supply caller predictions and targets explicitly')
        return np.asarray(prediction), np.asarray(target)


    def data_quality(self, data=None):
        """Inspect caller tables/arrays for missing values, infinities and constant columns."""
        value = self.experiment_log if data is None else data
        if hasattr(value, 'to_numpy'):
            value = value.to_numpy()
        array = np.asarray(value)
        frame = pd.DataFrame(array if array.ndim > 1 else array.reshape(-1, 1))
        numeric = frame.select_dtypes(include=[np.number])
        return {'rows': len(frame), 'columns': len(frame.columns),
                'missing': int(frame.isna().sum().sum()),
                'infinite': int(np.isinf(numeric.to_numpy()).sum()),
                'constant_columns': [str(column) for column in frame if frame[column].nunique(dropna=True) <= 1]}

    def confusion_value_diagnostics(self, values, *, predictions=None, targets=None,
                                    permutation_id=None, threshold=0.5):
        """Compare caller measurements within TP/FP/TN/FN for each class."""
        from scipy.stats import ks_2samp
        prediction, target = self._prediction_pair(permutation_id, predictions, targets)
        if prediction.ndim > 1 and prediction.shape[-1] > 1:
            prediction = prediction.argmax(axis=-1)
        elif np.issubdtype(prediction.dtype, np.floating):
            prediction = (prediction.reshape(-1) >= threshold).astype(int)
        if target.ndim > 1 and target.shape[-1] > 1:
            target = target.argmax(axis=-1)
        value = np.asarray(values, dtype=float).reshape(-1)
        if len(value) != len(target):
            raise ValueError('Measurements must align with predictions and targets')
        rows = []
        for label in np.unique(target):
            truth, positive = target.reshape(-1) == label, prediction.reshape(-1) == label
            groups = {'tp': value[truth & positive], 'fp': value[~truth & positive],
                      'tn': value[~truth & ~positive], 'fn': value[truth & ~positive]}
            row = {'class': label}
            for name, values in groups.items():
                finite = values[np.isfinite(values)]
                row.update({name + '_count': len(finite),
                            name + '_mean': float(finite.mean()) if len(finite) else np.nan,
                            name + '_median': float(np.median(finite)) if len(finite) else np.nan})
            a, b = groups['tp'], groups['fp']
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2)) if len(a)>1 and len(b)>1 else np.nan
            row['tp_fp_cohen_d'] = float((a.mean()-b.mean())/pooled) if pooled > 0 else np.nan
            row['tp_fp_ks'] = float(ks_2samp(a,b).statistic) if len(a) and len(b) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)


    def prediction_table(self, permutation_id=None, *, predictions=None, targets=None,
                         observations=None, task='classification', threshold=0.5):
        """Return aligned caller predictions, targets and per-observation errors."""
        prediction, target = self._prediction_pair(permutation_id, predictions, targets)
        if task == 'classification' and prediction.ndim > 1 and prediction.shape[-1] > 1:
            prediction = prediction.argmax(axis=-1)
        elif task == 'classification' and np.issubdtype(prediction.dtype, np.floating):
            prediction = (prediction.reshape(-1) >= threshold).astype(int)
        if task == 'classification' and target.ndim > 1 and target.shape[-1] > 1:
            target = target.argmax(axis=-1)
        if len(prediction) != len(target):
            raise ValueError('Predictions and targets must align')
        frame = pd.DataFrame(index=range(len(target))) if observations is None else pd.DataFrame(observations).reset_index(drop=True)
        if len(frame) != len(target):
            raise ValueError('Caller observations must align with predictions')
        frame['predictions'] = list(prediction)
        frame['actuals'] = list(target)
        matches = prediction == target
        frame['hit'] = matches if matches.ndim == 1 else matches.all(axis=tuple(range(1, matches.ndim)))
        frame['miss'] = ~frame['hit']
        if task == 'regression':
            errors = prediction - target
            frame['absolute_error'] = np.abs(errors) if errors.ndim == 1 else np.abs(errors).mean(axis=tuple(range(1, errors.ndim)))
        return frame
