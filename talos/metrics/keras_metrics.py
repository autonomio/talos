"""Lazy backend-native metrics with dataset-level classification accumulation."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from keras.metrics import Metric
    matthews: Metric
    precision: Metric
    recall: Metric
    f1score: Metric


def _framework():
    import keras
    if hasattr(keras, 'ops'):
        return keras, keras.ops
    import tensorflow as tf
    class Ops:
        abs = staticmethod(tf.abs)
        square = staticmethod(tf.square)
        sqrt = staticmethod(tf.sqrt)
        mean = staticmethod(tf.reduce_mean)
        sum = staticmethod(tf.reduce_sum)
        maximum = staticmethod(tf.maximum)
        clip = staticmethod(tf.clip_by_value)
        log = staticmethod(tf.math.log)
        cast = staticmethod(tf.cast)
        argmax = staticmethod(tf.argmax)
        one_hot = staticmethod(tf.one_hot)
        reshape = staticmethod(tf.reshape)
        expand_dims = staticmethod(tf.expand_dims)
    return keras, Ops


def _continuous_pair(y_true, y_pred, ops):
    dtype = 'float64' if 'float64' in str(getattr(y_pred, 'dtype', '')) else 'float32'
    y_true, y_pred = ops.cast(y_true, dtype), ops.cast(y_pred, dtype)
    if len(y_pred.shape) == len(y_true.shape) + 1 and y_pred.shape[-1] == 1:
        y_true = ops.expand_dims(y_true, axis=-1)
    elif len(y_true.shape) == len(y_pred.shape) + 1 and y_true.shape[-1] == 1:
        y_pred = ops.expand_dims(y_pred, axis=-1)
    return y_true, y_pred


def mae(y_true, y_pred):
    _, ops = _framework()
    y_true, y_pred = _continuous_pair(y_true, y_pred, ops)
    return ops.mean(ops.abs(y_pred - y_true), axis=-1)


def mse(y_true, y_pred):
    _, ops = _framework()
    y_true, y_pred = _continuous_pair(y_true, y_pred, ops)
    return ops.mean(ops.square(y_pred - y_true), axis=-1)


def rmae(y_true, y_pred):
    _, ops = _framework()
    return ops.sqrt(mae(y_true, y_pred))


def rmse(y_true, y_pred):
    _, ops = _framework()
    return ops.sqrt(mse(y_true, y_pred))


def mape(y_true, y_pred):
    _, ops = _framework()
    y_true, y_pred = _continuous_pair(y_true, y_pred, ops)
    return 100 * ops.mean(ops.abs((y_true - y_pred) / ops.maximum(ops.abs(y_true), 1e-7)), axis=-1)


def msle(y_true, y_pred):
    _, ops = _framework()
    y_true, y_pred = _continuous_pair(y_true, y_pred, ops)
    return ops.mean(ops.square(ops.log(ops.maximum(y_pred, 1e-7) + 1) - ops.log(ops.maximum(y_true, 1e-7) + 1)), axis=-1)


def rmsle(y_true, y_pred):
    _, ops = _framework()
    return ops.sqrt(msle(y_true, y_pred))


def _classification_pair(y_true, y_pred, ops, count):
    if count > 1:
        predicted = ops.one_hot(ops.argmax(y_pred, axis=-1), count)
        if len(y_true.shape) > 1 and y_true.shape[-1] == count:
            actual = ops.one_hot(ops.argmax(y_true, axis=-1), count)
        else:
            actual = ops.one_hot(ops.cast(ops.reshape(y_true, (-1,)), 'int32'), count)
    else:
        actual = ops.cast(ops.reshape(ops.cast(y_true, 'float32'), (-1, 1)) >= .5, 'float32')
        predicted = ops.cast(ops.reshape(ops.cast(y_pred, 'float32'), (-1, 1)) >= .5, 'float32')
    return ops.cast(actual, 'float32'), ops.cast(predicted, 'float32')


def classification_metric(kind='f1score', beta=1, num_classes=None, name=None):
    if beta < 0:
        raise ValueError('beta must be nonnegative.')
    framework, ops = _framework()
    class ClassificationMetric(framework.metrics.Metric):
        def __init__(self):
            super().__init__(name=name or kind)
            self.beta = beta
            self.class_count = num_classes
            self.tp = self.fp = self.fn = None
            if self.class_count:
                self._build(self.class_count)

        def _build(self, count):
            self.class_count = count
            self.tp = self.add_weight(name='tp', shape=(count,), initializer='zeros')
            self.fp = self.add_weight(name='fp', shape=(count,), initializer='zeros')
            self.fn = self.add_weight(name='fn', shape=(count,), initializer='zeros')

        def update_state(self, y_true, y_pred, sample_weight=None):
            width = y_pred.shape[-1] if len(y_pred.shape) > 1 else 1
            if self.tp is None:
                self._build(int(width))
            actual, predicted = _classification_pair(y_true, y_pred, ops, self.class_count)
            weight = 1
            if sample_weight is not None:
                weight = ops.cast(ops.reshape(sample_weight, (-1, 1)), 'float32')
            self.tp.assign_add(ops.sum(weight * actual * predicted, axis=0))
            self.fp.assign_add(ops.sum(weight * (1 - actual) * predicted, axis=0))
            self.fn.assign_add(ops.sum(weight * actual * (1 - predicted), axis=0))

        def result(self):
            if self.tp is None:
                return 0.0
            precision = self.tp / ops.maximum(self.tp + self.fp, 1e-7)
            recall = self.tp / ops.maximum(self.tp + self.fn, 1e-7)
            if kind == 'precision':
                return ops.mean(precision)
            if kind == 'recall':
                return ops.mean(recall)
            if kind == 'matthews':
                # Binary requires negative-class counts too; represented by total below.
                predicted, actual = self.tp + self.fp, self.tp + self.fn
                if self.class_count == 1:
                    return (self.tp[0] * self.tn - self.fp[0] * self.fn[0]) / ops.maximum(ops.sqrt((self.tp[0] + self.fp[0]) * (self.tp[0] + self.fn[0]) * (self.tn + self.fp[0]) * (self.tn + self.fn[0])), 1e-7)
                total = ops.sum(actual)
                numerator = ops.sum(self.tp) * total - ops.sum(predicted * actual)
                denominator = ops.sqrt((total ** 2 - ops.sum(predicted ** 2)) * (total ** 2 - ops.sum(actual ** 2)))
                return numerator / ops.maximum(denominator, 1e-7)
            bb = self.beta ** 2
            return ops.mean((1 + bb) * precision * recall / ops.maximum(bb * precision + recall, 1e-7))

        def reset_state(self):
            for variable in self.variables:
                variable.assign(variable * 0)

        def get_config(self):
            return {**super().get_config(), 'kind': kind, 'beta': self.beta, 'num_classes': self.class_count}
    if kind == 'matthews':
        original_build = ClassificationMetric._build
        original_update = ClassificationMetric.update_state
        def build(self, count):
            original_build(self, count)
            self.tn = self.add_weight(name='tn', shape=(), initializer='zeros')
        def update(self, y_true, y_pred, sample_weight=None):
            original_update(self, y_true, y_pred, sample_weight)
            if self.class_count == 1:
                actual, predicted = _classification_pair(y_true, y_pred, ops, 1)
                actual, predicted = ops.reshape(actual, (-1,)), ops.reshape(predicted, (-1,))
                weight = 1 if sample_weight is None else ops.cast(ops.reshape(sample_weight, (-1,)), 'float32')
                self.tn.assign_add(ops.sum(weight * (1 - actual) * (1 - predicted)))
        ClassificationMetric._build = build
        ClassificationMetric.update_state = update
    return ClassificationMetric()


def fbeta(y_true, y_pred, beta=1):
    """Compute batch F-beta without variables; classification_metric accumulates across batches."""
    if beta < 0:
        raise ValueError('beta must be nonnegative.')
    _, ops = _framework()
    count = int(y_pred.shape[-1]) if len(y_pred.shape) > 1 else 1
    actual, predicted = _classification_pair(y_true, y_pred, ops, count)
    tp = ops.sum(actual * predicted, axis=0)
    fp = ops.sum((1 - actual) * predicted, axis=0)
    fn = ops.sum(actual * (1 - predicted), axis=0)
    bb = beta ** 2
    return ops.mean((1 + bb) * tp / ops.maximum((1 + bb) * tp + bb * fn + fp, 1e-7))


def __getattr__(name):
    if name in ('precision', 'recall', 'f1score', 'matthews'):
        return classification_metric(name)
    raise AttributeError(name)


__all__ = ['mae', 'mse', 'rmae', 'rmse', 'mape', 'msle', 'rmsle', 'matthews', 'precision', 'recall', 'fbeta', 'f1score', 'classification_metric']
