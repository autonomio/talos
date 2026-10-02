"""Retain the logistic-regression scaler and inverse-transform exports."""

from talos.scalers.linear_scaler import LinearScaler, inverse_transform

__all__ = ['LinearScaler', 'LogRegScaler', 'inverse_transform']


class LogRegScaler(LinearScaler):
    """General linear scaling with caller rules; compatible public scaler name."""
