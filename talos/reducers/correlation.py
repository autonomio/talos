"""Prune legacy scan choices with unfavorable metric correlations."""

import math

import pandas as pd

from .reduce_utils import parameter_indicators


def correlation(self, method='spearman'):
    matrix, target, candidates = parameter_indicators(self)
    if len(target) < 3 or len(set(target)) < 2:
        return self
    best = None
    for index, candidate in enumerate(candidates):
        strength = pd.Series(matrix[:, index]).corr(pd.Series(target), method=method)
        unfavorable = strength if self.minimize_loss else -strength
        if math.isfinite(unfavorable) and unfavorable >= self.reduction_threshold and (best is None or unfavorable > best[0]):
            best = (unfavorable, candidate)
    if best is not None:
        self.param_object.remove_is(*best[1])
    return self
