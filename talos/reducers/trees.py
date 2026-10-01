import numpy as np
from .reduce_utils import parameter_indicators


def _tree_reduce(self, forest=False, quantile=.8):
    matrix, target, candidates = parameter_indicators(self)
    if len(target) < 3 or matrix.shape[1] == 0 or len(set(target)) < 2:
        return self
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
    if forest:
        estimator = RandomForestRegressor(n_estimators=64, random_state=getattr(self, 'seed', None))
        labels = target
    else:
        estimator = ExtraTreesClassifier(n_estimators=64, random_state=getattr(self, 'seed', None))
        cutoff = np.quantile(target, 1 - quantile if self.minimize_loss else quantile)
        labels = target < cutoff if self.minimize_loss else target > cutoff
        if len(set(labels)) < 2:
            return self
    estimator.fit(matrix, labels)
    global_mean = target.mean()
    unwanted = []
    for index, (label, value) in enumerate(candidates):
        mean = target[matrix[:, index].astype(bool)].mean()
        worse = mean > global_mean if self.minimize_loss else mean < global_mean
        if worse:
            unwanted.append((estimator.feature_importances_[index], index))
    if unwanted:
        _, index = min(unwanted)
        self.param_object.remove_is(*candidates[index])
    return self


def trees(self, quantile=.8):
    return _tree_reduce(self, quantile=quantile)
