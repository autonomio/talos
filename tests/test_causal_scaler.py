"""Verify trailing-window scaling on real Iris observations without future leakage."""
import numpy as np
import polars as pl
import pytest
from sklearn.datasets import load_iris

from talos.scalers.causal_rolling_robust_scaler import CausalRollingRobustScaler


def iris_frame():
    iris = load_iris()
    return pl.DataFrame({name: iris.data[:, index] for index, name in enumerate(
        ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])})


def test_causal_scaler_prefix_and_training_statistics_ignore_later_observations():
    data = iris_frame()
    train, heldout = data.head(80), data.tail(70)
    scaler = CausalRollingRobustScaler(train, window=10, min_samples=3)
    fitted = (dict(scaler.medians), dict(scaler.iqrs))
    complete = scaler.transform(heldout)
    prefix = scaler.transform(heldout.head(30))
    np.testing.assert_allclose(prefix.to_numpy(), complete.head(30).to_numpy())
    changed_future = pl.concat([heldout.head(30), heldout.tail(40) * 100])
    np.testing.assert_allclose(scaler.transform(changed_future).head(30).to_numpy(), prefix.to_numpy())
    assert (scaler.medians, scaler.iqrs) == fitted
    assert scaler.context_rows == 10
    assert complete.shape == heldout.shape
    assert np.isfinite(complete.to_numpy()).all()
    assert np.max(np.abs(complete.to_numpy())) <= 8
    expected_warmup = np.array([(heldout[name][0] - scaler.medians[name]) / scaler.iqrs[name]
                                for name in heldout.columns]).clip(-8, 8)
    np.testing.assert_allclose(complete.row(0), expected_warmup)


def test_current_observation_cannot_change_its_own_scaling_reference():
    data = iris_frame()
    heldout = data.tail(50)
    scaler = CausalRollingRobustScaler(data.head(100), window=8, min_samples=3, clip=1e6)
    baseline = scaler.transform(heldout)
    row = 20
    def alter(amount):
        return heldout.with_columns(pl.when(pl.int_range(pl.len()) == row)
                                    .then(pl.col('sepal_length') + amount)
                                    .otherwise(pl.col('sepal_length')).alias('sepal_length'))
    once = scaler.transform(alter(100))
    twice = scaler.transform(alter(200))
    below = scaler.transform(alter(-100))
    np.testing.assert_allclose(once.head(row).to_numpy(), baseline.head(row).to_numpy())
    delta = once['sepal_length'][row] - baseline['sepal_length'][row]
    assert delta > 0
    assert baseline['sepal_length'][row] - below['sepal_length'][row] == pytest.approx(delta)
    assert twice['sepal_length'][row] - baseline['sepal_length'][row] == pytest.approx(2 * delta)


def test_scaler_leaves_metadata_and_unfitted_columns_intact():
    real = iris_frame().head(20)
    frame = real.with_columns(pl.lit('Iris').alias('dataset'), pl.lit(None, dtype=pl.Float64).alias('missing'),
                              pl.lit(1.).alias('constant'))
    scaler = CausalRollingRobustScaler(frame, window=5, min_samples=2)
    output = scaler.transform(frame)
    assert output['dataset'].equals(frame['dataset'])
    assert output['missing'].equals(frame['missing'])
    assert output['constant'].to_list() == [0.] * len(frame)
    assert scaler.transform(frame.select('dataset')).equals(frame.select('dataset'))
    assert 'missing' not in scaler.medians


@pytest.mark.parametrize('options,match', [({'quantile_range': (.8, .2)}, 'quantile_range'),
                                         ({'window': 1}, 'window'), ({'clip': 0}, 'clip'),
                                         ({'window': 4, 'min_samples': 5}, 'min_samples')])
def test_scaler_rejects_unusable_statistical_configuration(options, match):
    with pytest.raises(ValueError, match=match):
        CausalRollingRobustScaler(iris_frame(), **options)
