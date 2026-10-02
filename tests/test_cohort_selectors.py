"""Select real Iris model results without losing objective or trial identity."""
import numpy as np
import polars as pl
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from talos.cohort.sfc import select_diverse_metrics, select_pareto


@pytest.fixture(scope='module')
def iris_results():
    x, y = load_iris(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=.3, stratify=y, random_state=17)
    rows = []
    for identifier, regularization in enumerate(np.logspace(-5, 3, 12)):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=float(regularization), max_iter=1000, random_state=17))
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_val)
        rows.append({'id': identifier, 'accuracy': accuracy_score(y_val, probability.argmax(1)),
                     'loss': log_loss(y_val, probability), 'regularization': float(regularization)})
    return pl.from_dicts(rows)


def test_diversity_keeps_distinct_real_iris_score_regions_and_seed(iris_results):
    context = {'results': iris_results}
    options = {'target_count': 5, 'n_clusters': 2, 'n_components': 2,
               'metric_cols': ['accuracy', 'loss'], 'random_state': 19}
    selected = select_diverse_metrics(context, **options)
    assert selected == select_diverse_metrics(context, **options)
    assert len(selected) == len(set(selected)) == 5
    assert set(selected) <= set(iris_results['id'])
    chosen = iris_results.filter(pl.col('id').is_in(selected))
    assert chosen['loss'].max() - chosen['loss'].min() > .5
    assert chosen['accuracy'].min() < chosen['accuracy'].max()
    rescaled = iris_results.with_columns((pl.col('loss') * 1000).alias('loss'))
    assert select_diverse_metrics({'results': rescaled}, **options) == selected
    assert context['results'].equals(iris_results)


def test_diversity_handles_single_cluster_and_constant_metric(iris_results):
    frame = iris_results.with_columns(pl.lit(1.).alias('constant'))
    selected = select_diverse_metrics({'results': frame}, target_count=4, n_clusters=1,
                                     metric_cols=['loss', 'constant'])
    assert len(selected) == len(set(selected)) == 4
    assert set(selected) <= set(frame['id'])
    assert frame['loss'].arg_max() in selected


@pytest.mark.parametrize('selector', [select_diverse_metrics, select_pareto])
def test_selectors_drop_invalid_measurements_and_preserve_string_ids(iris_results, selector):
    valid = iris_results.head(2).with_columns(pl.Series('id', [' 001 ', ' research-trial ']))
    invalid = valid.head(1).with_columns(pl.lit('bad').alias('id'), pl.lit(float('nan')).alias('loss'))
    result = selector({'results': pl.concat([valid, invalid])}, target_count=10,
                      metric_cols=['accuracy', 'loss'])
    assert result
    assert 'bad' not in result
    assert set(result) <= {1, 'research-trial'}
    assert all(isinstance(identifier, (int, str)) for identifier in result)
    assert selector({'results': invalid}, target_count=10, metric_cols=['accuracy', 'loss']) == []


@pytest.mark.parametrize('selector', [select_diverse_metrics, select_pareto])
@pytest.mark.parametrize('bad_id', [True, None, '', float('nan')])
def test_selectors_reject_ambiguous_trial_identity(iris_results, selector, bad_id):
    frame = iris_results.head(1).with_columns(pl.Series('id', [bad_id]))
    with pytest.raises(ValueError, match='permutation id'):
        selector({'results': frame}, metric_cols=['accuracy', 'loss'])


def test_pareto_respects_loss_direction_and_is_independent_of_input_order(iris_results):
    options = {'target_count': 12, 'metric_cols': ['accuracy', 'loss'],
               'directions': {'accuracy': 'max', 'loss': 'min'}}
    selected = select_pareto({'results': iris_results}, **options)
    assert selected == select_pareto({'results': iris_results.reverse()}, **options)
    assert selected
    for identifier in selected:
        row = iris_results.filter(pl.col('id') == identifier).row(0, named=True)
        competitors = iris_results.filter((pl.col('accuracy') >= row['accuracy']) &
                                         (pl.col('loss') <= row['loss']) &
                                         ((pl.col('accuracy') > row['accuracy']) |
                                          (pl.col('loss') < row['loss'])))
        assert competitors.is_empty()
    assert select_pareto({'results': iris_results}, **dict(options, target_count=1)) == selected[:1]
    equivalent = iris_results.with_columns((-pl.col('loss')).alias('loss'))
    assert select_pareto({'results': equivalent}, target_count=12,
                         metric_cols=['accuracy', 'loss']) == selected


def test_pareto_equal_metrics_keep_identity_with_deterministic_ties(iris_results):
    same = pl.concat([iris_results.head(1)] * 3).with_columns(pl.Series('id', ['trial-c', 'trial-a', 'trial-b']))
    assert select_pareto({'results': same}, target_count=2,
                         metric_cols=['accuracy', 'loss']) == ['trial-a', 'trial-b']
    floats = iris_results.head(1).with_columns(pl.lit(2.).alias('id'))
    assert select_pareto({'results': floats}, metric_cols=['accuracy', 'loss']) == [2]


@pytest.mark.parametrize('selector,options,match', [
    (select_diverse_metrics, {'target_count': 0}, 'target_count'),
    (select_diverse_metrics, {'n_clusters': 0}, 'n_clusters'),
    (select_diverse_metrics, {'n_components': 0}, 'n_components'),
    (select_diverse_metrics, {'iqr_multiplier': -1}, 'iqr_multiplier'),
    (select_pareto, {'target_count': 0, 'metric_cols': ['loss']}, 'target_count'),
    (select_pareto, {}, 'metric_cols'),
    (select_pareto, {'metric_cols': ['loss'], 'directions': {'loss': 'sideways'}}, 'directions'),
])
def test_selectors_reject_invalid_scientific_configuration(iris_results, selector, options, match):
    with pytest.raises(ValueError, match=match):
        selector({'results': iris_results}, **options)


@pytest.mark.parametrize('selector', [select_diverse_metrics, select_pareto])
@pytest.mark.parametrize('context,match', [({}, 'requires results'),
                                         ({'results': []}, 'polars DataFrame')])
def test_selectors_require_results_tables(selector, context, match):
    with pytest.raises(ValueError, match=match):
        selector(context, metric_cols=['loss'])


def test_diversity_infers_only_observed_numeric_metrics(iris_results):
    frame = iris_results.drop('regularization')
    options = {'target_count': 3, 'n_clusters': 3, 'random_state': 19}
    assert select_diverse_metrics({'results': frame}, **options) == select_diverse_metrics(
        {'results': frame}, metric_cols=['accuracy', 'loss'], **options)
    with pytest.raises(ValueError, match='at least two numeric metric columns'):
        select_diverse_metrics({'results': frame.select('id', 'loss')})


@pytest.mark.parametrize('selector', [select_diverse_metrics, select_pareto])
def test_selectors_reject_missing_trial_or_metric_columns(iris_results, selector):
    with pytest.raises(ValueError, match='missing required columns'):
        selector({'results': iris_results.drop('id')}, metric_cols=['loss'])
    with pytest.raises(ValueError, match='missing required columns'):
        selector({'results': iris_results}, metric_cols=['unmeasured_loss'])
